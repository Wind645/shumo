"""
参数重要性分析模块
=================

该模块提供多种方法来分析优化问题中参数的重要性，帮助识别：
1. 哪些参数对目标函数影响最大（核心参数）
2. 哪些参数几乎不影响结果（冗余参数）
3. 参数之间的交互效应

支持的分析方法：
- 敏感性分析 (Sensitivity Analysis)
- Sobol敏感性指数 (Sobol Indices)
- 随机森林特征重要性 (Random Forest Feature Importance)
- 数值梯度分析 (Numerical Gradient Analysis)
- 方差分析 (Variance Analysis)

使用示例：
    from abstract.parameter_importance import ParameterImportanceAnalyzer

    analyzer = ParameterImportanceAnalyzer(
        problem=3,
        bombs_count=3,
        evaluation_method="rough_caps_torch"
    )

    result = analyzer.analyze_importance(
        methods=['sensitivity', 'sobol', 'gradient'],
        n_samples=1000
    )

    print("参数重要性排名:")
    for param, importance in result.ranking:
        print(f"{param}: {importance:.4f}")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import numpy as np
import json
from pathlib import Path
import warnings
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from api.problems import evaluate_problem2, evaluate_problem3, evaluate_problem4, evaluate_problem5
from optimizer.spec import bounds_for_problem, decode_vector

# NOTE (router integration):
# The current parameter importance pipeline calls the high-level evaluation functions above,
# which internally perform per-frame occlusion judging. For large batches you can
# replace those calls (especially for Problem 3) with the router-based single-pass
# vectorized evaluation provided in `abstract.occlusion_eval.evaluate_q3_router`
# (built on `judges.occlusion_time_router`). When doing so:
#   - Map occlusion methods:
#       judge_caps_torch -> torch_exact
#       rough_caps_torch -> torch_rough
#       judge_caps       -> numpy_exact
#       rough_caps       -> numpy_rough
#   - Import: `from abstract.occlusion_eval import evaluate_q3_router`
#   - Swap only the Problem 3 branch to keep other problems unaffected.
# This keeps output structure consistent while reducing Python loop overhead.

# ---------------------------- 数据结构 ----------------------------

@dataclass
class ParameterInfo:
    """参数信息"""
    name: str
    index: int
    bounds: Tuple[float, float]
    description: str = ""

@dataclass
class ImportanceResult:
    """单个重要性分析方法的结果"""
    method: str
    values: np.ndarray  # 每个参数的重要性分数
    std: Optional[np.ndarray] = None  # 标准差（如果适用）
    confidence_interval: Optional[np.ndarray] = None  # 置信区间

@dataclass
class AnalysisResult:
    """完整的参数重要性分析结果"""
    parameter_names: List[str]
    parameter_info: List[ParameterInfo]
    method_results: Dict[str, ImportanceResult]
    ranking: List[Tuple[str, float]]  # 按重要性排序的参数列表
    summary: Dict[str, Any]

    def get_top_parameters(self, n: int = 5, method: str = 'combined') -> List[Tuple[str, float]]:
        """获取最重要的n个参数"""
        if method == 'combined':
            return self.ranking[:n]
        elif method in self.method_results:
            indices = np.argsort(self.method_results[method].values)[::-1]
            return [(self.parameter_names[i], self.method_results[method].values[i]) for i in indices[:n]]
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_redundant_parameters(self, threshold: float = 0.01, method: str = 'combined') -> List[Tuple[str, float]]:
        """获取冗余参数（重要性低于阈值）"""
        if method == 'combined':
            return [(name, score) for name, score in self.ranking if score < threshold]
        elif method in self.method_results:
            values = self.method_results[method].values
            return [(self.parameter_names[i], values[i]) for i in range(len(values)) if values[i] < threshold]
        else:
            raise ValueError(f"Unknown method: {method}")

# ---------------------------- 核心分析器 ----------------------------

class ParameterImportanceAnalyzer:
    """参数重要性分析器"""

    def __init__(
        self,
        problem: int,
        bombs_count: int = 3,
        evaluation_method: str = "rough_caps_torch",
        dt: float = 0.02,
        seed: Optional[int] = None
    ):
        self.problem = problem
        self.bombs_count = bombs_count
        self.evaluation_method = evaluation_method
        self.dt = dt
        self.seed = seed

        # 设置随机数生成器
        self.rng = np.random.default_rng(seed)

        # 获取参数边界和信息
        self.bounds = bounds_for_problem(problem, bombs_count)
        self.dim = len(self.bounds)
        self.parameter_info = self._create_parameter_info()
        self.parameter_names = [p.name for p in self.parameter_info]

        # 评估函数缓存
        self._eval_cache: Dict[tuple, float] = {}

    def _create_parameter_info(self) -> List[ParameterInfo]:
        """创建参数信息列表"""
        info_list = []

        if self.problem == 2:
            info_list = [
                ParameterInfo("speed", 0, self.bounds[0], "飞行速度 (m/s)"),
                ParameterInfo("azimuth", 1, self.bounds[1], "航向角 (rad)"),
                ParameterInfo("release_time", 2, self.bounds[2], "投放时间 (s)"),
                ParameterInfo("explode_delay", 3, self.bounds[3], "起爆延迟 (s)")
            ]
        elif self.problem == 3:
            info_list = [
                ParameterInfo("speed", 0, self.bounds[0], "飞行速度 (m/s)"),
                ParameterInfo("azimuth", 1, self.bounds[1], "航向角 (rad)")
            ]
            for i in range(self.bombs_count):
                info_list.extend([
                    ParameterInfo(f"bomb_{i+1}_time", 2 + 2*i, self.bounds[2 + 2*i], f"炸弹{i+1}投放时间 (s)"),
                    ParameterInfo(f"bomb_{i+1}_delay", 2 + 2*i + 1, self.bounds[2 + 2*i + 1], f"炸弹{i+1}起爆延迟 (s)")
                ])
        elif self.problem == 4:
            drone_names = ["FY1", "FY2", "FY3"]
            for i, drone in enumerate(drone_names):
                base_idx = i * 4
                info_list.extend([
                    ParameterInfo(f"{drone}_speed", base_idx, self.bounds[base_idx], f"{drone}飞行速度 (m/s)"),
                    ParameterInfo(f"{drone}_azimuth", base_idx + 1, self.bounds[base_idx + 1], f"{drone}航向角 (rad)"),
                    ParameterInfo(f"{drone}_time", base_idx + 2, self.bounds[base_idx + 2], f"{drone}投放时间 (s)"),
                    ParameterInfo(f"{drone}_delay", base_idx + 3, self.bounds[base_idx + 3], f"{drone}起爆延迟 (s)")
                ])
        elif self.problem == 5:
            drone_names = ["FY1", "FY2", "FY3", "FY4", "FY5"]
            for i, drone in enumerate(drone_names):
                base_idx = i * (2 + 2 * self.bombs_count)
                info_list.extend([
                    ParameterInfo(f"{drone}_speed", base_idx, self.bounds[base_idx], f"{drone}飞行速度 (m/s)"),
                    ParameterInfo(f"{drone}_azimuth", base_idx + 1, self.bounds[base_idx + 1], f"{drone}航向角 (rad)")
                ])
                for j in range(self.bombs_count):
                    bomb_idx = base_idx + 2 + 2 * j
                    info_list.extend([
                        ParameterInfo(f"{drone}_bomb_{j+1}_time", bomb_idx, self.bounds[bomb_idx], f"{drone}炸弹{j+1}投放时间 (s)"),
                        ParameterInfo(f"{drone}_bomb_{j+1}_delay", bomb_idx + 1, self.bounds[bomb_idx + 1], f"{drone}炸弹{j+1}起爆延迟 (s)")
                    ])

        return info_list

    def _evaluate_objective(self, x: np.ndarray, use_cache: bool = True) -> float:
        """评估目标函数（返回负遮蔽时间用于最小化）"""
        x_tuple = tuple(x.tolist())

        if use_cache and x_tuple in self._eval_cache:
            return self._eval_cache[x_tuple]

        try:
            # 解码参数
            decision = decode_vector(self.problem, x, self.bombs_count)
            if decision.invalid:
                result = 1e12  # 惩罚无效解
            else:
                # 选择评估函数
                if self.problem == 2:
                    eval_result = evaluate_problem2(
                        dt=self.dt,
                        occlusion_method=self.evaluation_method,
                        **decision.to_eval_kwargs()
                    )
                elif self.problem == 3:
                    eval_result = evaluate_problem3(
                        dt=self.dt,
                        occlusion_method=self.evaluation_method,
                        **decision.to_eval_kwargs()
                    )
                elif self.problem == 4:
                    eval_result = evaluate_problem4(
                        dt=self.dt,
                        occlusion_method=self.evaluation_method,
                        **decision.to_eval_kwargs()
                    )
                elif self.problem == 5:
                    eval_result = evaluate_problem5(
                        dt=self.dt,
                        occlusion_method=self.evaluation_method,
                        **decision.to_eval_kwargs()
                    )
                else:
                    raise ValueError(f"Unsupported problem: {self.problem}")

                # 提取遮蔽时间
                if self.problem <= 4:
                    occlusion = eval_result["occluded_time"]["M1"]
                else:
                    occlusion = eval_result["total"]

                result = -float(occlusion)  # 负号用于最小化

        except Exception as e:
            result = 1e12  # 评估失败时的惩罚

        if use_cache:
            self._eval_cache[x_tuple] = result

        return result

    def _generate_samples(self, n_samples: int, method: str = 'lhs') -> np.ndarray:
        """生成样本点"""
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])

        if method == 'lhs':
            # Latin Hypercube Sampling
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)
            unit_samples = sampler.random(n_samples)
            samples = lo + unit_samples * (hi - lo)
        elif method == 'sobol':
            # Sobol序列
            sampler = qmc.Sobol(d=self.dim, seed=self.seed)
            unit_samples = sampler.random(n_samples)
            samples = lo + unit_samples * (hi - lo)
        elif method == 'random':
            # 纯随机采样
            samples = self.rng.random((n_samples, self.dim))
            samples = lo + samples * (hi - lo)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return samples

    def sensitivity_analysis(
        self,
        n_samples: int = 500,
        perturbation: float = 0.1,
        base_point: Optional[np.ndarray] = None
    ) -> ImportanceResult:
        """敏感性分析：扰动单个参数观察目标函数变化"""

        if base_point is None:
            # 使用随机基准点
            base_point = self._generate_samples(1, 'lhs')[0]

        # 评估基准点
        base_value = self._evaluate_objective(base_point)

        sensitivities = np.zeros(self.dim)

        for i in range(self.dim):
            # 生成扰动
            perturbations = []
            lo, hi = self.bounds[i]
            span = hi - lo

            # 正负扰动
            for sign in [-1, 1]:
                perturbed = base_point.copy()
                delta = sign * perturbation * span
                perturbed[i] = np.clip(perturbed[i] + delta, lo, hi)
                perturbations.append(perturbed)

            # 计算敏感性
            perturbed_values = [self._evaluate_objective(p) for p in perturbations]

            # 使用数值导数的绝对值
            if len(perturbed_values) == 2:
                derivative = abs((perturbed_values[1] - perturbed_values[0]) / (2 * perturbation * span))
            else:
                derivative = abs(np.mean(perturbed_values) - base_value) / (perturbation * span)

            sensitivities[i] = derivative

        # 归一化
        if np.max(sensitivities) > 0:
            sensitivities = sensitivities / np.max(sensitivities)

        return ImportanceResult('sensitivity', sensitivities)

    def sobol_analysis(self, n_samples: int = 1000) -> ImportanceResult:
        """Sobol敏感性分析"""
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol
        except ImportError:
            warnings.warn("SALib not installed, using simplified Sobol analysis")
            return self._simplified_sobol_analysis(n_samples)

        # 定义问题
        problem = {
            'num_vars': self.dim,
            'names': self.parameter_names,
            'bounds': self.bounds
        }

        # 生成样本
        param_values = saltelli.sample(problem, n_samples)

        # 评估
        Y = np.array([self._evaluate_objective(x) for x in param_values])

        # Sobol分析
        Si = sobol.analyze(problem, Y)

        # 使用一阶敏感性指数
        first_order = Si['S1']
        first_order_conf = Si['S1_conf']

        return ImportanceResult('sobol', first_order, confidence_interval=first_order_conf)

    def _simplified_sobol_analysis(self, n_samples: int) -> ImportanceResult:
        """简化的Sobol分析（不依赖SALib）"""
        # 生成两组独立样本
        samples_A = self._generate_samples(n_samples, 'sobol')
        samples_B = self._generate_samples(n_samples, 'sobol')

        # 评估样本
        Y_A = np.array([self._evaluate_objective(x) for x in samples_A])
        Y_B = np.array([self._evaluate_objective(x) for x in samples_B])

        # 计算总方差
        Y_all = np.concatenate([Y_A, Y_B])
        var_total = np.var(Y_all)

        if var_total < 1e-12:
            return ImportanceResult('sobol_simplified', np.zeros(self.dim))

        first_order = np.zeros(self.dim)

        for i in range(self.dim):
            # 创建混合样本：除第i个参数外都来自B，第i个参数来自A
            samples_AB_i = samples_B.copy()
            samples_AB_i[:, i] = samples_A[:, i]

            Y_AB_i = np.array([self._evaluate_objective(x) for x in samples_AB_i])

            # 计算一阶敏感性指数
            mean_A = np.mean(Y_A)
            cov = np.mean(Y_B * (Y_AB_i - Y_A))
            first_order[i] = cov / var_total

        # 确保非负
        first_order = np.maximum(first_order, 0)

        return ImportanceResult('sobol_simplified', first_order)

    def random_forest_analysis(self, n_samples: int = 2000) -> ImportanceResult:
        """随机森林特征重要性分析"""
        # 生成样本
        X = self._generate_samples(n_samples, 'lhs')
        y = np.array([self._evaluate_objective(x) for x in X])

        # 过滤有效样本
        valid_mask = np.isfinite(y) & (y < 1e10)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) < 50:
            warnings.warn("Too few valid samples for Random Forest analysis")
            return ImportanceResult('random_forest', np.zeros(self.dim))

        # 训练随机森林
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=self.seed,
            max_depth=10,
            min_samples_split=5
        )

        rf.fit(X_valid, y_valid)

        # 获取特征重要性
        importance = rf.feature_importances_

        return ImportanceResult('random_forest', importance)

    def gradient_analysis(self, n_samples: int = 200, epsilon: float = 1e-6) -> ImportanceResult:
        """数值梯度分析"""
        # 生成样本点
        samples = self._generate_samples(n_samples, 'lhs')

        gradients = []

        for sample in samples:
            gradient = np.zeros(self.dim)
            base_value = self._evaluate_objective(sample)

            if not np.isfinite(base_value) or base_value > 1e10:
                continue

            for i in range(self.dim):
                lo, hi = self.bounds[i]

                # 前向差分
                perturbed = sample.copy()
                h = min(epsilon * (hi - lo), hi - sample[i])
                perturbed[i] += h

                forward_value = self._evaluate_objective(perturbed)

                if np.isfinite(forward_value) and forward_value < 1e10:
                    gradient[i] = abs((forward_value - base_value) / h)

            gradients.append(gradient)

        if not gradients:
            return ImportanceResult('gradient', np.zeros(self.dim))

        # 计算平均梯度幅值
        mean_gradient = np.mean(gradients, axis=0)

        # 归一化
        if np.max(mean_gradient) > 0:
            mean_gradient = mean_gradient / np.max(mean_gradient)

        return ImportanceResult('gradient', mean_gradient)

    def variance_analysis(self, n_samples: int = 1000) -> ImportanceResult:
        """方差分析：固定其他参数，变化单个参数观察方差"""
        # 生成基准样本
        base_samples = self._generate_samples(n_samples // 10, 'lhs')

        variances = np.zeros(self.dim)

        for i in range(self.dim):
            param_variances = []

            for base_sample in base_samples:
                # 固定其他参数，只变化第i个参数
                lo, hi = self.bounds[i]
                param_values = np.linspace(lo, hi, 20)

                objective_values = []
                for param_val in param_values:
                    test_sample = base_sample.copy()
                    test_sample[i] = param_val
                    obj_val = self._evaluate_objective(test_sample)

                    if np.isfinite(obj_val) and obj_val < 1e10:
                        objective_values.append(obj_val)

                if len(objective_values) > 1:
                    param_variances.append(np.var(objective_values))

            if param_variances:
                variances[i] = np.mean(param_variances)

        # 归一化
        if np.max(variances) > 0:
            variances = variances / np.max(variances)

        return ImportanceResult('variance', variances)

    def analyze_importance(
        self,
        methods: List[str] = ['sensitivity', 'sobol', 'random_forest', 'gradient'],
        n_samples: int = 1000,
        verbose: bool = True
    ) -> AnalysisResult:
        """完整的参数重要性分析"""

        if verbose:
            print(f"开始参数重要性分析 (Problem {self.problem}, {self.dim}维)")
            print(f"参数列表: {self.parameter_names}")
            print(f"分析方法: {methods}")

        method_results = {}

        # 执行各种分析方法
        for method in methods:
            if verbose:
                print(f"\n执行 {method} 分析...")

            try:
                if method == 'sensitivity':
                    result = self.sensitivity_analysis(n_samples=n_samples//2)
                elif method == 'sobol':
                    result = self.sobol_analysis(n_samples=n_samples//4)
                elif method == 'random_forest':
                    result = self.random_forest_analysis(n_samples=n_samples)
                elif method == 'gradient':
                    result = self.gradient_analysis(n_samples=n_samples//5)
                elif method == 'variance':
                    result = self.variance_analysis(n_samples=n_samples//2)
                else:
                    warnings.warn(f"Unknown method: {method}")
                    continue

                method_results[method] = result

                if verbose:
                    top_params = np.argsort(result.values)[::-1][:3]
                    print(f"  前3个重要参数: {[self.parameter_names[i] for i in top_params]}")

            except Exception as e:
                warnings.warn(f"Error in {method} analysis: {e}")
                continue

        # 综合排名
        if method_results:
            combined_scores = self._combine_scores(method_results)
            ranking = sorted(
                zip(self.parameter_names, combined_scores),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            ranking = [(name, 0.0) for name in self.parameter_names]

        # 生成摘要
        summary = {
            'total_parameters': self.dim,
            'methods_used': list(method_results.keys()),
            'top_3_parameters': [name for name, _ in ranking[:3]],
            'redundant_parameters': [name for name, score in ranking if score < 0.1],
            'analysis_quality': self._assess_analysis_quality(method_results)
        }

        if verbose:
            print(f"\n=== 分析结果摘要 ===")
            print(f"最重要的3个参数: {summary['top_3_parameters']}")
            print(f"可能冗余的参数: {summary['redundant_parameters']}")
            print(f"分析质量: {summary['analysis_quality']}")

        return AnalysisResult(
            parameter_names=self.parameter_names,
            parameter_info=self.parameter_info,
            method_results=method_results,
            ranking=ranking,
            summary=summary
        )

    def _combine_scores(self, method_results: Dict[str, ImportanceResult]) -> np.ndarray:
        """综合多种方法的重要性分数"""
        if not method_results:
            return np.zeros(self.dim)

        # 方法权重
        method_weights = {
            'sensitivity': 1.0,
            'sobol': 1.5,  # Sobol分析更可靠
            'sobol_simplified': 1.2,
            'random_forest': 1.0,
            'gradient': 0.8,
            'variance': 0.8
        }

        combined = np.zeros(self.dim)
        total_weight = 0

        for method, result in method_results.items():
            weight = method_weights.get(method, 1.0)

            # 归一化分数
            scores = result.values.copy()
            if np.max(scores) > 0:
                scores = scores / np.max(scores)

            combined += weight * scores
            total_weight += weight

        if total_weight > 0:
            combined = combined / total_weight

        return combined

    def _assess_analysis_quality(self, method_results: Dict[str, ImportanceResult]) -> str:
        """评估分析质量"""
        if len(method_results) == 0:
            return "差"
        elif len(method_results) == 1:
            return "一般"
        elif len(method_results) <= 2:
            return "良好"
        else:
            # 检查方法间的一致性
            scores_matrix = np.array([result.values for result in method_results.values()])
            correlations = np.corrcoef(scores_matrix)
            mean_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])

            if mean_correlation > 0.7:
                return "优秀"
            elif mean_correlation > 0.5:
                return "良好"
            else:
                return "一般"

    def save_result(self, result: AnalysisResult, filepath: str):
        """保存分析结果"""
        data = {
            'parameter_names': result.parameter_names,
            'parameter_info': [
                {
                    'name': p.name,
                    'index': p.index,
                    'bounds': p.bounds,
                    'description': p.description
                }
                for p in result.parameter_info
            ],
            'method_results': {
                method: {
                    'method': res.method,
                    'values': res.values.tolist(),
                    'std': res.std.tolist() if res.std is not None else None,
                    'confidence_interval': res.confidence_interval.tolist() if res.confidence_interval is not None else None
                }
                for method, res in result.method_results.items()
            },
            'ranking': result.ranking,
            'summary': result.summary,
            'meta': {
                'problem': self.problem,
                'bombs_count': self.bombs_count,
                'evaluation_method': self.evaluation_method,
                'dim': self.dim
            }
        }

        Path(filepath).write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------- 便捷函数 ----------------------------

def analyze_problem_importance(
    problem: int,
    bombs_count: int = 3,
    methods: List[str] = ['sensitivity', 'sobol', 'random_forest'],
    n_samples: int = 1000,
    evaluation_method: str = "rough_caps_torch",
    verbose: bool = True,
    save_path: Optional[str] = None
) -> AnalysisResult:
    """便捷函数：分析指定问题的参数重要性"""

    analyzer = ParameterImportanceAnalyzer(
        problem=problem,
        bombs_count=bombs_count,
        evaluation_method=evaluation_method
    )

    result = analyzer.analyze_importance(
        methods=methods,
        n_samples=n_samples,
        verbose=verbose
    )

    if save_path:
        analyzer.save_result(result, save_path)
        if verbose:
            print(f"结果已保存到: {save_path}")

    return result


# ---------------------------- 演示和测试 ----------------------------

def _demo():
    """演示参数重要性分析"""
    print("=== 参数重要性分析演示 ===")

    # 分析问题3的参数重要性
    print("\n分析问题3（单无人机3枚炸弹）的参数重要性...")

    result = analyze_problem_importance(
        problem=3,
        bombs_count=3,
        methods=['sensitivity', 'gradient', 'variance'],
        n_samples=200,  # 演示用小样本
        verbose=True
    )

    print("\n=== 详细结果 ===")
    print("参数重要性排名:")
    for i, (param, importance) in enumerate(result.ranking, 1):
        print(f"{i:2d}. {param:<20} {importance:.4f}")

    print(f"\n核心参数 (前5): {[name for name, _ in result.get_top_parameters(5)]}")
    print(f"冗余参数: {[name for name, _ in result.get_redundant_parameters(0.1)]}")

    # 展示各方法的结果
    print("\n各方法的结果对比:")
    for method, method_result in result.method_results.items():
        print(f"\n{method}:")
        top_indices = np.argsort(method_result.values)[::-1][:3]
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. {result.parameter_names[idx]}: {method_result.values[idx]:.4f}")


if __name__ == "__main__":
    _demo()
