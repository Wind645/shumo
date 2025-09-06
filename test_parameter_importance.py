#!/usr/bin/env python3
"""
参数重要性分析测试脚本
===================

用于测试和验证parameter_importance模块的功能
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from abstract.parameter_importance import ParameterImportanceAnalyzer, analyze_problem_importance


def test_basic_analysis():
    """基本分析测试"""
    print("=== 基本分析测试 ===")
    
    # 测试问题2（较简单，4个参数）
    print("\n1. 测试问题2（4参数）...")
    analyzer = ParameterImportanceAnalyzer(problem=2, bombs_count=1)
    
    # 使用少量样本进行快速测试
    result = analyzer.analyze_importance(
        methods=['sensitivity', 'gradient'],
        n_samples=50,
        verbose=True
    )
    
    print(f"参数数量: {len(result.parameter_names)}")
    print(f"参数名称: {result.parameter_names}")
    print(f"分析方法: {list(result.method_results.keys())}")
    print(f"重要性排名: {[(name, f'{score:.3f}') for name, score in result.ranking[:3]]}")
    
    return result


def test_problem3_analysis():
    """测试问题3的参数重要性分析"""
    print("\n=== 问题3分析测试 ===")
    
    result = analyze_problem_importance(
        problem=3,
        bombs_count=2,  # 使用2枚炸弹减少复杂度
        methods=['sensitivity', 'variance'],
        n_samples=100,
        verbose=True
    )
    
    print("\n分析结果:")
    print(f"总参数数: {result.summary['total_parameters']}")
    print(f"核心参数: {result.summary['top_3_parameters']}")
    print(f"可能冗余: {result.summary.get('redundant_parameters', [])}")
    
    return result


def test_individual_methods():
    """测试各个分析方法"""
    print("\n=== 单独方法测试 ===")
    
    analyzer = ParameterImportanceAnalyzer(problem=2, bombs_count=1)
    
    methods_to_test = ['sensitivity', 'gradient', 'variance']
    
    for method in methods_to_test:
        print(f"\n测试 {method} 方法...")
        try:
            if method == 'sensitivity':
                result = analyzer.sensitivity_analysis(n_samples=20)
            elif method == 'gradient':
                result = analyzer.gradient_analysis(n_samples=20)
            elif method == 'variance':
                result = analyzer.variance_analysis(n_samples=50)
            
            print(f"  重要性值: {result.values}")
            print(f"  最重要参数: {analyzer.parameter_names[np.argmax(result.values)]}")
            
        except Exception as e:
            print(f"  方法 {method} 测试失败: {e}")


def test_save_load():
    """测试保存和加载功能"""
    print("\n=== 保存加载测试 ===")
    
    # 进行分析
    result = analyze_problem_importance(
        problem=2,
        bombs_count=1,
        methods=['sensitivity'],
        n_samples=30,
        verbose=False,
        save_path="test_importance_result.json"
    )
    
    # 检查文件是否存在
    if Path("test_importance_result.json").exists():
        print("结果文件保存成功")
        
        # 读取并验证
        import json
        with open("test_importance_result.json") as f:
            data = json.load(f)
        
        print(f"保存的参数名称: {data['parameter_names']}")
        print(f"保存的排名: {data['ranking'][:2]}")
        
        # 清理测试文件
        Path("test_importance_result.json").unlink()
        print("测试文件已清理")
    else:
        print("保存测试失败")


def compare_methods():
    """比较不同方法的结果一致性"""
    print("\n=== 方法一致性测试 ===")
    
    analyzer = ParameterImportanceAnalyzer(problem=3, bombs_count=2)
    
    # 获取多种方法的结果
    methods = ['sensitivity', 'gradient', 'variance']
    results = {}
    
    for method in methods:
        try:
            if method == 'sensitivity':
                results[method] = analyzer.sensitivity_analysis(n_samples=50)
            elif method == 'gradient':
                results[method] = analyzer.gradient_analysis(n_samples=30)
            elif method == 'variance':
                results[method] = analyzer.variance_analysis(n_samples=50)
        except Exception as e:
            print(f"方法 {method} 失败: {e}")
    
    # 比较结果
    if len(results) >= 2:
        print("\n方法间的排名对比:")
        for method, result in results.items():
            top_params = np.argsort(result.values)[::-1][:3]
            top_names = [analyzer.parameter_names[i] for i in top_params]
            print(f"{method:12}: {top_names}")
        
        # 计算相关性
        method_names = list(results.keys())
        if len(method_names) >= 2:
            values1 = results[method_names[0]].values
            values2 = results[method_names[1]].values
            correlation = np.corrcoef(values1, values2)[0, 1]
            print(f"\n{method_names[0]} vs {method_names[1]} 相关性: {correlation:.3f}")


def main():
    """主测试函数"""
    print("开始参数重要性分析模块测试...")
    
    try:
        # 基本功能测试
        test_basic_analysis()
        
        # 问题3测试
        test_problem3_analysis()
        
        # 单独方法测试
        test_individual_methods()
        
        # 保存加载测试
        test_save_load()
        
        # 方法比较测试
        compare_methods()
        
        print("\n=== 所有测试完成 ===")
        print("如果看到这条消息，说明基本功能正常工作！")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
