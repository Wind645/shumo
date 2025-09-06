# Project quick start

Run a quick demo (package entrypoint):
	- python -m simcore
	- uv run -m simcore

Why the previous error?
Running a module file inside a package directly (e.g. `python simcore/simulator.py`) starts it as a script, not as part of the `simcore` package, so relative imports like `from .constants import ...` fail with "attempted relative import with no known parent package".
Use `-m package.module` or `-m package` (when `__main__.py` exists) so Python loads it as a package and relative imports work.

Optimization API (Problems 2–5): see `api/`.
Quick test: `python -c "from api.problems import evaluate_problem2; import numpy as np; print(evaluate_problem2(120.0, azimuth=np.arctan2(0.0,-1.0)))"`

## Run optimizers with uv

Install deps (first time):
```
uv sync
```

Examples:
```
uv run -m optimizer.cli --problem 2 --algo sa --iters 2000 -v
uv run -m optimizer.cli --problem 3 --algo ga --bombs 3 --gens 60 -m judge_caps
uv run -m optimizer.cli --problem 4 --algo pso --iters 150 -m sampling
uv run -m optimizer.cli --problem 5 --algo sa --bombs 2 --iters 800 -m vectorized_sampling
```

Flags:
 - `--problem/-p` 2|3|4|5
 - `--algo/-a` sa|ga|pso|hybrid (hybrid 仅支持题 2)
 - `--bombs/-b` 每架无人机炸弹数 (题3/5 使用)
 - `--method/-m` 遮蔽判定: sampling | judge_caps | vectorized_sampling
 - `--backend` judge_caps 的加速后端 (rough / vectorized / vectorized_torch_sampled 等，根据算法实现)
 - `--iters` / `--gens` 覆盖默认迭代次数
 - `--seed` 固定随机种子
 - `-v` 详细日志


Folder constraints
Each file <= ~300 lines (kept).
Keep root items small; dev-only scripts were moved to `examples/` if needed.
