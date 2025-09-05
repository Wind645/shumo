# Project quick start

Run a quick demo (package entrypoint):
	- python -m simcore
	- uv run -m simcore

Why the previous error?
Running a module file inside a package directly (e.g. `python simcore/simulator.py`) starts it as a script, not as part of the `simcore` package, so relative imports like `from .constants import ...` fail with "attempted relative import with no known parent package".
Use `-m package.module` or `-m package` (when `__main__.py` exists) so Python loads it as a package and relative imports work.

Optimization API (Problems 2–5): see `api/`.
Quick test: `python -c "from api.problems import evaluate_problem2; import numpy as np; print(evaluate_problem2(120.0, azimuth=np.arctan2(0.0,-1.0)))"`

Folder constraints
Each file <= ~300 lines (kept).
Keep root items small; dev-only scripts were moved to `examples/` if needed.
