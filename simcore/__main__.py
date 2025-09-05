from __future__ import annotations

"""
Convenience entrypoint so you can run:

  python -m simcore
  uv run -m simcore

This avoids the 'attempted relative import with no known parent package'
error that happens when executing a package module file directly.
"""

from sim_core import run_problem1


def main():
    out = run_problem1(dt=0.05, occlusion_method="judge_caps", verbose=True)
    print(f"[simcore] 有效遮蔽总时长 ≈ {out['occluded_time']:.3f} s")


if __name__ == "__main__":
    main()
