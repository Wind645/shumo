from api import *  # re-export compact API

if __name__ == "__main__":
    import numpy as np
    res = evaluate_problem2(
        speed=120.0,
        azimuth=np.arctan2(0.0, -1.0),
        release_time=1.5,
        explode_delay=3.6,
        occlusion_method="judge_caps",
        dt=0.05,
    )
    print("M1 遮蔽时长(s):", res["occluded_time"]["M1"], "总计(s):", res["total"])
