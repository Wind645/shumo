import numpy as np

MAX = 100000

Circle1 = np.asarray(0, 200, 0) # 圆柱体底面
Circle2 = np.asarray(0, 200, 10) # 圆柱体顶面
r = 7

R = 10
sphere = np.random.randn(1, 3) * MAX
while sphere[0, 2] <= 0:
    sphere = np.random.randn(1, 3) * MAX

