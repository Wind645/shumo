import numpy as np
def calculate_distance(point1, point2, point3):
    """
    这个方法接受三个参数，三个参数都是numpy array 要求是1X3的坐标，前
    两个点用来确定一条直线，然后计算第三个点到这条直线的距离
    """
    # 计算方向向量
    direction = point2 - point1
    # 计算向量从point1到point3
    vector_to_point = point3 - point1
    # 计算叉积
    cross_product = np.cross(vector_to_point, direction)
    # 计算距离
    distance = np.linalg.norm(cross_product) / np.linalg.norm(direction)
    return distance

def is_sight_hidden(r, distance):
    return r > distance