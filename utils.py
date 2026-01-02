import numpy as np
import pyvista as pv
import math
import json
import random
import pickle

def save_as_txt(filename, message):
    with open(filename, 'a') as file:
        if isinstance(message, dict):  # 检查 message 是否是字典类型
            json.dump(message, file)  # 将字典转换为 JSON 格式并写入文件
        elif isinstance(message, str):  # 如果是字符串
            file.write(message)  # 直接写入字符串
        else:
            print("Error: Cannot save_as_txt!")

def calculateR(v1, v2):
    '''Rodrigues 公式计算三维向量 v1 旋转到 v2 的旋转矩阵。'''
    # 转为 numpy 向量并归一化，避免 tuple 等类型无法除法
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    axis = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.linalg.norm(axis) < 1e-6:
        if dot > 0:  # 同向
            return np.eye(3)
        # 反向：选择任一与 v1 不平行的轴，旋转 180 度
        axis = np.cross(v1, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v1, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def R2T(R, v):
    '''
    将旋转矩阵扩展为位姿矩阵。
    R: 旋转矩阵。
    v: 平移向量。
    '''
    # 构造4*4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R  # 填入旋转部分
    x, y, z = v
    T[:3, 3] = [x, y, z]  # 填入平移部分
    return T


# TODO：旋转矩阵与该表示法下欧拉角的关系需要和机器人厂商确认。
def R2RPYEuler(R):
    '''将旋转矩阵R转换为RPY表示的欧拉角。'''
    # 提取旋转矩阵的元素
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    # 判断旋转矩阵是否有奇异情况（接近90度）
    singular = sy < 1e-6
    
    if not singular:
        # 计算Roll、Pitch、Yaw三个欧拉角（Rx、Ry、Rz）
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # 当出现奇异情况时，使用特殊计算方法
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    # 将欧拉角打包成数组返回
    RPYEuler = np.array([roll, pitch, yaw]) # 弧度制
    return np.degrees(RPYEuler) # 角度制

def RPYEuler2R(RPYEuler):
    '''
    将RPY表示的欧拉角转换为旋转矩阵R。
    
    参数:
    RPYEuler: 一个包含 Roll, Pitch, Yaw 的数组或列表 [roll, pitch, yaw]。
    
    返回:
    旋转矩阵 R (3x3 numpy array)。
    '''
    # 提取 Roll, Pitch, Yaw
    roll, pitch, yaw = np.radians(RPYEuler) # 角度制转化为弧度制

    # 计算各角的正弦和余弦值
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # 构造旋转矩阵
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ]
    ])
    return R

def pose2T(v):
    '''
    v: 6D pose
    '''
    trans_v = v[:3]
    rot_v = v[3:]

    R = RPYEuler2R(rot_v)
    T = R2T(R, trans_v)

    return T

def T2pose(T):
    trans_v = np.array([T[0][3], T[1][3], T[2][3]])
    rot_v = R2RPYEuler(T[:3, :3])
    pose = np.hstack((trans_v, rot_v))
    return pose

def calculateRPYEuler(v):
    '''计算一个三维向量对应的RPY欧拉角（向量指向视为相机坐标系的z轴，自旋为0）。'''
    return R2RPYEuler(calculateR((0,0,1), v))

def calculateT(position, euler_angle):
    '''计算从基坐标系到一个相机坐标系（由视点表示）的变换矩阵。'''
    R = RPYEuler2R(euler_angle)

    # 构造4*4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R  # 填入旋转部分
    x, y, z = position
    T[:3, 3] = [x, y, z]  # 填入平移部分
    return T

def calculateCorresPoint(T, P):
    '''
    已知从基坐标系到另一坐标系的变换矩阵T和另一坐标系中一点P(x1, y1, z1)，
    求点P在基坐标系中的坐标表示。
    
    参数:
    T: 4x4 的变换矩阵 (numpy array)
    P: 3x1 的点坐标 (list 或 numpy array)
    
    返回:
    点P在基坐标系中的坐标表示 [x2, y2, z2]
    '''
    # 将点P扩展为齐次坐标形式
    P_homogeneous = np.array([P[0], P[1], P[2], 1])
    
    # 计算基坐标系中的点
    P_transformed = np.dot(T, P_homogeneous)
    
    # 提取转换后的x2, y2, z2坐标（去掉齐次坐标的最后一维）
    x2, y2, z2 = P_transformed[:3]
    
    return [x2, y2, z2]

def plane_equation(p1, p2, p3):
    """
    通过三点确定平面方程 Ax + By + Cz + D = 0
    返回值：平面法向量normal和常数项D
    """
    # 计算两个向量
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    # 计算法向量
    normal = np.cross(v1, v2)
    # 计算D
    D = -np.dot(normal, p1)
    return (normal, D)

def if_same_side_of_plane(plane, point1, point2):
    """
    判断两个点是否在同一平面的一侧且没有点正好在平面上
    """
    (normal, D) = plane
    # 计算两个点代入平面方程的结果
    value1 = np.dot(normal, point1) + D
    value2 = np.dot(normal, point2) + D
    # 判断符号是否相同
    return value1 * value2 > 0

def if_point_in_body(point, vertices_top, vertices_bottom):
    """
    判断表面点是否在视域体内，True表示表面点在视域体内。
    """
    # 获取视域体中心点
    x_mid = (sum(vertice[0] for vertice in vertices_top) + sum(vertice[0] for vertice in vertices_bottom)) / 8.0
    y_mid = (sum(vertice[1] for vertice in vertices_top) + sum(vertice[1] for vertice in vertices_bottom)) / 8.0
    z_mid = (sum(vertice[2] for vertice in vertices_top) + sum(vertice[2] for vertice in vertices_bottom)) / 8.0
    p_mid = np.array([x_mid, y_mid, z_mid])
    # 获取视域体的六个面
    plane1 = plane_equation(vertices_top[0], vertices_top[1], vertices_top[2]) # 顶面
    plane2 = plane_equation(vertices_top[0], vertices_top[1], vertices_bottom[1])
    plane3 = plane_equation(vertices_top[1], vertices_top[2], vertices_bottom[2])
    plane4 = plane_equation(vertices_top[2], vertices_top[3], vertices_bottom[3])
    plane5 = plane_equation(vertices_top[3], vertices_top[0], vertices_bottom[0])
    plane6 = plane_equation(vertices_bottom[0], vertices_bottom[1], vertices_bottom[2]) # 底面
    # 判别点point是否在视域体内
    if if_same_side_of_plane(plane1, p_mid, point) == False: return False
    if if_same_side_of_plane(plane2, p_mid, point) == False: return False
    if if_same_side_of_plane(plane3, p_mid, point) == False: return False
    if if_same_side_of_plane(plane4, p_mid, point) == False: return False
    if if_same_side_of_plane(plane5, p_mid, point) == False: return False
    if if_same_side_of_plane(plane6, p_mid, point) == False: return False
    # 返回判别结果
    return True

def _precompute_frustum(vertices_top, vertices_bottom):
    """
    预计算可见性判断用的六个平面与中心点，便于向量化批量判断。
    返回 planes: (6,4) numpy 数组，每行 [nx, ny, nz, D]，以及中心点 p_mid。
    """
    x_mid = (sum(v[0] for v in vertices_top) + sum(v[0] for v in vertices_bottom)) / 8.0
    y_mid = (sum(v[1] for v in vertices_top) + sum(v[1] for v in vertices_bottom)) / 8.0
    z_mid = (sum(v[2] for v in vertices_top) + sum(v[2] for v in vertices_bottom)) / 8.0
    p_mid = np.array([x_mid, y_mid, z_mid])

    plane1 = plane_equation(vertices_top[0], vertices_top[1], vertices_top[2])  # 顶面
    plane2 = plane_equation(vertices_top[0], vertices_top[1], vertices_bottom[1])
    plane3 = plane_equation(vertices_top[1], vertices_top[2], vertices_bottom[2])
    plane4 = plane_equation(vertices_top[2], vertices_top[3], vertices_bottom[3])
    plane5 = plane_equation(vertices_top[3], vertices_top[0], vertices_bottom[0])
    plane6 = plane_equation(vertices_bottom[0], vertices_bottom[1], vertices_bottom[2])  # 底面

    planes = np.array([
        [*plane1[0], plane1[1]],
        [*plane2[0], plane2[1]],
        [*plane3[0], plane3[1]],
        [*plane4[0], plane4[1]],
        [*plane5[0], plane5[1]],
        [*plane6[0], plane6[1]]
    ], dtype=float)
    return planes, p_mid

def _points_in_frustum(points, planes, p_mid, eps=1e-9):
    """
    向量化判断点是否位于视锥体内（与 _precompute_frustum 配合使用）。
    points: (N,3)
    planes: (6,4) => nx,ny,nz,D
    返回 bool mask: (N,)
    """
    # 中心点在每个平面的符号，用来判断“同侧”
    mid_signs = planes[:, :3].dot(p_mid) + planes[:, 3]
    mid_signs = np.where(np.abs(mid_signs) < eps, np.sign(mid_signs) + eps, mid_signs)

    # 所有点到各平面的带符号距离（无需真实距离，只关心符号）
    signed = points.dot(planes[:, :3].T) + planes[:, 3]  # (N,6)
    return np.all(signed * mid_signs > 0, axis=1)

def if_point_occluded(model_mesh, point, viewpoint):
    """
    判断表面点是否被遮挡，False表示判断表面点没有被遮挡。

    :param model_mesh: 模型表面网格，是一个PolyData对象。
    :param point: 所考察表面点。
    :param viewpoint: 所考察视点。

    :return: 若返回True，则证明该点被遮挡；反之则无遮挡。
    """
    try:
        # 检查射线是否与网格相交
        intersection = model_mesh.ray_trace(viewpoint, point, first_point = True) # intersection[0]为表面点坐标
        if intersection is None or len(intersection[0]) == 0: # 如果没有交点，则不合理，发出警示并认为该点被遮挡
            return True
        if np.allclose(intersection[0], point, atol=1e-6): return False # 交点即为表面点，没有发生遮挡
        else: return True
    except (AttributeError, RuntimeError) as e:
        # 处理PyVista对象销毁时的错误，默认认为点被遮挡
        return True

def if_viewangle_under_threshold(n, point, viewpoint, threshold = 90.0):
    """
    判断表面点对应视角是否符合条件，True表示对应视角符合条件（即表面点对应法向量和表面点到视点的向量夹角小于给定阈值）。
    
    :param n: 表面点对应法向量。
    :param point: 所考察表面点。
    :param viewpoint: 所考察视点。
    :param threshold: 视角阈值，采用角度制。默认为90°，即为不加以限制。
    """
    v = viewpoint - point

    dot_product = np.dot(n, v) # 计算点积
    
    # 计算向量的模
    magnitude_n = np.linalg.norm(n)
    magnitude_v = np.linalg.norm(v)
    if magnitude_n * magnitude_v == 0: return False # 避免除零错误
    
    cos_theta = dot_product / (magnitude_n * magnitude_v) # 计算cos(theta)
    cos_theta = max(-1.0, min(1.0, cos_theta)) # 防止浮点数误差超出范围 [ -1, 1 ]
    theta_radians = math.acos(cos_theta) # 计算夹角（弧度）
    theta_degrees = math.degrees(theta_radians) # 将弧度转换为角度

    if theta_degrees < threshold: return True
    else: return False

def calculateVisibility(vp, vp_euler, model, model_mesh, vertices_top, vertices_bottom, points_visibility = None, mode = 0):
    """
    计算视点vp对应的model可见点数量。
    *mode=1时，会对输入参数中的points_visibility造成修改。

    :param vp: 视点位置。
    :param vp_euler: 视点欧拉角。
    :param model: 待重建表面模型点云，是一个存储了(point, normal)数据的列表。
    :param model_mesh: 待重建表面模型网格。
    :param vertices_top: 相机坐标系下视域体框架顶部顶点坐标。
    :param vertices_bottom: 相机坐标系下视域体框架底部顶点坐标。
    :param points_visibility: 当前表面点可见状态，一个Boolean量列表，长度与model相同。
    :param mode: 计算模式，默认为0，代表计算当前视点对应可见点数量；为1时计算新增可见点数量并修改points_visibility。

    :return: mode=0，返回视点vp对应的model可见点数量；mode=1，返回“新增可见点数量，数据冗余值”。
    """
    num_vis = 0  # 当前视点下的可见表面点数量

    # 统一为numpy数组，避免后续广播失败
    vp = np.asarray(vp, dtype=float)
    vp_euler = np.asarray(vp_euler, dtype=float)

    # 获取从世界坐标系到相机坐标系的变换矩阵
    T = calculateT(vp, vp_euler)

    # 将vertices转换到世界坐标系，并预计算平面（便于后续批量裁剪）
    vertices_top = [calculateCorresPoint(T, P) for P in vertices_top]
    vertices_bottom = [calculateCorresPoint(T, P) for P in vertices_bottom]
    planes, p_mid = _precompute_frustum(vertices_top, vertices_bottom)

    # 批量提取点与法向量
    points_arr = np.asarray([p for p, _ in model], dtype=float)
    normals_arr = np.asarray([n for _, n in model], dtype=float)

    # 1) 视锥裁剪（向量化）
    in_frustum_mask = _points_in_frustum(points_arr, planes, p_mid)

    # 2) 视角阈值（向量化）
    v_vec = vp - points_arr  # (N,3)
    mag_v = np.linalg.norm(v_vec, axis=1)
    mag_n = np.linalg.norm(normals_arr, axis=1)
    non_zero = (mag_v > 1e-12) & (mag_n > 1e-12)

    dot_nv = np.einsum("ij,ij->i", normals_arr, v_vec, optimize=True)
    safe_denom = mag_v * mag_n
    safe_denom = np.where(safe_denom < 1e-12, 1e-12, safe_denom)
    cos_theta = np.clip(dot_nv / safe_denom, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    angle_mask = theta_deg < 85.0

    # 综合初筛
    candidate_mask = in_frustum_mask & angle_mask & non_zero
    candidate_indices = np.nonzero(candidate_mask)[0]

    if mode == 0:
        for i in candidate_indices:
            point = points_arr[i]
            if if_point_occluded(model_mesh, point, vp):
                continue
            num_vis += 1
        return num_vis

    elif mode == 1:
        redun = 0  # 冗余可见点数
        for i in candidate_indices:
            point = points_arr[i]
            if if_point_occluded(model_mesh, point, vp):
                continue

            if points_visibility[i] == 0:  # 新增可见点
                num_vis += 1
                points_visibility[i] = 1
            elif points_visibility[i] == 1:  # 冗余可见点
                redun += 1
            else:
                print("非法的points_visibility[i]输入！")
                return -1  # 报错
        return num_vis, redun

    else:
        print("非法的mode输入！")
        return -1  # 报错
    
