# @author Jingmikun
# @time 2025/12/22

import numpy as np
import random
import open3d as o3d
from utils import *

"""
这是一个环境类，用于视点规划任务
这个环境类是任务本身的环境，也就是说它包含了所有算法可能会用到的数据和通用方法
所以这个类和继承与Gymnasium的强化学习环境不一样，在强化学习过程中也会用到这个类的object的一些信息
"""

class viewPlanningEnv():
    """
    这是视点规划的环境类，主要是用来保存视点和模型的信息和一些工具函数
    """
    def __init__(self, model_points, viewpoint, model_mesh):
        """
        保存模型和视点列表信息
        参数：
            model: 一个（N，7）的ndarray，列分别为（x,y,z,label,nx,ny,nz）
            viewpoints: 采样得到的视点坐标, 以一个字典的形式储存，其中
                key: 视点的索引 (int)
                value: (采样点坐标, 原始点, 法向量，距离）的元组，其中采样点坐标、原始点坐标、法向量为一个3维ndarray，距离是一个标量
            model_mesh: 一个pyvista对象，用于可见性计算
            _vp_kd = None        # KDTreeFlann 或 None
            _vp_coords = None    # (N,3)
            _vp_payload = None   # 与 coords 对齐的 (idx, surf_point, normal, distance)
        """ 
        self.model_points = model_points
        self.viewpoint = viewpoint  # dict: {idx: (vp_coord, surf_point, normal, distance)}
        self.model_mesh = model_mesh
        self.pointVisibility = self.get_init_points_visibility()
        self._vp_kd = None        # KDTreeFlann 或 None
        self._vp_coords = None    # (N,3)
        self._vp_payload = None   # 与 coords 对齐的 (idx, surf_point, normal, distance)

    def getCoordinates(self):
        """
        返回model里面的坐标信息
        """
        return self.model_points[:, :3]
    
    def getLabel(self):
        """
        返回model里面的标签信息
        """
        return self.model_points[:, 3]

    def getNormals(self):
        """
        返回model里面的法向量信息
        """
        return self.model_points[:, 4:7]
    
    def getPrepared(self):
        """
        为了进入calculateVisibility函数，把model调整成一个存储了(point, normal)数据的列表
        """
        return [(point, normal) for point, normal in zip(self.getCoordinates(), self.getNormals())]

    def get_init_points_visibility(self):
        """
        获取初始points_visibility。

        :return: 返回全为0的列表字典points_visibility。
        """
        points_visibility = [0] * len(self.model_points)
        return points_visibility
    
    def calculate_points_by_viewpoint(self, points_visibility, index,
                                      vertices_top = [ [105., -80., 250.], [-105., -80., 250.],
                                                       [-105., 80., 250.], [105., 80., 250.] ],
                                      vertices_bottom = [ [367.5, -245., 750.], [-367.5, -245., 750.],
                                                          [-367.5, 245., 750.], [367.5, 245., 750.] ]                 
                                      ):
        """
        根据新视点计算表面点可见状态，返回"新视点新增可见点数量，新视点数据冗余值"，会对输入参数中的points_visibility造成修改。

        :param points_visibility: 当前表面点可见状态，一个Boolean量列表。
        :param index: 新视点索引。
        :param vertices_top: 相机坐标系下各视场的边界点（顶层，250）。
        :param vertices_bottom: 相机坐标系下各视场的边界点（底层，750）。

        为了使用之前写的函数，还是要先处理一下self.pointVisibility
        """
        return calculateVisibility(vp = self.viewpoint[index][0],
                                   vp_euler = R2RPYEuler(calculateR((0,0,1),-self.viewpoint[index][2])),
                                   model = self.getPrepared(),
                                   model_mesh = self.model_mesh,
                                   vertices_top = vertices_top,
                                   vertices_bottom = vertices_bottom,
                                   points_visibility = points_visibility,
                                   mode = 1)

    def calculate_points_by_viewpoint_6Dpose(self, points_visibility, vp,
                                             vertices_top = [ [105., -80., 250.], [-105., -80., 250.],
                                                              [-105., 80., 250.], [105., 80., 250.] ],
                                             vertices_bottom = [ [367.5, -245., 750.], [-367.5, -245., 750.],
                                                                 [-367.5, 245., 750.], [367.5, 245., 750.] ]                 
                                            ):
        """
        根据新视点6Dpose计算表面点可见状态，返回"新视点新增可见点数量，新视点数据冗余值"，会对输入参数中的points_visibility造成修改。

        :param points_visibility: 当前表面点可见状态，一个Boolean量列表。
        :param vp: 新视点6D pose。
        :param vertices_top: 相机坐标系下各视场的边界点（顶层，250）。
        :param vertices_bottom: 相机坐标系下各视场的边界点（底层，750）。
        """
        return calculateVisibility(vp = vp[:3],
                                   vp_euler = vp[3:],
                                   model = self.getPrepared(),
                                   model_mesh = self.model_mesh,
                                   vertices_top = vertices_top,
                                   vertices_bottom = vertices_bottom,
                                   points_visibility = points_visibility,
                                   mode = 1)

    def change_viewpoint(self, index, code, points_visibility, lim_dis = 100):
        """
        根据code来改变视点，返回"新视点索引，新视点新增可见点数量，新视点数据冗余值"，会对输入参数中的points_visibility造成修改。

        :param index: 当前视点索引。
        :param code: 一个范围在0~8之间的整数，用于表示视点的变化方向。
        :param points_visibility: 当前表面点可见状态，一个Boolean量列表。
        :param lim_dis: 每次改变视点时限制的距离阈值。

        :return: 返回"新视点索引，新视点新增可见点数量，新视点数据冗余值"。

        整体的逻辑是先用code作掩码，然后再构建小范围的KDTree进行搜索。

        """
        (x0, y0, z0) = self.viewpoint[index][0] # 获取当前视点位置
        
        if code == 8: # code=8，不改变当前视点
            return index, *self.calculate_points_by_viewpoint(points_visibility, index)
        
        code = format(code, '03b') # 将code转换为3位二进制数
        code = [int(bit) for bit in code] # 解析二进制数的每一位值，并转换为整数

        # 延迟构建全量缓存
        if self._vp_coords is None:
            self._vp_coords = np.vstack([v[0] for v in self.viewpoint.values()])
            self._vp_payload = [(k, v[1], v[2], v[3]) for k, v in self.viewpoint.items()]

        # 利用 code 先做方向掩码，缩小候选，再建局部 KDTree
        coords = self._vp_coords
        payload = self._vp_payload
        mask_x = coords[:,0] >= x0 if code[0] == 1 else coords[:,0] < x0
        mask_y = coords[:,1] >= y0 if code[1] == 1 else coords[:,1] < y0
        mask_z = coords[:,2] >= z0 if code[2] == 1 else coords[:,2] < z0
        mask = mask_x & mask_y & mask_z

        if not np.any(mask):
            # 方向上没有候选，直接退回当前视点
            return index, *self.calculate_points_by_viewpoint(points_visibility, index)

        sub_coords = coords[mask]
        sub_payload = [p for m, p in zip(mask, payload) if m]

        # 在掩码后的子集上建临时 KDTree 半径查询
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub_coords)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        nearest_points = []
        k, idxs, dists = kdtree.search_radius_vector_3d([x0, y0, z0], lim_dis)
        for idx, dist2 in zip(idxs, dists):
            distance = float(np.sqrt(dist2))
            if distance < 1e-6:
                continue
            coords_tuple = tuple(sub_coords[idx])
            data = sub_payload[idx]
            nearest_points.append((distance, coords_tuple, data))
        nearest_points.sort(key=lambda x: x[0])

        if len(nearest_points) == 0: # 如果在给定距离lim_dis内没有近邻点，则不改变当前视点
            return index, *self.calculate_points_by_viewpoint(points_visibility, index)

        # 方向掩码已在构建阶段应用，这里只需返回最近一个
        if len(nearest_points) == 0:
            return index, *self.calculate_points_by_viewpoint(points_visibility, index)
        distance, coords_tuple, data = nearest_points[0]
        return data[0], *self.calculate_points_by_viewpoint(points_visibility, data[0])

    