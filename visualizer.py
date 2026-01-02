# @author Jingmikun
# @time 2026/1/2



class visualizer():
    def __init__(self):
        pass

    def paintTrajectory(trajectory):
        pass

    def paintCloud(self, cloud):
        """
        使用 Open3D 可视化点云。cloud 为 (N,7) ndarray，列为 (x,y,z,label,nx,ny,nz)。
        label 映射为颜色（简单随机映射），法向用于法线显示。
        """
        import numpy as np
        import open3d as o3d

        coords = np.asarray(cloud[:, :3], dtype=float)
        labels = cloud[:, 3].astype(int) if cloud.shape[1] > 3 else None
        normals = cloud[:, 4:7] if cloud.shape[1] >= 7 else None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(float))

        if labels is not None:
            # 简单的标签着色：根据 label 生成可重复的颜色表
            uniq = np.unique(labels)
            colors = np.zeros((labels.size, 3), dtype=float)
            rng = np.random.default_rng(42)
            palette = {lb: rng.random(3) for lb in uniq}
            for i, lb in enumerate(labels):
                colors[i] = palette[lb]
            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud",
                                          width=800, height=600)
