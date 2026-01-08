# @author Jingmikun
# @time 2026/1/2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class visualizer():
    def __init__(self):
        pass

    def paintGreedyTrajectory(self, trajectory):
        """
        给定一个trajectory组成类型，绘画出一些可用的图
        这个trajectory是一个列表，其中每个元素是一个字典，其中
        包含step, viewpoint, new_points, coverage等信息
        """

        trajectory_dfs = pd.DataFrame(trajectory)

        self.paintAvg(trajectory_dfs, 'step', 'new_points', "Time Step", "Viewpoint")
        self.paintAvg(trajectory_dfs, 'step', 'coverage', "Time Step", "Coverage /%")
        self.paintAvg(trajectory_dfs, 'step', 'weld_percent', "Time Step", "Weld Percentage /%")

    def paintAvg(self, data, xVal, yVal, xlabel, ylabel):
        """
        画出y轴为value，x轴为step的折线图，data为DataFrame格式，包含'step'和'value'列
        paintAvg返回的图像是浅色显示每次的轨迹，深色显示平均轨迹
        参数：
        data: DataFrame，包含 'step', 'value' 列
        xVal: x轴数据列名
        yVal: y轴数据列名
        xlabel: x轴标签
        ylabel: y轴标签
        """
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # 背景轨迹：按 run_id 区分，但全部统一为灰色
        hue_opts = {}
        if 'run_id' in data.columns:
            palette = {rid: 'grey' for rid in data['run_id'].unique()}
            hue_opts = {"hue": "run_id", "palette": palette, "legend": False}

        sns.lineplot(
            data=data,
            x=xVal,
            y=yVal,
            estimator=None,
            alpha=0.4,        # 透明度体现波动
            linewidth=1.5,
            zorder=1,         # 底层
            **hue_opts
        )

        sns.lineplot(
            data=data,
            x=xVal,
            y=yVal,
            color='firebrick', # 醒目的颜色
            linewidth=3,       # 均值线加粗
            label='Average Performance',
            errorbar=None,     # 关闭自动阴影
            zorder=2           # 图层置于顶层
        )

        # 3. 装饰图表
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()

        # 保存图像到 Logs/greedy/Figures 目录
        out_dir = Path("Logs/greedy/Figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"greedy_trajectory_{yVal}.png")
        plt.close()

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
        
    def saveFigure(self, filename):
        """
        保存当前图形到文件
        """
        plt.savefig(filename)
