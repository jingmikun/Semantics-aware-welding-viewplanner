# @author Jingmikun
# @time 2026/1/2
import random
import numpy as np
import copy

from viewPointTaskEnv import viewPlanningEnv
from tqdm import tqdm

"""
基于贪婪策略的视点选择器：
- 在当前视点附近选取候选（按半径 + 距离最近截断）
- 并行评估“新增可见点数”，选增益最大的视点
"""

class GreedyViewpointPlanner:
    def __init__(self, env: viewPlanningEnv):
        self.env = env

    def train(self, iteration=20):
        """
        运行贪婪视点规划，直到覆盖率达到 threshold。
        返回轨迹字典：{step: {"viewpoint": idx, "new_points": delta, "coverage": cov}}
        """
        points_visibility = self.env.get_init_points_visibility()
        coverage = 0.0
        trajectory = {}
        step = 1

        # 随机起点
        current = random.choice(list(self.env.viewpoint.keys()))

        loop = tqdm(total=iteration, desc="Greedy training", unit="iter", leave=True)
        loop.set_postfix({"coverage": f"{coverage:.2%}"})

        while step <= iteration:
            # 应用当前视点，更新可见性
            delta, redun = self.env.calculate_points_by_viewpoint(points_visibility, current)
            coverage = sum(points_visibility) / len(points_visibility)
            loop.update(1)
            loop.set_postfix({"coverage": f"{coverage:.2%}"})

            trajectory[step] = {
                "viewpoint": current,
                "new_points": delta,
                "redundant": redun,
                "coverage": coverage
            }
            step += 1

            # 选择下一个视点
            nxt = self.choose_next_viewpoint(points_visibility, current, step)
            # 若没有更好的候选，结束
            if nxt == current:
                break
            current = nxt
        loop.close()
        return trajectory

    def choose_next_viewpoint(self, points_visibility, index, step, lr=10, r=405):
        """
        通过梯度上升的方法，贪婪地选择下一个视点
        """
        copyPointsVisibility = copy.deepcopy(points_visibility)
        if step != 1:
            v1, num_vis_best = self.findNearestBestViewpoint(index, copyPointsVisibility, r)
            num_vis_center, _ = self.env.calculate_points_by_viewpoint(copy.deepcopy(points_visibility), index)
            vref = self.gradientAscent(v1, index, lr,
                                       num_vis_best=num_vis_best,
                                       num_vis_center=num_vis_center,
                                       step=step)
        else:
            vref = index

        nxt, _ = self.findNearestBestViewpoint(vref, points_visibility, r)
        return nxt

    def findNearestBestViewpoint(self, index, points_visibility, r):
        """
        一个寻找距离最近而可视增益最大的视点的函数
        """
        listNewVP = self.buildNearestList(index, r)

        # 遍历得到新增可见点数量最多的下一个候选视点
        best_idx = -1
        best_vis = -1

        for cand in listNewVP:
            num_vis, redun = self.env.calculate_points_by_viewpoint(copy.deepcopy(points_visibility), cand)
            if num_vis > best_vis:
                best_idx, best_vis = cand, num_vis

        if best_idx == -1:
            best_idx, best_vis = index, 0

        return best_idx, best_vis

    def buildNearestList(self, index, r):
        """
        以 index 视点为球心、半径 r 画球，按“候选视点到球面”的距离升序选出最接近的 10 个视点索引。

        距离定义：|‖vp_i - center‖ - r|，即点到球面的距离（而非到球心的距离）。
        使用 numpy 向量化一次性计算，避免 Python for 循环。
        """
        # 拉取所有视点坐标 (N,3)
        vp_coords = np.vstack([v[0] for v in self.env.viewpoint.values()])
        center = vp_coords[index]

        # 计算每个候选视点到球面的距离
        radial_dist = np.linalg.norm(vp_coords - center, axis=1)
        surface_dist = np.abs(radial_dist - r)

        # 排除自身，按距离排序后取前 10
        order = np.argsort(surface_dist)
        nearest_indices = [i for i in order if i != index][:10]
        return nearest_indices


    def gradientAscent(self, v_best, v_center, lr, num_vis_best=0, num_vis_center=0, step=0):
        """
        根据中心视点与最优视点的可见增益差，沿 (v_best - v_center) 方向做一步梯度上升，返回参考视点索引。
        """
        vp = np.array(self.env.viewpoint[v_best][0])
        vp_center = np.array(self.env.viewpoint[v_center][0])

        diff = vp - vp_center
        dist = np.linalg.norm(diff)

        if dist == 0:
            direction = np.zeros_like(diff)
            l0 = 30
        else:
            direction = diff / dist
            l0 = np.clip(lr * (num_vis_best - num_vis_center) / dist, 0, 30)

        print(f"[s{step}] l: {l0}")

        vp_ref = vp + l0 * direction

        # 找到距离参考点最近的现有视点索引
        vp_coords = np.vstack([v[0] for v in self.env.viewpoint.values()])
        nearest_idx = int(np.argmin(np.linalg.norm(vp_coords - vp_ref, axis=1)))
        return nearest_idx
    