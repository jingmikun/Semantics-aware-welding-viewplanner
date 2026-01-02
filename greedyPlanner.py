# @author Jingmikun
# @time 2026/1/2
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

    def train(self, threshold=0.95, radius=150.0, max_candidates=200, max_workers=8):
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

        loop = tqdm(total=threshold, desc="Greedy coverage", unit="cov", leave=True)
        loop.update(coverage)

        while coverage < threshold:
            # 应用当前视点，更新可见性
            delta, redun = self.env.calculate_points_by_viewpoint(points_visibility, current)
            coverage = sum(points_visibility) / len(points_visibility)
            loop.n = coverage
            loop.refresh()

            trajectory[step] = {
                "viewpoint": current,
                "new_points": delta,
                "redundant": redun,
                "coverage": coverage
            }
            step += 1

            # 选择下一个视点
            nxt = self.choose_next_viewpoint(points_visibility,
                                             center_index=current,
                                             radius=radius,
                                             max_candidates=max_candidates,
                                             max_workers=max_workers)
            # 若没有更好的候选，结束
            if nxt == current:
                break
            current = nxt
        loop.close()
        return trajectory

    def choose_next_viewpoint(self, points_visibility, center_index, radius=150.0,
                              max_candidates=200):
        """
        在中心视点半径内选择“新增可见点数”最大的视点（无并行，仅向量化筛选候选）。
        不修改传入的 points_visibility。
        """
        # 坐标缓存
        if not hasattr(self, "_coords"):
            self._coords = np.vstack([v[0] for v in self.env.viewpoint.values()])
            self._indices = np.array(list(self.env.viewpoint.keys()))

        center_coord = self.env.viewpoint[center_index][0]
        dists = np.linalg.norm(self._coords - center_coord, axis=1)
        # 设最小间隔，避免选到几乎重合的视点；远优先
        mask = (dists < radius) & (dists > 50.0)
        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0:
            return center_index

        # 截断“最远的若干”候选（距离降序）
        cand_order = np.argsort(-dists[cand_idx])
        cand_idx = cand_idx[cand_order[:max_candidates]]
        candidate_ids = self._indices[cand_idx]

        # 顺序评估增益（无并行）
        best_id, best_gain, best_redun = center_index, 0, 0
        for vp_id in candidate_ids:
            pv_copy = points_visibility.copy()
            gain, redun = self.env.calculate_points_by_viewpoint(pv_copy, vp_id)
            if gain > best_gain or (gain == best_gain and redun < best_redun):
                best_id, best_gain, best_redun = vp_id, gain, redun

        return best_id if best_gain > 0 else center_index
