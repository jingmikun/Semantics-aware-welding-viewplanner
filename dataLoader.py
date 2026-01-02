from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pyvista as pv


"""
数据加载模块
"""

CURRDIR = Path(__file__).resolve().parent
dataDIR = CURRDIR / "data"

def loadViewpoint():
    with open(dataDIR / "viewPoints.pkl", "rb") as f:
        viewpoints = pickle.load(f)
    return viewpoints

def loadModel():
    return pd.read_csv(dataDIR / "rawData.csv").to_numpy()

def loadMesh():
    mesh = pv.read(dataDIR / "weld_object_v3_surface_mesh.obj")
    return mesh

def downSampleViewpoint():
    """
    分桶等比例下采样：
      1) 按距离分桶（默认 5 个等宽区间）
      2) 桶内按体素分组，按比例抽取（默认每个体素保留 10%，至少 1 个）
    返回新的字典 {new_idx: (vp_coord, surf_point, normal, distance)}
    """
    voxel_size = 40.0    # 体素边长
    keep_ratio = 0.05    # 每个体素保留比例
    num_bins = 10        # 距离分桶数量

    viewpoints = loadViewpoint()
    dists = np.array([v[3] for v in viewpoints.values()])
    dist_min, dist_max = dists.min(), dists.max()
    bins = np.linspace(dist_min, dist_max, num_bins + 1)

    selected = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        idxs = [k for k, v in viewpoints.items() if b0 <= v[3] < b1]
        if len(idxs) == 0:
            continue

        cells = {}
        for k in idxs:
            coord = np.asarray(viewpoints[k][0], dtype=float)
            cell = tuple(np.floor(coord / voxel_size).astype(int))
            cells.setdefault(cell, []).append(k)

        for cell, keys in cells.items():
            n = len(keys)
            keep = max(1, int(np.ceil(keep_ratio * n)))
            choice = np.random.choice(keys, size=keep, replace=False)
            selected.extend(choice.tolist())

    down = {i: viewpoints[k] for i, k in enumerate(selected)}
    return down


def downSampleModel(voxel_size= 2.0):
    """
    对模型点云做体素下采样，返回 (N,7) ndarray，列为 (x,y,z,label,nx,ny,nz)。
    - 坐标按 voxel_size 体素分桶
    - 每个体素保留第一个点（简单策略，保留原 label/normal）
    """
    model = loadModel()
    coords = model[:, :3]
    labels = model[:, 3:4]
    normals = model[:, 4:7]

    cells = np.floor(coords / voxel_size).astype(int)
    seen = {}
    keep_idxs = []
    for i, cell in enumerate(map(tuple, cells)):
        if cell not in seen:
            seen[cell] = True
            keep_idxs.append(i)

    keep_idxs = np.array(keep_idxs, dtype=int)
    down_coords = coords[keep_idxs]
    down_labels = labels[keep_idxs]
    down_normals = normals[keep_idxs]
    down = np.hstack([down_coords, down_labels, down_normals])
    return down
