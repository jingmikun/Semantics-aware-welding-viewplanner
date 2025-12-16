#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import open3d as o3d
from pathlib import Path
import numpy as np
import pandas as pd


def viewPointExtractor():
    """读取点云与法向量并生成候选视点列表。
    
    Returns:
        viewPoints (List of array): 一个列表,其中每一个元素是一个ndarray,元素分别是(x,y,z,nx,ny,nz)
    """
    points, normals = loadRawData()
    viewPoints = gaussianSample(points, normals)
    return viewPoints


def loadRawData():
    """从 data/rawData.csv 读取点云坐标与法向量。"""
    CURRDIR = Path(__file__).resolve().parent

    dataPath = CURRDIR.parent / 'data' / 'rawData.csv'
    rawdata = pd.read_csv(dataPath)

    points = rawdata[['x', 'y', 'z']].to_numpy()
    normals = rawdata[['nx', 'ny', 'nz']].to_numpy()

    return points, normals


def gaussianSample(points,
                  normals,
                  num_samples: int = 4,
                  std: float = 250.0 / 3,
                  mean: float = 500.0):
    """
    在每个点的法向线上按高斯距离分布采样视点。

    参数：
        points (np.ndarray): 形状 (N, 3) 的点坐标数组。
        normals (np.ndarray): 形状 (N, 3) 的法向量数组。
        num_samples (int): 每个点除固定 500mm 外随机采样的数量。
        mean (float): 采样距离的均值，默认 500mm。
        std  (float): 采样距离的标准差，默认 250/3 mm。

    返回：
        sampled_points (np.ndarray): 采样得到的视点坐标，约 (N*(num_samples+1), 3)。
        points_with_normals (list[tuple]): [(原始点, 法向量), ...] 列表，长度为 N。
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    num_points = points.shape[0]

    # 构造 (N, S) 距离矩阵：第一列固定 500mm，其余列为高斯随机采样
    fixed_col = np.full((num_points, 1), mean, dtype=float)
    random_cols = np.random.normal(loc=mean, scale=std, size=(num_points, num_samples))
    distances = np.concatenate([fixed_col, random_cols], axis=1)

    # 仅保留 [mean-3*std, mean+3*std] 内的距离，其余置为 NaN 以便后续过滤
    valid_mask = (distances >= mean - 3 * std) & (distances <= mean + 3 * std)
    distances = np.where(valid_mask, distances, np.nan)

    # 采样点计算：point + d * normal，广播生成 (N, S, 3)
    sampled_points = points[:, None, :] + distances[:, :, None] * normals[:, None, :]

    # 对 z<0 的采样点，以原始点为中心做对称反射，同时翻转对应法向量
    z_neg_mask = sampled_points[:, :, 2] < 0
    sampled_points = np.where(
        z_neg_mask[:, :, None],
        2 * points[:, None, :] - sampled_points,
        sampled_points,
    )
    normals_broadcast = np.broadcast_to(normals[:, None, :], sampled_points.shape)
    normals_signed = np.where(z_neg_mask[:, :, None], -normals_broadcast, normals_broadcast)

    # 扁平化，并移除 NaN 对应的无效采样
    valid_flat = ~np.isnan(distances)
    sampled_points_flat = sampled_points[valid_flat]
    points_flat = np.broadcast_to(points[:, None, :], sampled_points.shape)[valid_flat]
    normals_flat = normals_signed[valid_flat]
    distances_flat = distances[valid_flat]

    # 可选：视点信息字典，如需要可在外部访问（此处不返回仅保留变量）
    viewpoints = {
        i: (sp, p, n, d)
        for i, (sp, p, n, d) in enumerate(
            zip(sampled_points_flat, points_flat, normals_flat, distances_flat)
        )
    }
    viewpoints_500 = {idx: val for idx, val in viewpoints.items() if abs(val[3] - mean) < 1e-6}
    _ = viewpoints_500  # 占位，避免未使用变量的警告

    points_with_normals = [(p, n) for p, n in zip(points, normals)]
    return sampled_points_flat, points_with_normals
