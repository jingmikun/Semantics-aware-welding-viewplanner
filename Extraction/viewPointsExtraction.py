#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import open3d as o3d
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def viewPointExtractor():
    """读取点云与法向量并生成候选视点列表。
    生成两个视点字典：一个是所有采样点，另一个是距离近似为500的视点。
    然后通过pickle保存这两个字典。便于后续读取
    """
    CURRDIR = Path(__file__).resolve().parent

    points, normals = loadRawData()
    viewPoints, viewPoints_500 = gaussianSample(points, normals)

    dataPathViewPoints = CURRDIR.parent / 'data' / 'viewPoints.pkl'
    with open(dataPathViewPoints, 'wb') as f:
        pickle.dump(viewPoints, f)

    dataPathViewPoints500 = CURRDIR.parent / 'data' / 'viewPoints_500.pkl'
    with open(dataPathViewPoints500, 'wb') as f:
        pickle.dump(viewPoints_500, f)


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
        viewpoints: 采样得到的视点坐标, 以一个字典的形式储存，其中
            key: 视点的索引 (int)
            value: (采样点坐标, 原始点, 法向量，距离）的元组，其中采样点坐标、原始点坐标、法向量为一个3维ndarray，距离是一个标量
        viewpoints_500: 在字典坐标中进一步筛选的距离近似为500的视点
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

    viewpoints = {
        i: (sp, p, n, d)
        for i, (sp, p, n, d) in enumerate(
            zip(sampled_points_flat, points_flat, normals_flat, distances_flat)
        )
    }
    viewpoints_500 = {idx: val for idx, val in viewpoints.items() if abs(val[3] - mean) < 1e-6}

    return viewpoints, viewpoints_500

if __name__ == "__main__":
    viewPointExtractor()