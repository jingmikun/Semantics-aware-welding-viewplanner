import open3d as o3d
import numpy as np
import pandas as pd
import os

def main():
    # ================= 配置区域 =================
    # 1. 你的 STL 模型路径
    stl_path = r"C:\Users\19578\Documents\weld_object_v3_surface.STL"
    
    # 2. 你刚才标好的 CSV 路径 (没有法向量的那个)
    csv_path = r"labeled_weld_undo_supported.csv"
    
    # 3. 输出文件名
    output_path = "labeled_weld_final_accurate.csv"
    # ===========================================

    print(f"正在加载模型: {os.path.basename(stl_path)} ...")
    # 使用 Open3D 的 Tensor API (t.geometry) 加载，这对于光线追踪/最近点查询非常快
    try:
        # 加载网格
        mesh = o3d.io.read_triangle_mesh(stl_path)
        # 计算网格的三角形法向量
        mesh.compute_triangle_normals()
        # 转换为 Tensor 格式以便进行高效查询
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print(f"正在读取点云数据: {csv_path} ...")
    try:
        # 读取 CSV，处理可能存在的空格或 # 号
        df = pd.read_csv(csv_path)
        # 清理列名 (比如 '# x' -> 'x')
        df.columns = [c.strip().replace('# ', '').replace(' ', '') for c in df.columns]
        
        # 提取 x, y, z 坐标
        points_np = df[['x', 'y', 'z']].values.astype(np.float32)
    except Exception as e:
        print(f"❌ CSV 读取失败: {e}")
        return

    print(f"正在计算 {len(points_np)} 个点的精准几何法向量...")
    
    # === 核心算法：RaycastingScene (场景光线追踪) ===
    # 建立一个场景，把模型放进去
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)
    
    # 将点云转换为 Tensor 查询点
    query_points = o3d.core.Tensor(points_np, dtype=o3d.core.float32)
    
    # 计算每个点到网格的最近点信息
    # 返回值包含 'primitive_ids'，即最近的三角形索引
    ans = scene.compute_closest_points(query_points)
    
    # 获取每个点对应的三角形 ID
    triangle_ids = ans['primitive_ids'].numpy()
    
    # 获取原始 mesh 的所有三角形法向量
    # (N_triangles, 3)
    tri_normals = np.asarray(mesh.triangle_normals)
    
    # 通过索引直接查表，得到每个点的法向量
    # 这里的逻辑是：点 P 落在三角形 T 上，那么 P 的法向量 = T 的法向量
    # (对于平滑曲面，也可以用插值法，但在工业焊缝检测中，面片法向通常更稳健)
    accurate_normals = tri_normals[triangle_ids]
    
    # === 保存结果 ===
    # 将法向量添加到 DataFrame
    df['nx'] = accurate_normals[:, 0]
    df['ny'] = accurate_normals[:, 1]
    df['nz'] = accurate_normals[:, 2]
    
    print("正在保存文件...")
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    print("-" * 50)
    print(f"✅ 成功！已生成包含精准 STL 法向的文件：\n   {os.path.abspath(output_path)}")
    print("-" * 50)
    print("前5行预览：")
    print(df.head())

if __name__ == "__main__":
    main()