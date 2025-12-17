import dataLoader 


"""
这是一个启动器，运行时运行整个Pipeline
"""

if __name__ == "__main__":

    # 定义一些相机参数，主要是视域体的信息

    # 导入两个数据
    viewpoints = dataLoader.loadViewpoint() 
    model = dataLoader.loadModel()

    # 训练模型

    # 加载训练好的模型，然后进行视点规划

    # 可视化


