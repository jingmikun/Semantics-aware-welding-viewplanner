import dataLoader
from viewPointTaskEnv import viewPlanningEnv
from viewPointsPlannerEnv import ViewPointsPlannerEnv
from greedyPlanner import GreedyViewpointPlanner
from visualizer import visualizer

"""
这是一个启动器，运行时运行整个Pipeline
"""

if __name__ == "__main__":

    v = visualizer()
    
    # 导入两个数据
    viewpoints = dataLoader.downSampleViewpoint() 
    model = dataLoader.downSampleModel()
    modelMesh = dataLoader.loadMesh()

    # 训练模型
    ## 作为Baseline, 训练和报告了一个贪婪梯度上升算法模型的结果
    greedyenv = viewPlanningEnv(model, viewpoints, modelMesh)
    greedyPlanner = GreedyViewpointPlanner(greedyenv)
    trajectory = greedyPlanner.train()

    # 加载训练好的模型，然后进行视点规划

    # 可视化


