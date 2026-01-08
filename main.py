import dataLoader
from viewPointTaskEnv import viewPlanningEnv
from viewPointsPlannerEnv import ViewPointsPlannerEnv
from greedyPlanner import GreedyViewpointPlanner
from visualizer import visualizer
from utils import *
import pandas as pd

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
    
    ## 跑六次Baseline实验
    greedyData = []
    for i in range(6):
        trajectory = pd.DataFrame(greedyPlanner.train(iteration=10, run_id=i))
        greedyData.append(trajectory)

    saveTrajectory("greedy", 1, greedyData)
    v.paintGreedyTrajectory(pd.concat(greedyData))

    # 加载训练好的模型，然后进行视点规划
    
    # 可视化


