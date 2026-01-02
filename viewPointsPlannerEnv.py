# @author Jingmikun
# @time 2025/12/22

import gymnasium
import numpy as np
import pandas as pd


class ViewPointsPlannerEnv(gymnasium.Env):
    def __init__(self, viewpoints, model):
        super().__init__()
        self.viewpoints = viewpoints
        self.model = model  