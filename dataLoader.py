from pathlib import Path
import pickle
import numpy as np
import pandas as pd


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
    return pd.read_csv(dataDIR / "model.csv").to_numpy()
