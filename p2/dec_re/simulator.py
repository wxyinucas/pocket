# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2018-12-19  11:45
# File      : simulator.py
# Project   : WXY_projects
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

用于：
1. 设定真值
2. 记录结果（文件&console）
3. 设定latex表格输出
"""

from utils import *
from tqdm import tqdm
from est_1218 import Estimator


def simulation(true_parameters, n_iter=500, n_sample=None):
    """
    计算并记录simulation的结果。
    """


    est = np.empty((n_iter, 4))
    std = np.empty((n_iter, 4))

    for p in tqdm(true_parameters):
        est[]
