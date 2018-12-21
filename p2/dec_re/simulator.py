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
from itertools import product

from utils import *
from tqdm import tqdm
from est_1220 import Estimator


def simulation(n_iter=10, n_sample=200):
    """
    todo 计算并记录simulation的结果。
    """

    def est_once(t_param):
        """
        算出一个true param 的bias 和 var

        :return: bias, var
        """
        bias = np.empty((n_iter, 4))
        std = np.empty((n_iter, 4))

        for i in range(n_iter):
            est = Estimator(t_param, n_sample)
            bias[i, :], std[i, :] = est.estimate(t_param)

        return bias.mean(axis=0), std.mean(axis=0)

    t_params = list(setting_params([-1], [1], [-1], [1]))
    n_test = len(t_params)

    bias = np.empty((n_test, 4))
    std = np.empty((n_test, 4))

    for ii in range(n_test):
        bias[ii, :], std[ii, :] = est_once(t_params[ii])
    print(bias)


def setting_params(arr1, arr2, arr3, arr4):
    params = product(arr1, arr2, arr3, arr4)
    return params


def save_print():
    return None


def latex():
    return None


if __name__ == '__main__':
    simulation(10, 200)
