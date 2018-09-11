# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/9/10  14:42
# @File      : data.py
# @Project   : WXY_projects
==================================

定义新类，用于储存一次试验中的所有数据.
"""
from timelist import change_time_matrix
from utils import compare, poisson_process, indicator, modify_ob

import numpy as np

TAU = 5


class Data:

    def __init__(self, true_beta, true_gamma, n_sample=200, pr=0.5, source='random'):
        """
        生成n_sample个数据。
        """

        self.beta = true_beta
        self.gamma = true_gamma

        assert source in ['random', 'import'], 'Wrong source type!'

        if source == 'random':
            self.gen(n_sample, pr)
        else:
            pass  # TODO:补全 import & export data

    def gen(self, n_sample, pr):

        # 协变量
        self.x = np.random.uniform(0, 1, n_sample)
        self.z = np.random.uniform(0, 1, n_sample)

        # 删失时间
        self.c = np.random.uniform(TAU / 2, TAU, n_sample)

        # 生成齐次泊松过程强度l
        lamb = np.exp(self.gamma * self.x) + self.beta * self.z
        assert lamb.shape == (n_sample,)

        # 生成复发时间t
        # 先生成总共观测次数；再生成每一次具体的观测时间。
        self.m = np.random.poisson(lamb * self.c)
        t = poisson_process(self.c, self.m)

        # 生成观测时间T
        self.M = np.random.poisson(self.c)
        T_tmp = poisson_process(self.c, self.M)

        # 生成区间指示变量r
        r_tmp = indicator(self.m, pr)

        # 预处理T&r，并生成观测time_list
        T, r = modify_ob(T_tmp, r_tmp)
        self.t, self.T, self.r = change_time_matrix(t, T, r)


if __name__ == '__main__':
    data = Data(1, 1)

    print('Do not panic.')
