# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# Time      : 2018/10/10  10:53
# File      : data.py
# Project   : WXY_projects
==================================

"""
import numpy as np
import matplotlib.pyplot as plt
from oct.utils import compare, c_generate, poisson_process

TAU = 10


class Data:

    def __init__(self, parameters: np.array, n_sample=200):
        self.true_alpha1 = parameters[0]
        self.true_alpha2 = parameters[1]
        self.true_beta1 = parameters[2]
        self.true_beta2 = parameters[3]

        self.n_sam = n_sample
        self.generate()

    def generate(self):
        # 协变量
        self.x = np.random.uniform(0, 1, self.n_sam)

        # 潜变量
        self.z = np.random.gamma(2, 5, self.n_sam)

        # 删失时间
        self.c = c_generate(self.x, self.z)

        # 死亡时间
        self.d =


if __name__ == '__main__':

    data = Data([-1, 1, -1, 1])
    c_mask = data.x < 0.5
    print('C1:')
    print(np.mean(data.c[c_mask]))
    plt.boxplot(data.c[c_mask], showmeans=True)
    plt.ylim(0, 20)
    plt.show()
    print('C2:')
    print(np.mean(data.c[~c_mask]))
    plt.boxplot(data.c[~c_mask], showmeans=True)
    plt.ylim(0, 20)
    plt.show()

