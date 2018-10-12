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
from oct.utils import c_generate, poisson_process, proper_d

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
        c = c_generate(self.x, self.z)
        self.c = np.minimum(c, TAU)

        # 死亡时间 h = t/400
        d_lamb = 1 / 100 * self.z * np.exp(self.x * (self.true_beta1 + self.true_beta2)) * (TAU ** 2)
        m_numerical = np.random.poisson(d_lamb)
        m = (m_numerical > 0).astype('int')
        d = poisson_process(np.ones(self.n_sam) * TAU, m)
        self.d = proper_d(d)

        # 观测时间 & delta
        self.y = np.minimum(self.c, self.d)
        self.delta = self.d <= self.c

        # 复发时间 l = t / 10
        r_lamb = 1 / 50 * self.z * np.exp(self.x * (self.true_alpha2 + self.true_alpha2)) * (self.y ** 2)
        self.m = np.random.poisson(r_lamb)
        self.t = poisson_process(self.y, self.m)


if __name__ == '__main__':
    data = Data([-1, 1, -1, 1])

    # Censoring Time
    c_mask = data.x < 0.5
    print('C1:')
    print(np.mean(data.c[c_mask]))
    plt.boxplot(data.c[c_mask], showmeans=True)
    plt.title('c1')
    plt.ylim(0, 20)
    plt.show()
    print('C2:')
    print(np.mean(data.c[~c_mask]))
    plt.boxplot(data.c[~c_mask], showmeans=True)
    plt.ylim(0, 20)
    plt.title('c2')
    plt.show()

    # Death time
    print('\n=======Death time========')
    print('The average death time is :')
    print(f'{np.mean(data.delta):.2f}')
    plt.boxplot(data.d)
    plt.title('death time')
    plt.show()
    plt.boxplot(data.y)
    plt.title('observed time')
    plt.show()
    print('\n=======Observe time========')
    print('The average observed time is:')
    print(np.mean(data.y))

    # Recurrent time
    print('\n========Recurrent Time========')
    print('the average recurrent time times are:')
    print(np.mean(data.m))
