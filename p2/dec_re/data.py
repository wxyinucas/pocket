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
import pandas as pd
import matplotlib.pyplot as plt
from utils import c_generate, poisson_process, proper_d, flatten

TAU = 10


class Data:

    def __init__(self, parameters: np.array, n_sample=200):
        self.true_alpha1 = parameters[0]
        self.true_alpha2 = parameters[1]
        self.true_beta1 = parameters[2]
        self.true_beta2 = parameters[3]

        self.n_sam = n_sample
        self.x = self.cov()
        self.z = self.latent()
        self.c = self.censor()
        self.d = self.death()
        self.y, self.delta = self.observe_indicator()
        self.m, self.t = self.recurrent()

        names = ['x', 'z', 'c', 'd', 'y', 'delta', 'm']
        values = np.stack([self.x, self.z, self.c, self.d, self.y, self.delta, self.m]).T

        self.dt = pd.DataFrame(data=values, columns=names)

    def __repr__(self):
        return f'true parameters {self.true_alpha1, self.true_alpha2, self.true_beta1, self.true_beta2}.'

    # 协变量
    def cov(self):
        """生成0，1均匀分布的协变量， 在对复发和终止使用相同的协变量。"""
        return np.random.uniform(0, 1, self.n_sam)

    # 潜变量
    def latent(self):
        """生成服从Gamma（2， 5）的潜变量。"""
        return np.random.gamma(2, 5, self.n_sam)

    # 删失时间
    def censor(self):
        """由x、z生成删失时间， 再与TAU取小。"""
        c_ = c_generate(self.x, self.z)
        return np.minimum(c_, TAU)

    # 死亡时间 h = t/400
    def death(self):
        """
        生成最终死亡时间。

        1. 生成强度lambda
        2. 生成发生次数， 并取大于零的数目
        3. 生成合适的死亡时间。
        """

        d_lamb = 1 / 100 * self.z * np.exp(self.x * (self.true_beta1 + self.true_beta2)) * (TAU ** 2)
        m_numerical = np.random.poisson(d_lamb)
        m = (m_numerical > 0).astype('int')
        d = poisson_process(np.ones(self.n_sam) * TAU, m)
        res = proper_d(d)
        if (m == 1).all():
            res = flatten(res)
        return res

    # 观测时间 & delta
    def observe_indicator(self):
        """生成观测时间和只是变量（观测到的是死亡还是删失）"""
        y = np.minimum(self.c, self.d)
        delta = self.d <= self.c
        return y, delta

    # 复发时间 l = t / 10
    def recurrent(self):
        """生成复发时间发生总数和每次的时间。"""
        r_lamb = 1 / 50 * self.z * np.exp(self.x * (self.true_alpha2 + self.true_alpha2)) * (self.y ** 2)
        m = np.random.poisson(r_lamb)
        t = poisson_process(self.y, m)
        return m, t


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
    print('In which the 0 times are:')
    print(np.where(data.m == 0)[0].shape[0])
    print(np.sum(data.m == 0))
    plt.boxplot(data.m)
    plt.title('recurrent times')
    plt.show()
