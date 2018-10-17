# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# Time      : 2018/10/11  14:30
# File      : estimator.py
# Project   : WXY_projects
==================================

"""
import numpy as np
import matplotlib.pyplot as plt

from oct.utils import make_a_star, flatten, find_loc_arr, compare
from oct.data import Data
from scipy.optimize import root


class Estimator(Data):

    def __init__(self, parameters, n_samples=200):
        super(Estimator, self).__init__(parameters, n_samples)

        self.lambda_y_a_hat = None
        self.a_hat = root(self.equation_sn, parameters[:2]).x

        # self.a_hat = np.array([-1, 1])  # 用于缩短测试时间
        # self.equation_sn(self.a_hat)  # 用于缩短测试时间

        self.b_hat = root(self.equation_hn, parameters[2:], method='lm').x
        # self.b_hat = 0  # 用于缩短测试时间

        # self.a_hat = np.array([-1, 1]) + np.random.normal(0, 1, 2)  # 用于测试
        # self.b_hat = np.array([-1, 1]) + np.random.normal(0, 1, 2)  # 用于测试

    def equation_sn(self, a):
        a1 = a[0]
        a2 = self.true_alpha2 + np.random.normal(0, 1)
        # a2 = a[1]

        t_star_a = make_a_star(self.t, np.exp(self.x * a1))
        y_star_a = make_a_star(self.y, np.exp(self.x * a1))
        exp = np.exp(self.x * (a2 - a1))

        lambda_y = self.lambda_y(t_star_a, y_star_a)
        #
        zero_loc = np.where(lambda_y == 0)
        lambda_y[zero_loc] = 1
        self.lambda_y_a_hat = lambda_y
        #
        mu = self.mu(exp, lambda_y)

        assert not (lambda_y == 0).any()

        s1 = self.x @ (exp * self.m * (1 / lambda_y) - mu)
        # s2 = (self.y * self.x) @ (exp * self.m * (1 / lambda_y) - mu)

        s = np.array([s1, 0 * s1])
        return s / self.n_sam

    def rl_arr(self, t_star_a, y_star_a):
        """生成rl"""
        sl = flatten(t_star_a)
        sl.sort()

        def rl_one(sl_scale):
            count_rl = 0
            for i in range(len(t_star_a)):
                comp = compare(t_star_a[i], sl_scale) * (sl_scale <= y_star_a[i])
                assert comp.shape == (t_star_a[i].shape[0], 1)
                count_rl += np.sum(comp)
            return count_rl

        func_arr = np.vectorize(rl_one, signature='()->()')

        return func_arr(sl)

    def lambda_arr(self, t_star_a, y_star_a):
        """生成lambda向量。"""
        rl_arr = self.rl_arr(t_star_a, y_star_a)
        factor = 1 - (1 / rl_arr)
        factor = np.append(factor, 1)
        assert factor.shape[0] == flatten(t_star_a).shape[0] + 1
        return np.cumprod(factor[::-1])[::-1]

    def lambda_y(self, t_star_a, y_star_a):
        """生成，时间为y_star的，lambda的估计。"""
        sl = flatten(t_star_a)
        sl.sort()

        locs = find_loc_arr(sl, y_star_a)
        assert locs.shape == y_star_a.shape
        lambda_arr = self.lambda_arr(t_star_a, y_star_a)
        return lambda_arr[locs]

    def mu(self, exp, lambda_y):
        tmp = (self.m * exp) @ (1 / lambda_y)
        assert tmp.shape == ()
        return tmp / self.n_sam

    def equation_hn(self, b):
        b1 = b[0]
        # b1 = -1
        b2 = self.true_beta2 + np.random.normal(0, 1)
        # b2 = b[1]

        a_hat = self.a_hat
        hat_z = self.hat_z(a_hat)

        exp = np.exp(self.x * (b2 - b1))
        assert exp.shape == self.x.shape

        y_star_b = self.y * np.exp(self.x * b1)
        assert y_star_b.shape == self.y.shape
        # g2 = y_star_b * self.x

        y_star_b_filter = y_star_b[self.delta]
        x_filter = self.x[self.delta]
        # g2_filter = g2[self.delta]

        h1_arr = x_filter - self.q_bar(self.x, hat_z, y_star_b_filter, y_star_b, exp)
        # h2_arr = g2_filter - self.q_bar(g2, hat_z, y_star_b_filter, y_star_b, exp)

        # assert h1_arr.shape == h2_arr.shape == (np.sum(self.delta),) == x_filter.shape
        h1 = np.sum(h1_arr)
        # h2 = np.sum(h2_arr)
        h2 = np.zeros_like(h1)
        # h1 = np.zeros_like(h2)
        h = np.array([h1, h2])

        return h / self.n_sam

    def hat_z(self, a_hat):
        a1 = a_hat[0]
        a2 = a_hat[1]

        exp = np.exp(self.x * (a2 - a1))
        z = self.m / (exp * self.lambda_y_a_hat)

        assert z.shape == self.m.shape
        return z

    def q_bar(self, x, z, time, c, exp):
        """
        生成z_bar, x_bar.
        :param x:  vec
        :param time:  vec
        :param exp: vec
        """
        assert type(x) != int

        tmp = (compare(time, c) @ (x * exp * z)) / (compare(time, c) @ (exp * z))

        assert time.shape == tmp.shape

        return tmp


if __name__ == '__main__':
    true_paras = np.array([-1, 1, -1, 1])
    est = Estimator(true_paras)

    # testing rl_arr
    ta = np.array([1, 2, 3])
    tb = np.array([3, 4, 5, 7])
    t_star = np.array([ta, tb])
    y_star = np.array([4, 8])
    print(est.rl_arr(t_star, y_star))  # result should be [1, 2, 4, 4, 5，2, 3]

    # testing lambda_arr
    # print(est.lambda_arr(t_star, y_star))  # result should be shape = 7, and values go up to 1
    # print(est.lambda_y(t_star, y_star))

    #
    est.equation_sn(np.array([-1, 1]))