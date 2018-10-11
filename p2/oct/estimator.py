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

from oct.utils import make_a_star, flatten, find_loc, compare
from oct.data import Data


class Estimator(Data):

    def __init__(self, parameters, n_samples=200):
        super(Estimator, self).__init__(parameters, n_samples)

    def equation_sn(self, a):
        a1 = a[0]
        a2 = a[1]
        t_star_a = make_a_star(self.t, np.exp(self.x * a1))
        y_star_a = make_a_star(self.y, np.exp(self.x * a1))

        # Todo: cycle 4

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


if __name__ == '__main__':
    true_paras = np.array([-1, 1, -1, 1])
    est = Estimator(true_paras)

    # testing rl_arr
    ta = np.array([1, 2, 3])
    tb = np.array([3, 5, 7])
    t_star = np.array([ta, tb])
    y_stat = np.array([6, 8])
    print(est.rl_arr(t_star, y_stat))  # result should be [1, 2, 4, 4, 5, 3]

    # testing lambda_arr
    print(est.lambda_arr(t_star, y_stat))  # result should be shape = 7, and values go up to 1

