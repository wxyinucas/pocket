# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# Time      : 2018/9/11  11:55
# File      : estimator.py
# Project   : WXY_projects
==================================

"""
from data import Data
from utils import compare
from scipy.optimize import fsolve

import numpy as np


class Estimator(Data):
    """
    继承了data的结构，直接估计即可。
    """

    def __init__(self, true_beta, true_gamma, n_sample=200, pr=0.5, source='random'):
        super(Estimator, self).__init__(true_beta, true_gamma, n_sample, pr, source)

        # 3个估计量
        self.hat_beta = 0
        self.hat_gamma = 0
        self.variance = np.zeros([2, 2])

        # 2个积分时使用的变量
        self.dN_arr = 0
        self.dt_arr = 0

    def vec_dN(self, q, exp):
        """
        取值于dN,即t_ij构成的向量。Q_i - Q(theta, t_ij)
        """
        vec = []
        for i in range(self.n):
            try:
                tmp = q[i] - compare(self.t[i], self.c) @ (exp * q) / (compare(self.t[i], self.c) @ exp)
                assert (tmp.shape == self.t[i].shape)
                vec.append(np.sum(tmp, axis=0))
            except ValueError:
                pass

        vec = np.array(vec)
        assert vec.shape[0] == self.n

        return vec

    def vec_dt(self, q, exp):
        """
        取值于dt,Q_i - Q(theta, c_j)
        """

        cj = (self.Y @ (exp * q)) / (self.Y @ exp)
        t = self.z[None, :] - cj[:, None]

        return t

    def cal_equation(self, para):
        """
        计算估计方程，beta, gamma均为估计值。

        对照笔记中，按照t的来源，分为dN和dt两部分。
        """
        assert para.shape == (2,)
        beta, gamma = para

        exp = np.exp(gamma * self.x)

        self.dN_arr = np.array([self.vec_dN(self.x, exp), self.vec_dN(self.z, exp)])
        a = self.z @ (self.Y.T * self.vec_dt(self.x, exp)) @ self.dt * beta
        b = self.z @ (self.Y.T * self.vec_dt(self.z, exp)) @ self.dt * beta

        self.dt_arr = np.array([a, b])

        assert self.dN_arr.shape == (2, self.n)
        assert self.dt_arr.shape == (2, )

        dN = np.sum(self.dN_arr, axis=1)
        dt = self.dt_arr

        assert dN.shape == dt.shape
        assert dN.shape == (2,)

        return (dN - dt) / self.n


if __name__ == '__main__':
    hat_paras = []


    for _ in range(100):
        est = Estimator(1, 0)
        sol = fsolve(est.cal_equation, np.array([1, 0]))
        hat_paras.append(sol)

    hat_paras = np.array(hat_paras)

    print("hello")
