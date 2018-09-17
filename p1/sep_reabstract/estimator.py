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
from utils import compare, separate, flatten
from scipy.optimize import fsolve
from tqdm import tqdm

import numpy as np


class Estimator(Data):
    """
    继承了data的结构，直接估计即可。
    """

    def __init__(self, true_beta, true_gamma, n_sample=200, pr=0, source='random'):
        super(Estimator, self).__init__(true_beta, true_gamma, n_sample, pr, source)

        # 3个估计量(在哪里用？)
        self.hat_beta = 0
        self.hat_gamma = 0
        self.variance = np.zeros([2, 2])

        # 2个积分时使用的变量（又在哪里用？）
        self.dN_arr = 0
        self.dt_arr = 0

        # 把T取出来用
        T = self.T
        self.T_t, self.T_n = separate(T)

    def vec_dN(self, q, times, exp):
        """
        取值于dN,即t_ij构成的向量。Q_i - Q(theta, t_ij)
        """
        vec = []
        for i in range(self.n):
            tmp = q[i] - ((compare(times[i], self.c) @ (exp * q)) / (compare(times[i], self.c) @ exp))
            assert tmp.shape == times[i].shape
            vec.append(np.nansum(tmp, axis=0))

        vec = np.array(vec)
        assert vec.shape == (self.n,)

        return vec

    def vec_dt(self, q, exp, beta):
        """
        取值于dt,Q_i - Q(theta, c_j)
        """

        q_bar = (self.Y @ (exp * q)) / (self.Y @ exp)
        matrix = q[:, None] - q_bar[None, :]

        vec = self.z[:, None] * (self.Y.T * matrix) @ self.dt * beta

        return vec

    # TODO: 重构下面两个函数。
    def panel_dN(self, q, times, exp):
        """
        仅用于panel data 的数据处理
        """
        vec = []
        for i in range(self.n):
            tmp = q[i] - ((compare(times[i], self.c) @ (exp * q)) / (compare(times[i], self.c) @ exp) * self.T_n[i])
            assert tmp.shape == times[i].shape
            vec.append(np.nansum(tmp, axis=0))

        vec = np.array(vec)
        assert vec.shape == (self.n,)

        return vec

    def panel_dt(self, q, exp, beta):
        """
        仅用于panel data的数据处理。

        T__ 是压平厚的T_t
        """
        T__ = flatten(self.T_t)

        q_bar = (compare(T__, self.c) @ (exp * q)) / (compare(T__, self.c) @ exp)
        matrix = q[:, None] - q_bar[None, :]

        vec = beta * self.z[:, None] * (matrix * compare(T__, self.c).T) @ (
                T__ / (compare(T__, self.c) @ exp))

        return vec

    def cal_equation(self, para):
        """
        计算估计方程，beta, gamma均为估计值。

        对照笔记中，按照t的来源，分为dN和dt两部分。
        """
        assert para.shape == (2,)
        beta, gamma = para

        exp = np.exp(gamma * self.x)

        re_dN_arr = np.array([self.vec_dN(self.z, self.t, exp), self.vec_dN(self.x, self.t, exp)])
        re_dt_arr = np.array([self.vec_dt(self.z, exp, beta), self.vec_dt(self.x, exp, beta)])

        pa_dN_arr = np.array([self.panel_dN(self.z, self.T_t, exp),
                              self.panel_dN(self.x, self.T_t, exp)])
        pa_dt_arr = np.array([self.panel_dt(self.z, exp, beta), self.panel_dt(self.x, exp, beta)])

        assert re_dN_arr.shape == (2, self.n)
        assert pa_dN_arr.shape == (2, self.n)

        dN = np.sum(re_dN_arr, axis=1) + np.sum(pa_dN_arr, axis=1)
        dt = np.sum(re_dt_arr, axis=1) + np.sum(pa_dt_arr, axis=1)
        # dN = np.sum(re_dN_arr, axis=1)
        # dt = np.sum(re_dt_arr, axis=1)
        # print(dN[1], dt[1])

        assert dN.shape == dt.shape
        assert dN.shape == (2,)

        return (dN - dt) / self.n


if __name__ == '__main__':
    hat_paras = []
    zeros = []

    true_values = np.array([0, 1])

    for _ in tqdm(range(20)):
        est = Estimator(*true_values)
        zero = est.cal_equation(true_values)
        sol = fsolve(est.cal_equation, true_values)
        hat_paras.append(sol)
        zeros.append(zero)

    hat_paras = np.array(hat_paras)
    zeros = np.array(zeros)

    print(f'bias is {true_values - np.mean(hat_paras, axis=0)}.')
