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

    def __init__(self, true_beta, true_gamma, n_sample=200, pr=0.8, source='random'):
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

    def vec_dN(self, q, time_stack, exp, num_stack=0):
        """
        取值于dN,即t_ij构成的向量。Q_i - Q(theta, t_ij)
        """
        # 初始化向量，并生成num_stack
        vec = []
        if type(num_stack) == int:
            num_stack = np.ones(q.shape[0])

        for i in range(self.n):
            tmp = (q[i] - ((compare(time_stack[i], self.c) @ (exp * q)) / (compare(time_stack[i], self.c) @ exp))) * \
                  num_stack[i]
            assert tmp.shape == time_stack[i].shape
            vec.append(np.nansum(tmp, axis=0))

        vec = np.array(vec)
        assert vec.shape == (self.n,)

        return vec

    def vec_dt(self, q, time_stack, exp, beta):
        """
        取值于dt,Q_i - Q(theta, c_j)
        """

        try:
            time_arr = flatten(time_stack)
            factor = time_arr / (compare(time_arr, self.c) @ exp)
        except TypeError:
            time_arr = time_stack
            factor = self.dt

        com = compare(time_arr, self.c)

        q_bar = (com @ (exp * q)) / (com @ exp)
        matrix = q[:, None] - q_bar[None, :]

        vec = beta * self.z[:, None] * (com.T * matrix) @ factor

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
        re_dt_arr = np.array([self.vec_dt(self.z, self.c, exp, beta), self.vec_dt(self.x, self.c, exp, beta)])

        pa_dN_arr = np.array([self.vec_dN(self.z, self.T_t, exp, self.T_n),
                              self.vec_dN(self.x, self.T_t, exp, self.T_n)])
        pa_dt_arr = np.array([self.vec_dt(self.z, self.T_t, exp, beta), self.vec_dt(self.x, self.T_t,exp, beta)])

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
    np.random.seed(42)

    for _ in tqdm(range(20)):
        est = Estimator(*true_values)
        sol = fsolve(est.cal_equation, true_values)
        hat_paras.append(sol)

    hat_paras = np.array(hat_paras)
    zeros = np.array(zeros)

    print(f'bias is {true_values - np.mean(hat_paras, axis=0)}.')
