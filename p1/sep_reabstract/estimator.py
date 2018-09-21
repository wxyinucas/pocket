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

        # 2个积分时使用的变量（又在哪里用？）
        self.dN_arr = 0
        self.dt_arr = 0

        # 把T取出来用
        T = self.T
        self.T_t, self.T_n = separate(T)

    def q_bar(self, q, time, exp):
        """
        生成z_bar, x_bar.
        :param q:  vec
        :param time:  vec
        :param exp: vec
        """
        assert type(q) != int

        tmp = (compare(time, self.c) @ (exp * q)) / (compare(time, self.c) @ exp)
        assert time.shape == tmp.shape

        return tmp

    def vec_dN(self, q, time_stack, exp, num_stack=0):
        """
        取值于dN,即t_ij构成的向量。Q_i - Q(theta, t_ij)
        num_stack 用于 panel
        """
        # 初始化向量，并生成num_stack
        vec = []
        if type(num_stack) == int:
            num_stack = np.ones(q.shape[0])

        for i in range(self.n):
            tmp = (q[i] - self.q_bar(q, time_stack[i], exp)) * num_stack[i]
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

        q_bar = self.q_bar(q, time_arr, exp)
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
        pa_dt_arr = np.array([self.vec_dt(self.z, self.T_t, exp, beta), self.vec_dt(self.x, self.T_t, exp, beta)])

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

    # Now variance

    def v_matrix(self):
        """
        估计矩阵V
        """
        gamma = self.hat_gamma
        exp = np.exp(gamma * self.x)

        v_mat = np.zeros((2, 2))
        time_stack = self.t

        for i in range(self.n):
            x_arr = self.x[i] - self.q_bar(self.x, time_stack[i], exp)
            z_arr = self.z[i] - self.q_bar(self.z, time_stack[i], exp)

            arr = np.array([z_arr, x_arr])  # 生成两行
            # arr = np.array([x_arr, z_arr])  # 生成两行
            assert x_arr.shape == time_stack[i].shape
            assert arr.shape == (2, time_stack[i].shape[0])
            v_mat += arr @ arr.T

        assert v_mat.shape == (2, 2)

        return v_mat

    def a_dt(self, q1, q2):
        gamma = self.hat_gamma

        exp = np.exp(gamma * self.x)
        tmp = (compare(self.c, self.c).T * (q1[:, None] - self.q_bar(q1, self.c, exp)[None, :]) * (
                q2[:, None] - self.q_bar(q2, self.c, exp)[None, :])) @ \
              self.dt
        assert tmp.shape == (self.n,)

        return np.sum(tmp)

    def a_dmu(self, q1, q2):
        gamma = self.hat_gamma
        beta = self.hat_beta

        exp = np.exp(gamma * self.x)
        time_arr = flatten(self.t)

        dN_arr = exp @ ((q1[:, None] - self.q_bar(q1, time_arr, exp)[None, :]) * (
                q2[:, None] - self.q_bar(q2, time_arr, exp)[None, :])
                        * compare(time_arr, self.c).T) / (compare(time_arr, self.c) @ exp)
        assert dN_arr.shape == time_arr.shape
        dN = np.sum(dN_arr)

        dt = beta * ((exp @ ((q1[:, None] - self.q_bar(q1, self.c, exp)[None, :]) * (
                q2[:, None] - self.q_bar(q2, self.c, exp)[None, :])
                             * compare(self.c, self.c).T)) / (compare(self.c, self.c) @ exp)) @ (
                     compare(self.c, self.c) * self.dt[:, None]) @ self.z

        result = dN - dt
        assert result.shape == ()
        return result

    def ase(self, hat_paras: np.array):
        # 载入估计量
        self.hat_beta = hat_paras[0]
        self.hat_gamma = hat_paras[1]

        v_mat = self.v_matrix()

        a11 = self.a_dt(self.z, self.z)
        a12 = self.a_dmu(self.z, self.x)
        a21 = self.a_dt(self.z, self.x)
        a22 = self.a_dmu(self.x, self.x)

        a_mat = np.array([[a11, a12], [a21, a22]])
        a_inv = np.linalg.inv(a_mat)

        asv = a_inv * v_mat * a_inv.T
        ase = np.sqrt(asv[[0, 1], [0, 1]])
        return ase


if __name__ == '__main__':
    hat_paras_list = []
    hat_std_list = []

    true_values = np.array([1, 0])
    np.random.seed(42)

    for _ in tqdm(range(1000)):
        # bias
        est = Estimator(*true_values)
        sol = fsolve(est.cal_equation, true_values)
        hat_paras_list.append(sol)

        # var
        hat_std_list.append(est.ase(sol))

    # 处理估计结果, 两列
    hat_paras_arr = np.array(hat_paras_list)
    hat_std_arr = np.array(hat_std_list)

    # 计算bias
    est_values = np.mean(hat_paras_arr, axis=0)
    bias = true_values - est_values

    # 计算ase & esd
    ase = np.mean(hat_std_arr, axis=0)
    esv = np.cov(hat_paras_arr, rowvar=False)
    esd = np.sqrt(esv[[0, 1], [0, 1]])

    # 计算cp
    cp_beta_count = compare(est_values[0] - 1.9)

    print(f'bias is {bias}.')
    print(f'ase is {ase}; esd is {esd}')
