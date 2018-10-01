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
from utils import compare, separate, flatten, r_i, r_mat
from scipy.optimize import fsolve, root
from tqdm import tqdm
from time import time

import matplotlib.pyplot as plt
import numpy as np


class Estimator(Data):
    """
    继承了data的结构，直接估计即可。
    """

    def __init__(self, true_beta, true_gamma, n_sample=200, pr=0.8, source='random'):
        super(Estimator, self).__init__(true_beta, true_gamma, n_sample, pr, source)

        # 计算估计方差时被导入
        self.hat_beta = None
        self.hat_gamma = None

        # 把T0 & T取出来用
        T0 = self.T0
        self.T0_t, self.T0_n = separate(T0)

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

    def q_matrix(self, q, time, exp):
        """
        生成矩阵q_i - q_bar(q)_j
        """
        tmp = q[:, None] - self.q_bar(q, time, exp)[None, :]
        assert tmp.shape == (q.shape[0], time.shape[0])
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
        取值于dt
        Q_i - Q(theta, c_j)

        time_stack 决定了是recurrent 还是 panel data
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

        pa_dN_arr = np.array([self.vec_dN(self.z, self.T0_t, exp, self.T0_n),
                              self.vec_dN(self.x, self.T0_t, exp, self.T0_n)])
        pa_dt_arr = np.array([self.vec_dt(self.z, self.T0_t, exp, beta), self.vec_dt(self.x, self.T0_t, exp, beta)])

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

    def mu(self, t):
        """
        Baseline hazard function of recurrent events.
        """
        t = np.array(t)
        assert t.shape == ()
        beta = self.hat_beta
        gamma = self.hat_gamma
        exp = np.exp(gamma * self.x)

        # 计算dN
        t_stack = self.t
        time_arr = flatten(t_stack)
        T_stack = self.T_t
        r_stack = self.r

        # 计算dt_arr
        dt_arr = np.array([0, *flatten(T_stack), *self.c])
        dt_arr.sort()
        dt_arr = dt_arr[dt_arr < 5]

        dt = np.diff(dt_arr)
        dt_arr = dt_arr[1:]

        # 计算mu
        mu_dN = np.sum(compare(time_arr, t).reshape(-1) / (
                (r_mat(time_arr, T_stack, r_stack).T * compare(time_arr, self.c)) @ exp), axis=0)

        # 计算dt
        mu_dt = beta * np.nansum((((r_mat(dt_arr, T_stack, r_stack).T * compare(dt_arr, self.c)) @ self.z) / (
                (r_mat(dt_arr, T_stack, r_stack).T * compare(dt_arr, self.c)) @ exp)) * (
                                         compare(dt_arr, t).reshape(-1) * dt))

        return mu_dN - mu_dt

    def v_matrix(self):
        """
        估计矩阵V
        """
        gamma = self.hat_gamma
        exp = np.exp(gamma * self.x)

        t_stack = self.t
        T0_stack = self.T0_t
        num_stack = self.T0_n

        v_mat = np.zeros((2, 2))

        for i in range(self.n):
            # recurrent-event part
            x_rec_arr = self.x[i] - self.q_bar(self.x, t_stack[i], exp)
            z_rec_arr = self.z[i] - self.q_bar(self.z, t_stack[i], exp)

            # panel-data part
            x_pan_arr = (self.x[i] - self.q_bar(self.x, T0_stack[i], exp)) * num_stack[i]
            z_pan_arr = (self.z[i] - self.q_bar(self.z, T0_stack[i], exp)) * num_stack[i]

            # sum them separately
            rec_arr = np.array([z_rec_arr, x_rec_arr])  # 生成两行
            pan_arr = np.array([z_pan_arr, x_pan_arr])

            assert x_rec_arr.shape == t_stack[i].shape
            assert x_pan_arr.shape == T0_stack[i].shape
            assert rec_arr.shape == (2, t_stack[i].shape[0])
            assert pan_arr.shape == (2, T0_stack[i].shape[0])
            v_mat += rec_arr @ rec_arr.T + 0.6 * pan_arr @ pan_arr.T

        assert v_mat.shape == (2, 2)

        return v_mat

    def a_dt(self, q1, q2):
        gamma = self.hat_gamma

        exp = np.exp(gamma * self.x)
        tmp = (compare(self.c, self.c).T * r_mat(self.c, self.T_t, self.r) *
               self.q_matrix(q1, self.c, exp) * self.q_matrix(q2, self.c, exp)) @ self.dt
        assert tmp.shape == (self.n,)

        return np.sum(tmp)

    def a_dmu(self, q1, q2):
        gamma = self.hat_gamma
        beta = self.hat_beta

        exp = np.exp(gamma * self.x)
        time_arr = flatten(self.t)

        dN_arr = exp @ (self.q_matrix(q1, time_arr, exp) * self.q_matrix(q2, time_arr, exp) * r_mat(time_arr, self.T_t,
                                                                                                    self.r)
                        * compare(time_arr, self.c).T) / (compare(time_arr, self.c) @ exp)
        assert dN_arr.shape == time_arr.shape
        dN = np.sum(dN_arr)

        dt = beta * ((exp @ (r_mat(self.c, self.T_t, self.r) *
                             self.q_matrix(q1, self.c, exp) * self.q_matrix(q2, self.c, exp) * compare(self.c,
                                                                                                       self.c).T)) / (
                             compare(self.c, self.c) @ exp)) @ (compare(self.c, self.c) * self.dt[:, None]) @ self.z

        result = dN - dt  # result还是太大了，原因是dt太小了么？
        assert result.shape == ()
        return result

    def a_tdlam(self, q1, q2):
        gamma = self.hat_gamma

        exp = np.exp(gamma * self.x)
        T0_t = flatten(self.T0_t)

        tmp = ((compare(T0_t, self.c).T * self.q_matrix(q1, T0_t, exp) * self.q_matrix(q2, T0_t, exp)) / (
                compare(T0_t, self.c) @ exp)) @ T0_t
        assert tmp.shape == self.c.shape
        return np.sum(tmp)

    def a_mudlam(self, q1, q2):
        gamma = self.hat_gamma

        exp = np.exp(gamma * self.x)
        T0_t = flatten(self.T0_t)

        tmp = ((exp @ (compare(T0_t, self.c).T * self.q_matrix(q1, T0_t, exp) * self.q_matrix(q2, T0_t, exp))) / (
                compare(T0_t, self.c) @ exp)) @ T0_t
        assert tmp.shape == ()
        return tmp

    def ase(self, hat_paras: np.array):
        # 载入估计量
        self.hat_beta = hat_paras[0]
        self.hat_gamma = hat_paras[1]

        # 分别计算V和A
        v_mat = self.v_matrix()

        a11 = self.a_dt(self.z, self.z) + self.a_tdlam(self.z, self.z)
        a12 = self.a_dmu(self.z, self.x) + self.a_mudlam(self.z, self.x)
        a21 = self.a_dt(self.z, self.x) + self.a_tdlam(self.z, self.x)
        a22 = self.a_dmu(self.x, self.x) + self.a_mudlam(self.x, self.x)

        a_mat = np.array([[a11, a12], [a21, a22]])
        a_inv = np.linalg.inv(a_mat)

        asv = a_inv * v_mat * a_inv
        ase = np.sqrt(asv[[0, 1], [0, 1]])
        return ase


if __name__ == '__main__':

    def simulation():
        hat_paras_list = []
        hat_std_list = []

        true_values = np.array([1, 1])

        # np.random.seed(42)

        start_time = time()
        for _ in tqdm(range(200)):
            # bias
            est = Estimator(*true_values, n_sample=100)
            # sol = fsolve(est.cal_equation, true_values)
            sol = root(est.cal_equation, true_values, method='Krylov').x
            hat_paras_list.append(sol)

            # var
            hat_std_list.append(est.ase(sol))
            # hat_std_list.append(est.ase(true_values))  # 真值代入效果也不好

            # # mu
            # for i in range(5):
            #     print('\n')
            #     print(f'mu({i}) is {est.mu(i)}')
            # x = np.arange(0, 5, 0.1)
            # y = []
            # for i in x:
            #     y.append(est.mu(i))
            # plt.plot(x, y)
            # plt.ylim(0, 6)
            # plt.show()


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
        cp_beta_count = (est_values[0] - 1.96 * ase[0] <= hat_paras_arr[:, 0]) * (
                hat_paras_arr[:, 0] <= est_values[0] + 1.96 * ase[0]
        )
        cp_gamma_count = (est_values[1] - 1.96 * ase[1] <= hat_paras_arr[:, 1]) * (
                hat_paras_arr[:, 1] <= est_values[1] + 1.96 * ase[1])
        cp_count = np.array([cp_beta_count, cp_gamma_count])

        cp = np.mean(cp_count, axis=1)

        print('\n=======================================================')
        print(f'running time is {time() - start_time:.2f}s.')
        print('-------------------------------------------------------')
        print(f'bias is {bias}.')
        print(f'ase is {ase}; esd is {esd}')
        print(f'cp is {cp}.')
        print('=======================================================\n')


    for _ in range(10):
        simulation()
