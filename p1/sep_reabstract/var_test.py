# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# Time      : 2018/9/11  11:55
# File      : estimator.py
# Project   : WXY_projects
==================================

"""
from var_data_test import TestData
from utils import compare, separate, flatten
from scipy.optimize import fsolve, root
from tqdm import tqdm
from time import time

import numpy as np


class Estimator(TestData):
    """
    继承了data的结构，直接估计即可。
    """

    def __init__(self, true_beta, true_gamma, n_sample=200, pr=1, source='random'):
        super(Estimator, self).__init__(true_beta, true_gamma, n_sample, pr, source)

        # 3个估计量(在哪里用？)
        self.hat_beta = 0
        self.hat_gamma = 0

        # 把T取出来用
        T = self.T
        self.T_t, self.T_n = separate(T)

        self.making_Y()

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

        t_stack = self.t
        T_stack = self.T_t
        num_stack = self.T_n

        v_mat = np.zeros((2, 2))

        for i in range(self.n):
            # recurrent-event part
            x_rec_arr = self.x[i] - self.q_bar(self.x, t_stack[i], exp)
            z_rec_arr = self.z[i] - self.q_bar(self.z, t_stack[i], exp)

            # panel-data part
            x_pan_arr = (self.x[i] - self.q_bar(self.x, T_stack[i], exp)) * num_stack[i]
            z_pan_arr = (self.z[i] - self.q_bar(self.z, T_stack[i], exp)) * num_stack[i]

            # sum them separately
            rec_arr = np.array([z_rec_arr, x_rec_arr])  # 生成两行
            pan_arr = np.array([z_pan_arr, x_pan_arr])

            assert x_rec_arr.shape == t_stack[i].shape
            assert x_pan_arr.shape == T_stack[i].shape
            assert rec_arr.shape == (2, t_stack[i].shape[0])
            assert pan_arr.shape == (2, T_stack[i].shape[0])
            v_mat += rec_arr @ rec_arr.T + pan_arr @ pan_arr.T

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

        result = dN - dt  # result还是太大了，原因是dt太小了么？
        assert result.shape == ()
        return result

    def ase(self, hat_paras: np.array):
        # 载入估计量
        self.hat_beta = hat_paras[0]
        self.hat_gamma = hat_paras[1]

        # 分别计算V和A
        v_mat = self.v_matrix()

        a11 = self.a_dt(self.z, self.z)
        a12 = self.a_dmu(self.z, self.x)
        a21 = self.a_dt(self.z, self.x)
        a22 = self.a_dmu(self.x, self.x)

        a_mat = np.array([[a11, a12], [a21, a22]])
        a_inv = np.linalg.inv(a_mat)

        asv = a_inv * v_mat * a_inv
        ase = np.sqrt(asv[[0, 1], [0, 1]])
        return ase

    # original method
    def making_Y(self, detail=False):
        # create a lower triangular matrix
        self.tri = np.tri(self.n)

        ## This Y is \tilda{Y} in calculation
        self.Y = np.array([np.zeros(self.n)])
        for i in range(self.n):
            temp = np.array([(k < self.c).astype(int) for k in self.t[i]])
            try:
                self.Y = np.append(self.Y, temp, axis=0)
            except ValueError:
                pass
        self.Y = np.delete(self.Y, 0, 0)  # 1 indicate which vector to delete, 0 is axis.

        self.length = np.array([len(i) for i in self.t])
        self.index = np.cumsum(self.length)
        self.index = np.insert(self.index, 0, 0)

        if detail:
            print('The shape of Y suppose to be ({0}, {1}), \n'
                  'while the actual shape is {2} '.
                  format(np.sum(self.length), self.n, self.Y.shape))

    def making_matrix(self, x, detail=False):
        # x[0]: beta
        # x[1]: gamma
        self.q0 = np.exp(x[1] * self.x)
        self.tri = compare(self.c, self.c).T

        ## Zt is [z_k - \bar{z}(c_j-)]_{(j,k)}
        Z_cj = (self.tri.T @ (self.q0 * self.z)) / (self.tri.T @ self.q0)
        self.Z_t = self.z[None, :] - Z_cj[:, None]

        ## Z_N is [z_k - \bar{z}(t_ij)]_{(ij,k)}, whose dimension is (l,n)
        Z_tij = (self.Y @ (self.q0 * self.z)) / (self.Y @ self.q0)
        self.Z_N = self.z[None, :] - Z_tij[:, None]

        X_cj = (self.tri.T @ (self.q0 * self.x)) / (self.tri.T @ self.q0)
        self.X_t = self.x[None, :] - X_cj[:, None]
        X_tij = (self.Y @ (self.q0 * self.x)) / (self.Y @ self.q0)
        self.X_N = self.x[None, :] - X_tij[:, None]

        if detail:
            print('The shape of Z_t suppose to be ({0},{0}), while the actual shape is {1}'.format(
                self.n, self.Z_t.shape))
            print('The shape of Z_N suppose to be ({0},{1}), while the actual shape is {2}'.format(
                np.sum(self.length), self.n, self.Z_N.shape))

    # estimating
    def equation(self, x):
        ## S = A - B
        self.making_matrix(x=x)

        self.A_z = 0
        self.A_x = 0

        for i in range(self.n):
            self.A_z += np.sum(self.Z_N[self.index[i]:self.index[i + 1], i])
            self.A_x += np.sum(self.X_N[self.index[i]:self.index[i + 1], i])

        self.B_z = self.z @ (self.tri * self.Z_t.T) @ self.dt * x[0]
        self.B_x = self.z @ (self.tri * self.X_t.T) @ self.dt * x[0]

        result = np.array([self.A_z - self.B_z, self.A_x - self.B_x])  # / self.num
        return result

    def calculator(self):
        # self.sol = optimize.root(self.equation, np.array([self.beta, self.gamma]),
        #                          method='Krylov')
        self.sol = root(self.equation, np.array([0, 0]),
                        method='Krylov')
        return np.array(self.sol.x)

    def A_bb(self):
        result = self.z @ (self.tri * self.Z_t.T) @ self.dt
        return result

    def A_bg(self):
        Alpha = np.sum(((self.Y * self.Z_N) @ (self.q0 * self.x)) /
                       (self.Y @ self.q0))
        Beta = self.hat_beta * self.z @ self.tri @ \
               (self.dt * ((self.tri.T * self.Z_t) @ (self.q0 * self.x) /
                           (self.tri.T @ self.q0)))

        return Alpha - Beta

    def A_gb(self):
        result = self.z @ (self.tri * self.X_t.T) @ self.dt
        return result

    def A_gg(self):
        Alpha = np.sum(((self.Y * self.X_N) @ (self.q0 * self.x)) /
                       (self.Y @ self.q0))
        Beta = self.hat_beta * self.z @ self.tri @ \
               (self.dt * ((self.tri.T * self.X_t) @ (self.q0 * self.x) /
                           (self.tri.T @ self.q0)))

        return Alpha - Beta

    def A_matrix(self):
        result = np.array([[self.A_bb(), self.A_bg()], [self.A_gb(), self.A_gg()]]) \
                 / self.n
        return result

    # variance estimating
    def V_matrix(self, detail=False):
        temp = 0
        for i in range(self.n):
            alpha = self.Z_N[self.index[i]:self.index[i + 1], i][None, :]
            delta = self.X_N[self.index[i]:self.index[i + 1], i][None, :]
            zeta = np.append(alpha, delta, axis=0)
            temp += zeta @ zeta.T
        if detail:
            print('shape of alpha is', alpha.shape)
            print('shape of zeta is ', zeta.shape)
            print('Shape of temp = ', temp.shape)
        return temp / self.n

    def ASE(self):
        """
        1. self.calculator
        2. self.get_hat(self.sol.x)
        3. self.making_matrix([self.beta_h, self.gamma_h])
        """
        self.making_Y()
        self.making_matrix([self.hat_beta, self.hat_gamma])

        A = self.A_matrix()
        V = self.V_matrix()

        asv = np.linalg.inv(A) * V * np.linalg.inv(A).T
        ase = np.sqrt(asv[[0, 1], [0, 1]])
        return ase / np.sqrt(self.n)


if __name__ == '__main__':
    def simulation():
        global est

        hat_paras_list = []
        hat_std_list = []
        origin_paras_list = []
        origin_std_list = []

        true_values = np.array([1, 1])

        # np.random.seed(42)

        start_time = time()
        for _ in tqdm(range(100)):
            # bias
            est = Estimator(*true_values, n_sample=200)
            sol = fsolve(est.cal_equation, true_values)
            hat_paras_list.append(sol)
            origin_paras_list.append(est.calculator())

            # var
            hat_std_list.append(est.ase(sol))
            origin_std_list.append(est.ASE())

        # 处理估计结果, 两列
        hat_paras_arr = np.array(hat_paras_list)
        hat_std_arr = np.array(hat_std_list)

        origin_paras_arr = np.array(origin_paras_list)
        origin_std_arr = np.array(origin_std_list)

        # 计算bias
        est_values = np.mean(hat_paras_arr, axis=0)
        bias = true_values - est_values

        origin_est_values = np.mean(origin_paras_arr, axis=0)
        origin_bias = true_values - origin_est_values

        # 计算ase & esd
        ase = np.mean(hat_std_arr, axis=0)
        origin_ase = np.mean(origin_std_arr, axis=0)
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

        # 计算cp
        o_cp_beta_count = (origin_est_values[0] - 1.96 * origin_ase[0] <= origin_paras_arr[:, 0]) * (
                hat_paras_arr[:, 0] <= origin_est_values[0] + 1.96 * origin_ase[0]
        )
        o_cp_gamma_count = (origin_est_values[1] - 1.96 * origin_ase[1] <= origin_paras_arr[:, 1]) * (
                hat_paras_arr[:, 1] <= origin_est_values[1] + 1.96 * origin_ase[1])
        o_cp_count = np.array([o_cp_beta_count, o_cp_gamma_count])

        o_cp = np.mean(o_cp_count, axis=1)

        print('\n=======================================================')
        print(f'running time is {time() - start_time:.2f}s.')
        print('-------------------------------------------------------')
        print(f'bias is {bias}.')
        print(f'ase is {ase}; esd is {esd}')
        print(f'cp is {cp}.')
        print('-------------------------------------------------------')
        print(f'origin bias is {origin_bias}')
        print(f'origin ase is {origin_ase}; esd is {esd}')
        print(f'cp is {o_cp}')
        print('=======================================================\n')


    for _ in range(10):
        simulation()
