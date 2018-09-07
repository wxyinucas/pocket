# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/9/7  10:00
# @File      : estimator.py
# @Project   : WXY_projects
==================================

"""
from scipy import optimize
from p1.sep_class.generator import *
import csv
import numpy as np


class Estimator(object):
    """
    n_sample , beta, gamma, x_gen, z_gen
    """

    def __init__(self, n_sample, beta=1, gamma=1):

        # init parameters
        self.beta = beta
        self.gamma = gamma
        self.x_gen = x_gen
        self.z_gen = z_gen
        # init list
        self.c = []
        self.z = []
        self.x = []
        self.time_list = []

        # generate data
        self.put_in(n_sample)

        # generate Y and \tilda{Y}
        self.making_Y()

    def put_in(self, n_sample):
        # a Bridge between 'Record' and 'Estimator'
        for _ in range(n_sample):
            temp = Record(self.beta, self.gamma, self.x_gen, self.z_gen)
            # the loop is for test
            if temp.time_list:
                self.c.append(temp.dict['c'])
                self.z.append(temp.dict['z'])
                self.x.append(temp.dict['x'])
                self.time_list.append(np.array(temp.dict['time_list']))

        self.c = np.array(self.c)
        self.z = np.array(self.z)
        self.x = np.array(self.x)
        self.time_list = np.array(self.time_list)

        # print(len(self.c))

        index = np.argsort(self.c)
        # re_arrange z, x,c, q_i
        self.c = self.c[index]
        self.z = self.z[index]
        self.x = self.x[index]
        self.time_list = self.time_list[index]

        self.num = len(self.c)
        self.length = np.array([len(i) for i in self.time_list])

        # calculate diff c
        self.diff_c = np.insert(np.diff(self.c), 0, self.c[0])

    def making_Y(self, detail=False):
        # create a lower triangular matrix
        self.tri = np.tri(self.num)

        ## This Y is \tilda{Y} in calculation
        self.Y = np.array([np.zeros(self.num)])
        for i in range(self.num):
            temp = np.array([(k < self.c).astype(int) for k in self.time_list[i]])
            self.Y = np.append(self.Y, temp, axis=0)
        self.Y = np.delete(self.Y, 0, 0)  # 1 indicate which vector to delete, 0 is axis.

        self.index = np.cumsum(self.length)
        self.index = np.insert(self.index, 0, 0)

        if detail:
            print('The shape of Y suppose to be ({0}, {1}), \n'
                  'while the actual shape is {2} '.
                  format(np.sum(self.length), self.num, self.Y.shape))

    def making_matrix(self, x, detail=False):
        # x[0]: beta
        # x[1]: gamma
        self.q0 = np.exp(x[1] * self.x)

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
                self.num, self.Z_t.shape))
            print('The shape of Z_N suppose to be ({0},{1}), while the actual shape is {2}'.format(
                np.sum(self.length), self.num, self.Z_N.shape))

    def equation(self, x):
        ## S = A - B
        self.making_matrix(x=x)

        self.A_z = 0
        self.A_x = 0

        for i in range(self.num):
            self.A_z += np.sum(self.Z_N[self.index[i]:self.index[i + 1], i])
            self.A_x += np.sum(self.X_N[self.index[i]:self.index[i + 1], i])

        self.B_z = self.z @ (self.tri * self.Z_t.T) @ self.diff_c * x[0]
        self.B_x = self.z @ (self.tri * self.X_t.T) @ self.diff_c * x[0]

        result = np.array([self.A_z - self.B_z, self.A_x - self.B_x])  # / self.num
        return result

    def calculator(self):
        # self.sol = optimize.root(self.equation, np.array([self.beta, self.gamma]),
        #                          method='Krylov')
        self.sol = optimize.root(self.equation, np.array([0, 0]),
                                 method='Krylov')
        return np.array(self.sol.x)

    def detail(self):
        print("A_z", self.A_z)
        print("B_z", self.B_z)
        print("A_x", self.A_x)
        print("B_x", self.B_x)

    def data(self, show=False):
        if show:
            print('z = ', self.z)
            print('x = ', self.x)
            print('c = ', self.c)
            print('life_time = ', self.time_list)
        return self.c, self.z, self.x, self.time_list

    ####################Line before Variance Estimating#######################

    def get_hat(self, x):
        # x[0]: hat_beta
        # x[1]: hat_gamma
        self.beta_h = x[0]
        self.gamma_h = x[1]

    def A_bb(self):
        result = self.z @ (self.tri * self.Z_t.T) @ self.diff_c
        return result

    def A_bg(self):
        Alpha = np.sum(((self.Y * self.Z_N) @ (self.q0 * self.x)) /
                       (self.Y @ self.q0))
        Beta = self.beta_h * self.z @ self.tri @ \
               (self.diff_c * ((self.tri.T * self.Z_t) @ (self.q0 * self.x) /
                               (self.tri.T @ self.q0)))

        return Alpha - Beta

    def A_gb(self):
        result = self.z @ (self.tri * self.X_t.T) @ self.diff_c
        return result

    def A_gg(self):
        Alpha = np.sum(((self.Y * self.X_N) @ (self.q0 * self.x)) /
                       (self.Y @ self.q0))
        Beta = self.beta_h * self.z @ self.tri @ \
               (self.diff_c * ((self.tri.T * self.X_t) @ (self.q0 * self.x) /
                               (self.tri.T @ self.q0)))

        return Alpha - Beta

    def A_matrix(self):
        result = np.array([[self.A_bb(), self.A_bg()], [self.A_gb(), self.A_gg()]]) \
                 / self.num
        return result

    def V_matrix(self, detail=False):
        temp = 0
        for i in range(self.num):
            alpha = self.Z_N[self.index[i]:self.index[i + 1], i][None, :]
            delta = self.X_N[self.index[i]:self.index[i + 1], i][None, :]
            zeta = np.append(alpha, delta, axis=0)
            temp += zeta @ zeta.T
        if detail:
            print('shape of alpha is', alpha.shape)
            print('shape of zeta is ', zeta.shape)
            print('Shape of temp = ', temp.shape)
        return temp / self.num

    def ASE(self):
        """
        1. self.calculator
        2. self.get_hat(self.sol.x)
        3. self.making_matrix([self.beta_h, self.gamma_h])
        """
        self.get_hat(self.sol.x)
        self.making_matrix([self.beta_h, self.gamma_h])

        A = self.A_matrix()
        V = self.V_matrix()

        asv = np.linalg.inv(A) * V * np.linalg.inv(A).T
        ase = np.sqrt(asv[[0, 1], [0, 1]])
        return ase / np.sqrt(self.num)

    def compare(self):
        com_data = []
        for time in self.time_list:
            if np.random.rand() > 0.8:
                com_data.append(time)
            else:
                com_data.append(time[[0, -1]])

        self.time_list = np.array(com_data)

