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
from utils import compare, flatten

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

    def cal_equation(self, gamma, beta):
        """
        计算估计方程，gamma，beta均为估计值。

        对照笔记中，A B C D 四个部分。
        """

        exp = np.exp(gamma * self.x)

        def A():
            tmp = np.sum(self.q * self.m, axis=1)
            assert tmp.shape[0] == 2

            return tmp / self.n

        def B():
            tmp = np.sum(self.z * self.c * self.q, axis=1)
            assert tmp.shape[0] == 2

            return beta * tmp / self.n

        def C():
            t_c_matrix = compare(flatten(self.t), self.c)
            x_part_tmp = np.sum(t_c_matrix * exp * self.x, axis=1) / np.sum(t_c_matrix * exp, axis=1)
            x_part = np.sum(x_part_tmp, axis=0)

            assert x_part_tmp.shape[0] == flatten(self.t).shape[0]
            assert type(x_part) == np.float64

            z_part_tmp = np.sum(t_c_matrix * exp * self.z, axis=1) / np.sum(t_c_matrix * exp, axis=1)
            z_part = np.sum(z_part_tmp, axis=0)

            tmp = np.array([x_part, z_part])

            return tmp / self.n

        def D():
            # 每个tmp都是笔记中矩形框中部分
            def factor(loc):
                """
                生成笔记中向量x_part_j每个entry。
                """
                num = loc + 1
                
                x_part_tmp = np.sum(np.tri(num) * exp * self.x, axis=1) / np.sum(np.tri(num) * exp, axis=1)
                x_part_vec = x_part_tmp * self.dt[:num]
            
                assert x_part_tmp.shape[0] == num
                assert x_part_vec.shape[0] == num

                z_part_tmp = np.sum(np.tri(num) * exp * self.z, axis=1) / np.sum(np.tri(num) * exp, axis=1)
                z_part_vec = z_part_tmp * self.dt[:num]

                return x_part_vec, z_part_vec

            factor(1)

        return A() - B() - C() + D()


if __name__ == '__main__':
    test = Estimator(1, 1)
    test.cal_equation(1, 1)

