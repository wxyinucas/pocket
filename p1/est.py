# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/9/7  11:03
# @File      : est.py
# @Project   : WXY_projects
==================================

重构后的估计方程
"""

from scipy import optimize
from p1.__generator import *
import time
import csv
import numpy as np
from p1.true_value import BETA, GAMMA


def load(n_sample=200):
    """
    Load simulated data.
    :param n_sample:
    :return:
    """
    global c, x, z, time_list, num, length, diff_c
    c = []
    z = []
    x = []
    time_list = []

    for _ in range(n_sample):
        temp = Record(BETA, GAMMA)
        # the loop is for test
        if temp.time_list:
            c.append(temp.dict['c'])
            z.append(temp.dict['z'])
            x.append(temp.dict['x'])
            time_list.append(np.array(temp.dict['time_list']))

    c = np.array(c)
    z = np.array(z)
    x = np.array(x)
    time_list = np.array(time_list)

    num = len(c)
    length = np.array([len(i) for i in time_list])
    diff_c = np.insert(np.diff(c), 0, c[0])


def making_y():
    global tri, Y, index
    global num, length, time_list
    # create a lower triangular matrix
    tri = np.tri(num)

    # This Y is \tilda{Y} in calculation
    Y = np.array([np.zeros(num)])
    for i in range(num):
        temp = np.array([(k < c).astype(int) for k in time_list[i]])
        Y = np.append(Y, temp, axis=0)
    Y = np.delete(Y, 0, 0)  # 1 indicate which vector to delete, 0 is axis.

    index = np.cumsum(length)
    index = np.insert(index, 0, 0)

    assert Y.shape == (np.sum(length), len(num)), f'Wrong Y shape {Y.shape} != ({np.sum(length)}, {len(num)})'


def making_matrix(para_hat):
    # para[0]: beta_hat
    # para[1]: gamma_hat
    global x, z, tri, Y, num, length
    global q0, Z_t, Z_N, X_t, X_N

    q0 = np.exp(para_hat[1] * x)

    # Zt is [z_k - \bar{z}(c_j-)]_{(j,k)}
    Z_cj = (tri.T @ (q0 * z)) / (tri.T @ q0)
    Z_t = z[None, :] - Z_cj[:, None]

    # Z_N is [z_k - \bar{z}(t_ij)]_{(ij,k)}, whose dimension is (l,n)
    Z_tij = (Y @ (q0 * z)) / (Y @ q0)
    Z_N = z[None, :] - Z_tij[:, None]

    X_cj = (tri.T @ (q0 * x)) / (tri.T @ q0)
    X_t = x[None, :] - X_cj[:, None]
    X_tij = (Y @ (q0 * x)) / (Y @ q0)
    X_N = x[None, :] - X_tij[:, None]

    assert Z_t.shape == (num, num), \
        f'The shape of Z_t suppose to be ({num},{num}), while the actual shape is {Z_t.shape}.'

    assert Z_N.shape == (np.sum(length), num), \
        f'The shape of Z_N suppose to be ({np.sum(length)},{num}), while the actual shape is {Z_N.shape}.'


# TODO:下午一个小时全部改完！


def equation(para_hat):
    """
    Estimating function.
    """
    # S = A - B
    global index, tri, Z_t, X_t, diff_c

    making_matrix(para_hat=para_hat)

    A_z = 0
    A_x = 0

    for i in range(num):
        A_z += np.sum(Z_N[index[i]:index[i + 1], i])
        A_x += np.sum(X_N[index[i]:index[i + 1], i])

    B_z = z @ (tri * Z_t.T) @ diff_c * x[0]
    B_x = z @ (tri * X_t.T) @ diff_c * x[0]

    result = np.array([A_z - B_z, A_x - B_x])  # / num
    return result


def calculator():
    """
    Call this function to estimate parameters.
    
    :return: values of the estimator.
    """
    global sol

    sol = optimize.root(equation, np.array([BETA, GAMMA]), method='Krylov')
    return np.array(sol.x)


def get_hat(para_hat):
    # para_hat[0]: hat_beta
    # para_hat[1]: hat_gamma
    global beta_h, gamma_h

    beta_h = para_hat[0]
    gamma_h = para_hat[1]


def A_bb():
    global z, tri, Z_t, diff_c

    result = z @ (tri * Z_t.T) @ diff_c
    return result


def A_bg():
    global Y, Z_N, q0, x, Y, tri, diff_c

    Alpha = np.sum(((Y * Z_N) @ (q0 * x)) / (Y @ q0))
    Beta = beta_h * z @ tri @  (diff_c * ((tri.T * Z_t) @ (q0 * x) / (tri.T @ q0)))

    return Alpha - Beta


def A_gb():
    global Y, Z_N, q0, x, Y, tri, diff_c

    result = z @ (tri * X_t.T) @ diff_c
    return result


def A_gg():
    global Y, Z_N, q0, x, Y, tri, diff_c

    Alpha = np.sum(((Y * X_N) @ (q0 * x)) /
                   (Y @ q0))
    Beta = beta_h * z @ tri @ \
           (diff_c * ((tri.T * X_t) @ (q0 * x) /
                      (tri.T @ q0)))

    return Alpha - Beta


def A_matrix():
    global num

    result = np.array([[A_bb(), A_bg()], [A_gb(), A_gg()]]) / num
    return result

def V_matrix():
    global Z_N, X_N, num, index

    temp = 0
    for i in range(num):
        alpha = Z_N[index[i]:index[i + 1], i][None, :]
        delta = X_N[index[i]:index[i + 1], i][None, :]
        zeta = np.append(alpha, delta, axis=0)
        temp += zeta @ zeta.T

    return temp / num

def ASE():
    """
    1. calculator
    2. get_hat(sol.x)
    3. making_matrix([beta_h, gamma_h])
    """
    get_hat(sol.x)
    making_matrix([beta_h, gamma_h])

    A = A_matrix()
    V = V_matrix()

    asv = np.linalg.inv(A) * V * np.linalg.inv(A).T
    ase = np.sqrt(asv[[0, 1], [0, 1]])
    return ase / np.sqrt(num)


def compare():
    global time_list

    com_data = []
    for time in time_list:
        if np.random.rand() > 0.8:
            com_data.append(time)
        else:
            com_data.append(time[[0, -1]])

    time_list = np.array(com_data)