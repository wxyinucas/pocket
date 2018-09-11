# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/9/9  17:08
# @File      : utils.py
# @Project   : WXY_projects
==================================

"""
import numpy as np


def compare(v1, v2):
    assert len(v1.shape) == 1 or v1.shape[1] == 1, f'v1 wrong shape {v1.shape}.'
    assert len(v2.shape) == 1 or v2.shape[0] == 1, f'v2 wrong shape {v2.shape}.'

    return (v1.reshape([-1, 1]) <= v2.reshape([1, -1])).astype(np.int32)


def poisson_process(terminal, num) -> np.array:
    """
    用于order statistic method生成poisson分布。

    :param terminal: 删失或最终观测时间。
    :param num: 每次观测到的记录总数。
    :return: matrix
    """
    assert terminal.shape[0] == num.shape[0]

    t_list = []

    for ite in range(len(num)):
        tmp = np.random.uniform(0, terminal[ite], num[ite])
        tmp.sort()
        t_list.append(tmp)

    return np.array(t_list)


def indicator(num, pr) -> np.array:
    """
    :param num: 每次观测到的记录总数。
    :return: matrix
    """

    r_list = []

    for ite in range(len(num)):
        tmp = np.random.binomial(1, pr, num[ite])
        r_list.append(tmp)

    return np.array(r_list)


def modify_ob(T, r):
    """
    用于修正数据结构以适合time_list
    """
    assert T.shape[0] == r.shape[0]
    T_tmp, r_tmp = map(list, [T, r])

    for ite in range(len(T)):
        T_tmp[ite] = np.array([0, *T_tmp[ite], 5])  # TAU = 5 here
        r_tmp[ite] = np.array([*r_tmp[ite], 0])

    T_result, r_result = map(np.array, [T_tmp, r_tmp])

    return T_result, r_result


if __name__ == '__main__':
    # v1 = np.array([4, 6, 1])
    # v2 = np.arange(5)
    # print(compare(v1, v2))
    #
    # terminal = np.array([4, 4, 5, 5, 6, 6])
    num = np.arange(6) + 1
    # print(poisson_process(terminal, num))

    print(indicator(num, 0.5))

    # T = np.array([[1, 2], [2, 3]])
    # r = np.array([[1, 1], [1, 0]])
    # print(modify_ob(T, r))
