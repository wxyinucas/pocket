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
    assert len(v1.shape) in [0, 1] or v1.shape[1] == 1, f'v1 wrong shape {v1.shape}.'
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
    :param num: 每次观测到的T的总数，为了填补[T, tau]，需要再加1。
    :return: matrix
    """

    r_list = []

    for ite in range(len(num)):
        tmp = np.random.binomial(1, pr, num[ite] + 1)
        r_list.append(tmp)

    return np.array(r_list)


def modify_ob(T):
    """
    用于修正数据结构以适合time_list
    """
    T_tmp = list(T)

    for ite in range(len(T)):
        T_tmp[ite] = np.array([0, *T_tmp[ite], 5])  # TAU = 5 here, 因为c不一定观测的到

    T_result = np.array(T_tmp)

    return T_result


def flatten(t):
    """
    将复合np.array 压平。
    """
    tmp = []
    for ite in t:
        tmp.extend(ite.tolist())
    return np.array(tmp)


def separate(T):
    """
    将T分解成T_ij 和 N，但保持分组
    """
    n = T.shape[0]

    time = []
    count = []
    for i in range(n):
        if T[i].shape[0]:
            time.append(T[i][:, 0])
            count.append(T[i][:, 1])
        else:
            time.append(np.array([]))
            count.append(np.array([]))

    return np.array(time), np.array(count)


def r_i(time_arr, T_arr, r_arr):
    """
    给定时间向量time_arr， 判断每一个是否可观测。
    返回值是与time_arr等长的0，1arr。
    """
    assert len(time_arr.shape) in [0, 1] or time_arr.shape[1] == 1, f'time_arr wrong shape {time_arr.shape}.'
    assert len(r_arr.shape) == 1 or r_arr.shape[0] == 1, f'r_arr wrong shape {r_arr.shape}.'
    assert r_arr.shape[0] == T_arr.shape[0] - 1

    com = compare(time_arr, T_arr)
    t_in_T_index = len(T_arr) - np.sum(com, axis=1)
    t_r_marker = r_arr[t_in_T_index]

    assert t_in_T_index.shape == time_arr.shape
    return t_r_marker


if __name__ == '__main__':
    # compare
    # v1 = np.array([4, 6, 1])
    # v2 = np.arange(5)
    # print(compare(v1, v2))

    # poisson_process
    # terminal = np.array([4, 4, 5, 5, 6, 6])
    # num = np.arange(6) + 1
    # print(poisson_process(terminal, num))

    # indicator
    # print(indicator(num, 0.5))

    # modify
    # T = np.array([[1, 2], [2, 3]])
    # r = np.array([[1, 1], [1, 0]])
    # print(modify_ob(T, r))

    # flatten
    # r = np.array([[1, 1], [1, 0]])
    # print(flatten(r))
    # T_ = np.array([np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 2], [3, 4], [5, 6]])])
    # print(flatten(T_))

    # separate
    # T = np.array([np.array([]), np.array([[2, 3], [3, 4], [5, 2]])])
    # print(separate(T))

    # r_i
    t = np.arange(4) + 0.5
    T = np.array([0, 1, 2, 3, 4, 5])
    r = np.array([0, 1, 0, 1, 0])
    print(r_i(t, T, r))
