# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# Time      : 2018/10/10  10:50
# File      : utils.py
# Project   : WXY_projects
==================================

"""
import numpy as np


def compare(v1, v2):
    assert len(v1.shape) in [0, 1] or v1.shape[1] == 1, f'v1 wrong shape.'
    assert len(v2.shape) in [0, 1] or v2.shape[0] == 1, f'v2 wrong shape.'

    return (v1.reshape([-1, 1]) <= v2.reshape([1, -1])).astype(np.int32)


def c_generate(x_arr, z_arr):
    """按照协变量的不同生成不同的删失时间。"""
    def c_scale(x, z):
        assert 0 <= x <= 1

        if x < 0.5:
            return np.random.exponential(10)
        else:
            return 300 / (z ** 2)

    c_arr = np.vectorize(c_scale, signature='(),()->()')

    return c_arr(x_arr, z_arr)


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
