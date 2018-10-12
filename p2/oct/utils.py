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


def flatten(t):
    """
    将复合np.array 压平。
    """
    tmp = []
    for ite in t:
        tmp.extend(ite.tolist())
    return np.array(tmp)


def c_generate(x_arr, z_arr):
    """按照协变量的不同生成不同的删失时间。"""

    def c_scale(x, z):
        assert 0 <= x <= 1

        if x < 0.5:
            return np.random.exponential(10)
        else:
            return 200 / (z ** 2)

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


def proper_d(d_arr):
    def d_scale(d):

        if d:
            return d
        else:
            return np.array([11])

    tmp = np.vectorize(d_scale, signature='()->()')

    return tmp(d_arr)


def make_a_star(time_stack, cov1_arr):
    """将时间做一个指数变换。"""
    assert time_stack.shape == cov1_arr.shape
    tmp = np.zeros_like(time_stack)

    for i in range(len(time_stack)):
        tmp[i] = time_stack[i] * cov1_arr[i]
        assert tmp[i].shape == time_stack[i].shape

    return tmp


def find_loc_scale(arr: np.array, num) -> int:
    """arr is sorted. Find the location of num in arr."""
    assert (np.sort(arr) == arr).all()

    tmp = np.append(arr, num)
    tmp.sort()
    loc = np.where(tmp == num)
    return loc[0][0]


def find_loc_arr(arr: np.array, nums) -> np.array:

    vec_func = np.vectorize(find_loc_scale, signature='(n),()->()')

    return vec_func(arr, nums)


if __name__ == '__main__':
    # 修正死亡时间，将空缺的（未观测到的）用11替换（TAU=10）
    tmp_a = np.array([7])
    tmp_b = np.array([])
    tmp_c = np.array([11])
    d = np.array([tmp_a, tmp_a, tmp_b, tmp_b, tmp_a])
    assert (proper_d(d) == np.array([7, 7, 11, 11, 7])).all()

    # 测试make_a_star
    a = np.array([1, 2, 3])
    b = np.array([2, 3])
    c = np.array([])
    d = np.array([4, 3])
    t = np.array([a, b, c, d])
    a1 = np.array([1, 1, 1, 1])
    print(make_a_star(t, a1))  # the prints should be same
    print(np.array([a, b, c, d]))

    # 测试find_loc_scale
    arr = np.array([0, 0.03, 0.05, 1.4])
    num = 1.2
    assert find_loc_scale(arr, num) == 3

    # 测试find_loc_arr
    nums = np.array([0.01, 1.2])
    assert (find_loc_arr(arr, nums) == np.array([1, 3])).all()