# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/9/9  16:47
# @File      : timelist.py
# @Project   : WXY_projects
==================================

由于观测数据比较复杂，定义一个new function来处理这种数据结构。
"""
import numpy as np
from utils import compare


def change_time_list(t: np.array, T: np.array, r: np.array):
    """
    用于simulation，将生成的数据更改为观测到的格式。

    :param t: 实际发生时间
    :param T: 观测时间
    :param r: 是否观测得到（是否panel data）


    Test：
    T = [1, 2, 3]
    r = [1, 0, 1]
    t = [0.5, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3]

    Expect result:
    T = [(2: 5)]
    r = [1, 0, 1]
    t = [0.5, 2.5, 3]
    """
    T = np.append(0, T)
    comp = compare(T, t)


