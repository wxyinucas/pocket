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
    每次处理一条数据。

    :param t: 实际发生时间
    :param T: 观测时间（传入时一定包含0， TAU） (传入时只包含TAU，而没有0了！！)
    :param r: 是否观测得到（是否panel data），且len(r) = len(T) - 1
    :returns: 4个分量，分别是——
        1. 被观测到的recurrent event的时间
        2. 被观测到的panel data 的 (ob_time, ob_numbers) pairs。
        3. r indicators
        4. 实际panel data 的 (ob_time, ob_numbers) pairs。

    Test：
    t = [0.5, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4]
    T = [1, 2, 3, 5]  TAU = 5 here
    r = [1, 0, 1, 0]

    Expect result:
    t = [0.5, 2.5, 3]
    T0 = [(2, 6)]
    r = [1, 0, 1, 0]
    T = [(1, 1), (2, 6), (3, 8), (4, 8)]
    """
    # 检查变量
    assert T[-1] == 5
    assert len(r) == len(T)

    # 预处理T&r，生成比较矩阵comp
    comp = compare(t, T)

    # 生成T pairs
    n = np.sum(comp, axis=0)
    T_pairs = np.array(tuple(zip(T, n)))

    # 生成t的location matrix t_loc
    t_leq_T = np.sum(comp, axis=1)
    t_in_T = len(T) - t_leq_T

    # 生成可被观测到的t
    t_mask = r[t_in_T].astype(bool)
    t_ob = t[t_mask]

    # 生成panel data pairs
    T_mask = r.astype(bool)
    T0 = T[~T_mask]
    n0 = n[~T_mask]
    T0_pairs = np.array(tuple(zip(T0, n0)))

    # 删除掉T0中，观测时间为TAU的pair
    if r[-1] == 0:
        T0_pairs = T0_pairs[:-1]

    # 删除最后一个观测区间：除了r=1有意义，且已经被保留，剩下的无所谓了。
    # T_pairs = T_pairs[:-1]
    # r = r[:-1]

    return t_ob, T0_pairs, r, T_pairs


def change_time_matrix(t: np.array, T: np.array, r: np.array):
    """
    封装change_time_list, 使之能处理不规则np.array。

    返回值均是np.array，更易处理。
    """
    assert t.shape[0] == T.shape[0] == r.shape[0]

    t_list, T_list, r_list = map(list, [t, T, r])
    T0_list = list(range(len(t_list)))

    for ite in range(len(t_list)):
        t_list[ite], T0_list[ite], r_list[ite], T_list[ite] = change_time_list(t_list[ite], T_list[ite], r_list[ite])

    t_arr, T0_arr, r_arr, T_arr = list(map(np.array, [t_list, T0_list, r_list, T_list]))
    return  t_arr, T0_arr, r_arr, T_arr


if __name__ == '__main__':
    # test 1
    T = [1, 2, 3, 5]
    r = [1, 0, 1, 0]
    t = [0.5, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4]

    T, r, t = list(map(np.array, [T, r, t]))

    tt, TT0, rr, TT = change_time_list(t, T, r)

    assert (tt == np.array([0.5, 2.5, 3])).all()
    # assert TT == {2: 6}
    assert (TT0 == np.array([[2, 6]])).all()
    # assert (rr == [1, 0, 1, 0]).all()

    # todo: 在project中设定全局常量。

    # test 2: 如何传播
    # T_test = np.array([T, T])
    T_test = np.array([T, np.array([1, 5]), np.array([2, 3, 5])])
    r_test = np.array([r, np.array([1, 1]), np.array([0, 0, 0])])
    t_test = np.array([t, t, t])

    tt, TT0, rr, TT = change_time_matrix(t_test, T_test, r_test)
    """
    理论结果
    ----------------------------------------------
    tt[0] = [0.5, 2.5, 5]
    tt[1] = t
    tt[2] = None
    
    TT0[0] = [(2, 6)]
    TT0[1] = None
    TT0[2] = [(2, 6), (3, 8)]
    
    TT[0] = [(1, 1), (2, 6), (3, 8), (5, 9)]
    TT[1] = [(1, 1), (5, 9)]
    TT[2] = [(2, 6), (3, 8), (5, 9)]
    """
    print(tt, TT0, rr, TT, sep='\n')
