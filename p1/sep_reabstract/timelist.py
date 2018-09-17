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
    :param T: 观测时间（传入时一定包含0， TAU）
    :param r: 是否观测得到（是否panel data），且len(r) = len(T) - 1


    Test：
    T = [0, 1, 2, 3, 5]  TAU = 5 here
    r = [1, 0, 1, 0]
    t = [0.5, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4]

    Expect result:
    T = [(2, 6)]
    r = [1, 0, 1, 0]
    t = [0.5, 2.5, 3]
    """
    # 检查变量
    assert T[0] == 0
    assert T[-1] == 5
    assert len(r) == len(T) - 1

    # 预处理T&r，生成比较矩阵comp
    comp = compare(t, T)

    # 生成t的location matrix t_loc
    t_leq_T = np.sum(comp, axis=1)
    t_in_T = len(T) - t_leq_T - 1

    # 生成panel_index
    r_0_loc = np.where(r == 0)[0].tolist()

    t_panel_loc = []
    for iters in r_0_loc:
        t_panel_loc.extend(np.where(t_in_T == iters)[0].tolist())

    # 集合作用得到剩余t的location
    t_loc_set = set(range(len(t)))
    t_rec_loc = list(t_loc_set.difference(t_panel_loc))
    t_result = t[t_rec_loc]

    # 生成每个T区间对应的事件发生数量
    n = np.sum(comp, axis=0)

    # 计算r=0对应的T右端点
    r_0_end = (np.array(r_0_loc) + 1).tolist()
    T_result = np.array(list(zip(T[r_0_end], n[r_0_end])))

    if r[-1] == 0:
        T_result = T_result[:-1]

    return t_result, T_result, r


def change_time_matrix(t: np.array, T: np.array, r: np.array):
    """
    封装change_time_list, 使之能处理不规则np.array。
    """
    assert t.shape[0] == T.shape[0] == r.shape[0]

    t_list, T_list, r_list = map(list, [t, T, r])

    for ite in range(len(t_list)):
        t_list[ite], T_list[ite], r_list[ite] = change_time_list(t_list[ite], T_list[ite], r_list[ite])

    return np.array(t_list), np.array(T_list), r


if __name__ == '__main__':
    # test 1
    T = [0, 1, 2, 3, 5]
    r = [1, 0, 1, 0]
    t = [0.5, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4]

    T, r, t = list(map(np.array, [T, r, t]))

    tt, TT, rr = change_time_list(t, T, r)

    assert (tt == np.array([0.5, 2.5, 3])).all()
    # assert TT == {2: 6}
    assert (TT == np.array([[2, 6]])).all()
    assert (rr == [1, 0, 1, 0]).all()

    # todo: 在project中设定全局常量。

    # test 2: 如何传播
    # T_test = np.array([T, T])
    T_test = np.array([T, np.array([0, 1, 5])])
    r_test = np.array([r, np.array([1, 1])])
    t_test = np.array([t, t])

    tt, TT, rr = change_time_matrix(t_test, T_test, r_test)

    print(tt, TT, rr, sep='\n')
