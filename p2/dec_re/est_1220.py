# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2018-12-20  10:47
# File      : est_1220.py
# Project   : WXY_projects
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================


实现细节
"""
from utils import *
from data import Data

from scipy.optimize import root
import numpy as np
import pandas as pd


# np.random.seed = 42


class Estimator:

    def __init__(self, parameters, n_sample=200):
        gen_data = Data(parameters, n_sample)

        self.raw_dt = gen_data.dt
        self.raw_t = gen_data.t  # stack r
        self.n_sam = n_sample

    def estimate(self, guess):
        """
        TODO:  仅输出最终结果；debug也只修改此处代码。
        1. 全部data估计出的est
        2. 10次抽样后得到的variance

        :return: est var
        """
        # sn = lambda a: self.sn_prototype(self.raw_dt, self.raw_t, [-1, a])[1]
        # a_hat = root(sn, [1]).x
        a_hat = np.array([0, 0])

        hn = lambda b: self.hn_prototype(self.raw_dt, self.raw_t, a_hat, b)
        b_hat = root(hn, guess[2:]).x
        # b_hat = np.array([0, 0])

        # result
        hat = np.append(a_hat, b_hat)
        std = [0, 0, 0, 0]
        return hat, std

    def sn_prototype(self, df, raw_t, a, type=0):
        """
        TODO: 改成迭代求解。
        :param df: 用于估计的数据，总数据、或其子集
        :param a: 用于本次计算的估计值
        :param type: 用于区分计算估计方程还是输出向量lambda{Y*(a^)}。
        :return: 估计方程的结果，或者是用a计算出的lambda估计向量。
        """

        def make_star():
            """
            生成本次估计alpha过程中，需要变换的时间变量。
            最后得到的还是stack的arr。
            """
            star = np.exp(df.x * a1)
            y = make_a_star(df.y, star)
            t = make_a_star(raw_t, star)
            return y, t

        def s2r():
            """
            给定一个变换后的时间， 得到对应的r(l) arr
            以及相应的 {s(l) < y_i}
            注意：是vectorize的
            """
            sl = flatten(t)
            sl.sort()

            def rl_row(sl_scale):
                counter = 0
                for i in range(len(y)):
                    if sl_scale < y[i]:
                        counter = counter + np.sum(t[i] <= sl_scale)  # 怎么表达此处的stack r star？
                return counter

            rl = np.vectorize(rl_row, signature='()->()')

            rl_arr = rl(sl)
            y_in_s = np.sum(compare(sl, y), axis=0)
            assert all(rl_arr > 0)
            assert y_in_s.shape == y.shape

            return rl_arr, y_in_s

        def lam():
            """
            用于在本次循环中生成向量lambda{Y*(a)}
            用于在估计beta时传过去lambda{Y*(a^)}
            :return:
            """

            rl, y_in_s = s2r()

            factor = 1 - (1 / rl)
            factor = np.append(factor, 1)  # todo uncertain?
            assert factor.shape[0] == len(rl) + 1

            lam_arr = np.cumprod(factor[::-1])[::-1]
            lam_arr[np.where(lam_arr == 0)] = 1
            assert all(lam_arr > 0)

            return lam_arr[y_in_s]  # todo 对应关系?

        def cal_uz():
            """
            计算u_z, 返回一个实数
            """
            res = (df.m * exp) @ (1 / l_arr)
            assert res.shape == ()
            return res / n

        # 初值设定
        a1, a2 = a
        assert abs(a1) < 100 and abs(a2) < 100

        exp = np.exp(df.x * (a1 - a2))
        n = self.n_sam

        # 组合上面的组件，进行计算，得到方程即可。
        y, t = make_star()
        l_arr = lam()
        uz = cal_uz()

        # 以(2, )向量作为结果输出
        sn1 = df.x @ (exp * df.m * (1 / l_arr) - uz)
        sn2 = (y * df.x) @ (exp * df.m * (1 / l_arr) - uz) / n  # todo uncertain?

        if type == 0:
            return np.array([sn1, sn2]) / n
        else:
            return l_arr

    def hn_prototype(self, df, t, a_hat, b):
        """
        TODO 估计方程h_n
        :return: 估计方程的结果。
        """

        def integral(arr, weight, mask):
            res = (arr[mask] - bar(arr, weight))
            return np.mean(res, axis=0)

        def z_est():
            """返回估计 z 的 arr"""
            return df.m * (1 / np.exp(df.x * (a2 - a1))) * (1 / l_arr)

        # 初始设定
        a1, a2 = a_hat
        b1, b2 = b

        n = self.n_sam
        mask = df.delta == 1
        exp = np.exp(df.x * (b2 - b1))
        l_arr = self.sn_prototype(df, t, a_hat, type=1)

        # 中间变量
        z_hat = z_est()
        y = make_a_star(df.y, np.exp(df.x * b1))
        weights = (z_hat * exp).values[None, 1] * compare(y, y[mask]).astype(bool)

        # 以(2, )向量作为结果输出
        hn1 = integral(df.x, weights, mask)
        hn2 = integral(df.x * y, weights, mask)

        return np.array([hn1, hn2]) / n

    def shuffle(self, raw_df):
        """
        Todo
        :return: 返回重抽样后的df
        """
