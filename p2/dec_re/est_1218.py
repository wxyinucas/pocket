# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2018-12-18  12:42
# File      : est_1218.py
# Project   : WXY_projects
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

设计框架
"""
from utils import *
from data import Data


class Estimator:

    def __init__(self, parameters, n_sample=200):
        gen_data = Data(parameters, n_sample)

        self.raw_dt = gen_data.dt
        self.raw_t = gen_data.t  # stack r

    def main(self):
        """
        TODO:  用于储存结果的最终函数；方程的求解也在此进行。
        1. 全部data估计出的est
        2. 10次抽样后得到的variance

        :return: est var
        """

    def sn(self, df, a, type=0):
        """
        TODO: 估计方程S_n。
        :param df: 用于估计的数据，总数据、或其子集
        :param a: 用于本次计算的估计值
        :param type: 用于区分计算估计方程还是输出向量lambda{Y*(a^)}。
        :return: 估计方程的结果，或者是用a计算出的lambda估计向量。
        """

        a1, a2 = a
        exp = np.exp(df.x * (a2 - a1))

        def make_star():
            """生成本次估计alpha过程中，需要变换的时间变量。"""
            star = np.exp(df.x * a1)
            y = make_a_star(df.y, star)
            t = make_a_star(df.t, star)
            return y, t

        def s2r():
            """
            TODO
            给定一个变换后的时间， 得到对应的r(l)
            注意：是vectorize的
            """

        def lam():
            """
            TODO 生成lambda关于Y(a)的向量
            用于在本次循环中生成向量lambda{Y*(a)}
            用于在估计beta时传过去lambda{Y*(a^)}
            :return:
            """

        def cal_uz():
            """
            Todo
            计算u_z, 返回一个实数
            """

        # 组合上面的组件，进行计算，得到方程即可。
        return None

    def hn(self, df, a_hat, b):
        """
        TODO 估计方程h_n
        :return: 估计方程的结果。
        """

        a1, a2 = a_hat
        b1, b2 = b
        exp = np.exp(df.x * (b2 - b1))
        lam_arr = self.sn(df, a_hat, type=1)

        def bar():
            """
            TODO how to calculate q_bar
            :return:
            """

    def shuffle(self, raw_df):
        """
        Todo
        :return: 返回重抽样后的df
        """
