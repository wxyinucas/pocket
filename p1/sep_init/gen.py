# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/9/7  11:04
# @File      : gen.py
# @Project   : WXY_projects
==================================

重构后的生成模型
"""

import numpy as np


class Record:
    """
    This class generator one life observation record.
    Return a dictionary.

    "time_table": The first small list of the recurrent time.
                  The second list is the observation time.
                  The number is whether the precess is observed or not.

    "c"
    "z"
    "n_interval"
    """

    def __init__(self, beta=0, gamma=0, x_gen='1', z_gen='1'):
        """
        :param pr: the probability of time interval been observed
        :param beta: real value of beta
        """
        self.beta = beta
        self.gamma = gamma
        self.x_gen = x_gen
        self.z_gen = z_gen

        # some fixed parameters
        self.tau = 5

        # some values we interest with
        self.c = 0

        # some list
        self.time_list = []  # the real occur time

        # generate
        self.gen()

    def gen(self):

        # np.random.seed(42)
        # generate some values
        self.c = np.random.uniform(self.tau / 2, self.tau)
        self.z = np.random.uniform(0, 1)
        self.x = np.random.uniform(0, 1)
        self.time_list = [0]

        # initial life time before censoring
        while self.time_list[-1] < self.c:
            # scale = 1 / lambda
            # dmu_0 = 1
            # if 可加可乘
            self.time_list.append(self.time_list[-1] + np.random.exponential(scale=1 /
                                                                                   (np.exp(
                                                                                       self.x * self.gamma) + self.beta * self.z)))
        else:
            self.time_list.pop(-1)
            self.time_list.pop(0)

        # put all result together
        self.dict = {'time_list': self.time_list,
                     'c': self.c,
                     'z': self.z,
                     'x': self.x}

    def show(self):
        """Show the detail of one record."""
        print('The censor time is {} \n'.format(self.c))
        print('The parameter are beta = {}, gamma = {}'.format(self.beta, self.gamma))
        print('The covariant are z = {}, x = {}\n'.format(self.z, self.x))
        print('The time list is: \n {} \n'.format(self.time_list))
