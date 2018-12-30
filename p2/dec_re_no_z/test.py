# -*- coding: utf-8 -*-
"""
==================================
# Time      : 2018-12-18  14:22
# File      : test.py
# Project   : WXY_projects
# Author    : Wang Xiaoyu
# Contact   : wxyinucas@gmail.com
==================================

"""
import numpy as np
import pandas as pd
from est_1230 import Estimator
from scipy.stats import trim_mean

# est = Estimator(np.array([-1, 1, -1, 1]))
# # est.sn_prototype(est.raw_dt, est.raw_t, [-1, 1])
# print(est.estimate([-1, 1, -1, 1]))
n_rep = 50

hats = np.empty((n_rep, 3))
for i in range(n_rep):
    est = Estimator(np.array([-1, 1, -1, 1]), 40)
    # est.sn_prototype(est.raw_dt, est.raw_t, [-1, 1])
    hats[i], _ = est.estimate([0])  # todo 和真值无关，估计正确
    if any(np.abs(hats[i]) > 5):
        print('wait')
    print(f'{hats[i]} ' + 'and' + f' {sum(est.l_arr < 0.01)}')

result = trim_mean(hats, 0.05, axis=0)
print(result)

# 在估计a_2时，bias 大约为1.6
# 至少可以得出，存在l_arr < 0.01情况时，会出现很大的异常值；但显然有异常值a=>异常值l_arr原因待定


