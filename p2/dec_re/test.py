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
from est_1220 import Estimator
from scipy.stats import trim_mean

# est = Estimator(np.array([-1, 1, -1, 1]))
# # est.sn_prototype(est.raw_dt, est.raw_t, [-1, 1])
# print(est.estimate([-1, 1, -1, 1]))
n_rep = 50

hats = np.empty((n_rep, 4))
for i in range(n_rep):
    est = Estimator(np.array([-1, 1, -1, 1]), 40)
    # est.sn_prototype(est.raw_dt, est.raw_t, [-1, 1])
    hats[i], _ = est.estimate([-2, 1, -2, 1])  # todo 和真值无关，估计正确
    print(hats[i])

print(trim_mean(hats, 0.05, axis=0))
