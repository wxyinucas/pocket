# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# Time      : 2018/9/12  16:31
# File      : test_gen.py
# Project   : WXY_projects
==================================

"""
import numpy as np

dts = []
for _ in range(1000):
    m = np.random.poisson(10000)
    t = np.random.uniform(0, 10000, m)
    t.sort()
    dt = np.diff(np.array([0, *t]))
    dts.extend(dt.tolist())

print(np.mean(dts), np.var(dts))

dts = []
for _ in range(1000):
    t = 0
    while t < 5:
        t_cur = np.random.exponential(1)
        t += t_cur
        dts.append(t_cur)
    # dts.pop()

print(np.mean(dts), np.var(dts))

dts = np.random.exponential(1, 1000)
print(np.mean(dts), np.var(dts))
