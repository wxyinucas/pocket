# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'
"""
==================================
# @Time      : 2018/8/23  下午4:39
# @File      : save_df.py
# @Project   : WXY_Project
==================================

Test save and load df
"""

import pandas as pd
import numpy as np

test_df = pd.DataFrame(np.arange(12).reshape(3, -1), index=['a', 'b', 'c'], columns=['A', 'B', 'C', 'D'])
print(test_df)
test_df.to_csv('df.csv')
test_1_df = pd.read_csv('df.csv', index_col=0)
print(test_1_df)

test_df['dic'] = [{}, {}, {}]
test_1_df.to_dict()
pd.DataFrame(test_1_df.to_dict())
