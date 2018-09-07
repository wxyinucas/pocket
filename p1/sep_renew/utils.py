# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/9/9  17:08
# @File      : utils.py
# @Project   : WXY_projects
==================================

"""
import numpy as np


def compare(v1, v2):
    assert len(v1.shape) == 1 or v1.shape[1] == 1, f'v1 wrong shape {v1.shape}.'
    assert len(v2.shape) == 1 or v2.shape[0] == 1, f'v2 wrong shape {v2.shape}.'

    return (v1.reshape([-1, 1]) <= v2.reshape([1, -1])).astype(np.int32)


if __name__ == '__main__':
    v1 = np.array([4, 6, 1])
    v2 = np.arange(5)
    print(compare(v1, v2))
