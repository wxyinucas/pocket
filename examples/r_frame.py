#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==============================
# @Time    : 2018/8/4 14:20
# @Author  : Xiaoyu Wang
# @File    : r_frame.py
===============================

"""


def r_frame(x):
    if x > 0:
        x -= 1
        r_frame(x)
    return x

if __name__ == '__main__':
    print(r_frame(10))