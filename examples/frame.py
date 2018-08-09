#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==============================
# @Time    : 2018/8/4 13:48
# @Author  : Xiaoyu Wang
# @File    : frame.py
===============================

"""

from operator import add, mul


def square(x):
    x += 1
    x -= 1
    return mul(x, x)


def sum_squares(x, y):
    return add(square(x), square(y))


result = sum_squares(5, 12)
