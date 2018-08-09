#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==============================
# @Time    : 2018/8/4 16:43
# @Author  : Xiaoyu Wang
# @File    : high_order_func.py
===============================

"""


def summation(n, term):
    total, k = 0, 1
    while k <= n:
        total, k = total + term(k), k + 1
    return total


def cube(x):
    return x * x * x


def sum_cubes(n):
    return summation(n, cube)


result = sum_cubes(3)
