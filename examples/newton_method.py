#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==============================
# @Time    : 2018/8/5 12:50
# @Author  : Xiaoyu Wang
# @File    : newton_method.py
===============================

"""


# Newton method 模块
def improve(update, close, guess=1):
    while not close(guess):
        guess = update(guess)
    return guess


def newton_update(f, df):
    def update(x):
        return x - f(x) / df(x)

    return update


def find_zero(f, df):
    def near_zero(x):
        return approx_eq(f(x), 0)

    return improve(newton_update(f, df), near_zero)


def approx_eq(x, y, tol=1e-15):
    return abs(x - y) < tol


# 测试模块
def power(x, n):
    """Return x * x * x * ... * x for x repeated n times."""
    product, k = 1, 0
    while k < n:
        product, k = product * x, k + 1
    return product


def nth_root_of_a(n, a):
    def f(x):
        return power(x, n) - a

    def df(x):
        return n * power(x, n - 1)

    return find_zero(f, df)


if __name__ == '__main__':
    print(nth_root_of_a(3, 64))
