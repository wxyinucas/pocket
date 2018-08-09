#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==============================
# @Time    : 2018/8/4 15:20
# @Author  : Xiaoyu Wang
# @File    : doc__test.py
===============================

"""


def sum_naturals(n):
    """Return the sum of the first n natural numbers.

    >>> sum_naturals(10)
    55
    >>> sum_naturals(100)
    5050
    """
    total, k = 0, 1
    while k <= n:
        total, k = total + k, k + 1
    return total


if __name__ == '__main__':
    from doctest import testmod

    testmod()

