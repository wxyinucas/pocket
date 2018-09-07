# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/8/28  上午12:15
# @File      : value_change.py
# @Project   : WXY_Project
==================================

"""
import os

size_range = [200, 500]
num_range = [0, 0.5, 1]


def change(size, odd_end, even_end):
    try:
        os.remove('./true_value.py')
    except FileNotFoundError:
        pass

    with open('./true_value.py', 'w+') as w:
        w.write('import numpy as np\n')

        w.write(f'SIZE = {size}\n')

        paras = f'''
ALPHA1 = np.array([-1, {odd_end}])
ALPHA2 = np.array([-1, {even_end}])
BETA1 = np.array([-1, {odd_end}])
BETA2 = np.array([-1, {even_end}])
'''
        w.write(paras)


if __name__ == '__main__':
    change(200, 0, 1)
