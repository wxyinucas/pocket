# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# Time      : 2018/10/18  14:19
# File      : simulation.py
# Project   : WXY_projects
==================================

"""
import numpy as np
import os
import sys

from oct.estimator import Estimator
from time import time
from tqdm import tqdm
from oct.utils import mean_std_cp, cal_ase
from itertools import product


def simulation(true_parameters, title=None):
    # simulation settings
    reduplicates = 100

    a_hat_paras_list = []
    b_hat_paras_list = []

    # reduplicate procedure
    for _ in tqdm(range(reduplicates)):
        est = Estimator(true_parameters, n_samples=200)

        a_hat = est.a_hat
        a_hat_paras_list.append(a_hat)

        b_hat = est.b_hat
        b_hat_paras_list.append(b_hat)

    # process outputs
    a_arr = np.array(a_hat_paras_list)
    a1_arr = a_arr[:, 0]
    a2_arr = a_arr[:, 1]

    b_arr = np.array(b_hat_paras_list)
    b1_arr = b_arr[:, 0]
    b2_arr = b_arr[:, 1]

    arrs = [a1_arr, a2_arr, b1_arr, b2_arr]

    a1_ase, a2_ase, b1_ase, b2_ase = list(map(cal_ase, arrs))

    a1_mean, a1_std, a1_cp = mean_std_cp(a1_arr, a1_ase)
    a2_mean, a2_std, a2_cp = mean_std_cp(a2_arr, a2_ase)
    b1_mean, b1_std, b1_cp = mean_std_cp(b1_arr, b1_ase)
    b2_mean, b2_std, b2_cp = mean_std_cp(b2_arr, b2_ase)

    # print outputs
    # print('\n====================Results======================')
    # print('Bias:')
    # print(f'a1 :{a1_mean - true_parameters[0]:0.3f}')
    # print(f'a2 :{a2_mean - true_parameters[1]:0.3f}')
    # print(f'b1 :{b1_mean - true_parameters[2]:0.3f}')
    # print(f'b2 :{b2_mean - true_parameters[3]:0.3f}')
    # print('\n ASE & std:')
    # print(f'a1 ase: {a1_ase:0.3f}, \t  std: {a1_std:0.3f}')
    # print(f'a2 ase: {a2_ase:0.3f}, \t  std: {a2_std:0.3f}')
    # print(f'b1 ase: {b1_ase:0.3f}, \t  std: {b1_std:0.3f}')
    # print(f'b2 ase: {b2_ase:0.3f}, \t  std: {b2_std:0.3f}')
    # print('\n CP:')
    # print(f'a1 cp: {a1_cp}')
    # print(f'a2 cp: {a2_cp}')
    # print(f'b1 cp: {b1_cp}')
    # print(f'b2 cp: {b2_cp}')

    # Latex
    a1_bias = a1_mean - true_parameters[0]
    a2_bias = a2_mean - true_parameters[1]
    b1_bias = b1_mean - true_parameters[2]
    b2_bias = b2_mean - true_parameters[3]

    a1_str = f'& {a1_bias} & {a1_std} & {a1_ase} & {a1_cp} '
    a2_str = f'& {a2_bias} & {a2_std} & {a2_ase} & {a2_cp} '
    b1_str = f'& {b1_bias} & {b1_std} & {b1_ase} & {b1_cp} '
    b2_str = f'& {b2_bias} & {b2_std} & {b2_ase} & {b2_cp} \\\\'

    if not title:
        title = ' & & & '
    print(title + a1_str + a2_str + b1_str + b2_str)


if __name__ == '__main__':
    start_time = time()
    true_parameters = np.array([-1, 1, -1, 1])

    console = sys.stdout
    try:
        os.remove('latex.txt')
    except FileNotFoundError:
        pass

    # table title
    first_part = '$\\alpha_1$ & $\\alpha_2$ & $\\beta_1$ & $\\beta_2$'
    last_part = '& \multicolumn{4}{c}{$\\alpha_1$} & \multicolumn{4}{c}{$\\alpha_2$} & ' \
                '\multicolumn{4}{c}{$\\beta_1$} & \multicolumn{4}{c}{$\\beta_2$} \\\\'

    # table body
    one = [-1, 0, 1]
    two = [-1, 1]

    for true_parameters in product(one, two, one, two):
        with open('./latex.txt', 'a+') as f:
            sys.stdout = f
            file_start = os.SEEK_CUR
            print('\n================================================')
            new_para_title = f'{true_parameters[0]} & alpha2: {true_parameters[1]} &' \
                             f' {true_parameters[2]} & {true_parameters[3]}'

            simulation(true_parameters, new_para_title)
            for _ in range(2):
                simulation(true_parameters)

print(f'running time: {time()-start_time:.2f}sec.')
