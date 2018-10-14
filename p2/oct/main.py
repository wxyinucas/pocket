# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# Time      : 2018/10/12  09:01
# File      : main.py
# Project   : WXY_projects
==================================

"""
import numpy as np

from oct.estimator import Estimator
from time import time
from tqdm import tqdm

if __name__ == '__main__':
    start_time = time()

    # simulation settings
    reduplicates = 10
    true_parameters = np.array([-1, 1, -1, 1])
    a_hat_paras_arr = np.array([])
    b_hat_paras_arr = np.array([])

    # reduplicate procedure
    for _ in tqdm(range(reduplicates)):
        est = Estimator(true_parameters, n_samples=200)

        a_hat = est.a_hat
        a_hat_paras_arr = np.append(a_hat_paras_arr, a_hat)

        b_hat = est.b_hat
        b_hat_paras_arr = np.append(b_hat_paras_arr, b_hat)

    # process outputs
    a_hat_paras = np.mean(np.array(a_hat_paras_arr).reshape((-1, 2)), axis=0)
    b_hat_paras = np.mean(np.array(b_hat_paras_arr).reshape((-1, 2)), axis=0)

    # ase
    a_ase = np.std(np.random.choice(a_hat_paras_arr, size=200, replace=True), axis=1)
    b_ase = np.std(np.random.choice(b_hat_paras_arr, size=200, replace=True), axis=1)

    # std
    a_std = np.std(a_hat_paras_arr, axis=1)
    b_std = np.std(b_hat_paras_arr, axis=1)

    # cp
    a_cp_count =


    # print outputs
    print(f'a :{a_hat_paras}')
    print(f'b :{b_hat_paras}')

    print(f'running time: {time()-start_time:.2f}sec.')
    np.choose()