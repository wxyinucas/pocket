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
from scipy.optimize import fsolve, root
from time import time
from tqdm import tqdm


def simulation(redupicates=10):
    true_parameters = np.array([-1, 1, -1, 1])
    a_hat_paras_list = []
    b_hat_paras_list = []

    for _ in tqdm(range(redupicates)):
        est = Estimator(true_parameters, n_samples=200)

        a_hat = est.a_hat
        a_hat_paras_list.append(a_hat)

        b_hat = est.b_hat
        b_hat_paras_list.append(b_hat)

    a_hat_paras = np.mean(np.array(a_hat_paras_list), axis=0)
    b_hat_paras = np.mean(np.array(b_hat_paras_list), axis=0)
    print(f'a :{a_hat_paras}')
    print(f'b :{b_hat_paras}')


if __name__ == '__main__':
    start_time = time()
    simulation()
    print(f'running time: {time()-start_time:.2f}sec.')
