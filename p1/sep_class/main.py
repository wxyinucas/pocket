# -*- coding: utf-8 -*-
__author__ = 'Xiaoyu Wang'

"""
==================================
# @Time      : 2018/9/7  16:20
# @File      : main.py
# @Project   : WXY_projects
==================================

"""
import numpy as np
from tqdm import tqdm
import csv
from p1.sep_class.estimator import Estimator


def simulation(sample_size=100, beta=1, gamma=1, iterations=100, detail=False):
    """
    :return: estimator, esd, ase, cp
    """
    estimators = np.zeros(2)[None, :]
    ases = np.zeros(2)[None, :]
    for _ in range(iterations):
        am = Estimator(sample_size, beta=beta, gamma=gamma)
        am.calculator()
        estimators = np.append(estimators, am.sol.x[None, :], axis=0)
        ases = np.append(ases, am.ASE()[None, :], axis=0)

    # NOW we get the vectors of estimator and ASE
    estimators = np.delete(estimators, 0, axis=0)
    ases = np.delete(ases, 0, 0)

    # Take the mean of vectors above
    estimator = np.mean(estimators, axis=0)
    bias = estimator - np.array([beta, gamma])
    ase = np.mean(ases, axis=0)

    # Next to get the the value of ESD
    esv = np.cov(estimators.T)
    esd = np.sqrt(esv[[0, 1], [0, 1]])

    # Next to calculate CP Statistics
    cp_beta = np.mean(
        ((estimator[0] - 1.96 * ase[0]) < estimators[:, 0]) & (estimators[:, 0] < ((estimator[0] + 1.96 * ase[0]))))
    cp_gamma = np.mean(
        ((estimator[1] - 1.96 * ase[1]) < estimators[:, 1]) & (estimators[:, 1] < ((estimator[1] + 1.96 * ase[1]))))
    cp = np.array([cp_beta, cp_gamma])

    if detail:
        print('BIAS = ', bias)
        print('ESD = ', esd)
        print('ASE = ', ase)
        print('CPs = ', cp)

    return np.array([*bias, *esd, *ase, *cp])


def result(iterations=10):
    with open('./simulation_result.csv', 'w', newline='') as cf:
        fieldnames = ['Sample_size', 'beta', 'gamma', 'beta_bias', 'beta_esd', 'beta_ase', 'beta_cp',
                      'gamma_bias', 'gamma_esd', 'gamma_ase', 'gamma_cp']
        writer = csv.DictWriter(cf, fieldnames=fieldnames)

        writer.writeheader()
        for n in [50, 100]:
            writer.writerow({'Sample_size': n})
            for i in [0, 0.5, 1]:
                for j in [0, 0.5, 1]:
                    for _ in range(5):
                        result = simulation(sample_size=n, beta=i, gamma=j, iterations=iterations)
                        reorder = result[[0, 2, 4, 6, 1, 3, 5, 7]]
                        o = [format(i, '0.4f') for i in reorder]  # o for output
                        dictionary = {'beta': i if j == 0 else '',
                                      'gamma': j,
                                      'beta_bias': o[0],
                                      'beta_esd': o[1],
                                      'beta_ase': o[2],
                                      'beta_cp': o[3],
                                      'gamma_bias': o[4],
                                      'gamma_esd': o[5],
                                      'gamma_ase': o[6],
                                      'gamma_cp': o[7]}
                        writer.writerow(dictionary)
                    writer.writerow({})


if __name__ == '__main__':
    result(10)
