import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from generator_file import *
from functions import *



def preprocess(a):
    '''
    a: estimate parameter
    '''
    global df, r, df_star, r_star

    df, r = generator()

    # 乘法因子
    e_star = np.exp(np.array([df.x1, df.x2]).T @ a)

    # justify df^*
    df_star = df.copy()
    df_star.y = df.y * e_star
    df_star.c = df.c * e_star
    df_star.d = df.d * e_star

    # {t_ij^*(a)}
    r_star = r * e_star


#     return df, r, df_star, r_star


def compute_lambda(a):
    '''
    a: estimate parameter
    '''
    global lambda_a

    # N
    N = np.sum(df.m)

    # Next to sort {sl}
    sl = flatten(r_star)

    # {rl}
    compare = df_star.y.values.reshape((1, -1)) < sl.reshape((-1, 1))
    rl = np.arange(N) + 1 - compare @ df.m.values

    # index
    index = np.sum((1 - compare).T, axis=1)

    # vector {\hat{\Lambda}_n}
    factor = 1 - 1 / rl
    factor = np.append(factor, 1)
    lambda_ = np.cumprod(factor[::-1])[::-1]

    # vector{\lambda(Y^*(a))}
    lambda_a = lambda_[[index]]


# def compute_mu(a):


def s_n(a):
    preprocess(a)
    compute_lambda(a)
    part2 = df.m @ (1 / lambda_a) / size
    result = [df.x1.values @ (df.m / lambda_a - part2).values, df.x2.values @ (df.m / lambda_a - part2).values]
    return np.array(result) / size