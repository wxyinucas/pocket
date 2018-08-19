import numpy as np
import pandas as pd
from numpy.linalg import norm
from p2.Simulation_Aug_print.utils import flatten
from p2.Simulation_Aug_print.true_value import SIZE


def load(_df, _r):
    global df, r
    df = _df
    r = _r


def pre_process(a1: np.array):
    """
    a1: estimate parameter
    """
    global df, r, df_star, r_star

    # 乘法因子
    e_star = np.exp(np.array([df.x1, df.x2]).T @ a1)

    # justify df^*
    df_star = df.copy()
    df_star.y = df.y * e_star
    df_star.c = df.c * e_star
    df_star.d = df.d * e_star

    # {t_ij^*(a1)}
    r_star = r * e_star


#     return df, r, df_star, r_star


def compute_lambda(a1):
    """
    a1: estimate parameter
    """
    global lambda_a, df

    # N
    n = np.sum(df.m)

    # Next to sort {sl}
    sl = flatten(r_star)

    # {rl}
    compare = df_star.y.values.reshape((1, -1)) < sl.reshape((-1, 1))
    rl = np.arange(n) + 1 - compare @ df.m.values

    # index
    index = np.sum((1 - compare).T, axis=1)

    # vector {\hat{\Lambda}_n}
    factor = 1 - 1 / rl
    factor = np.append(factor, 1)
    lambda_ = np.cumprod(factor[::-1])[::-1]

    # vector{\lambda(Y^*(a1))}
    lambda_a = lambda_[tuple([index])]


def compute_mu(a1: np.array, a2: np.array):
    global mu_z

    mu_z = df_star.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) @ (1 / lambda_a) / SIZE


def equ_a_1(a1, a2):
    result = [df.x1.values @ (df.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) / lambda_a - mu_z).values,
              df.x2.values @ (df.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) / lambda_a - mu_z).values]
    return np.array(result) / SIZE


def equ_a_2(a1: np.array, a2: np.array):
    """
    Also known as g_2(t;X, a)
    """
    r_sum = np.array([sum(row) for row in r])
    result = [r_sum * df.x1.values @ (df.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) / lambda_a - mu_z).values,
              r_sum * df.x2.values @ (df.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) / lambda_a - mu_z).values]
    return np.array(result) / SIZE


def s_n(a: np.array):
    a1 = a[:2]
    a2 = a[-2:]
    pre_process(a1)
    compute_lambda(a1)
    compute_mu(a1, a2)
    result_vec = np.array([equ_a_1(a1, a2), equ_a_2(a1, a2)])
    result_vec_named = pd.DataFrame(result_vec, index=['equation1', 'equation2'], columns=['a1', 'a2'])
    return flatten(result_vec)


###################
#
#
# Beta
#
#
####################


def df_b_transform(b1: np.array, a1_hat: np.array):
    global df, df_b, df_1b, r_1b
    df_b = df.copy()

    # e_star is vector {exp(x @ a1)}
    e_star = np.exp(np.array([df.x1, df.x2]).T @ b1)

    # justify df^*
    df_b.y = df.y * e_star
    df_b.c = df.c * e_star
    df_b.d = df.d * e_star

    # calculate \Lambda_n{\Y_i^*(\hat{a1})}
    pre_process(a1_hat)
    compute_lambda(a1_hat)
    _lambda_a = pd.Series(lambda_a, name='lambda_a')
    df_b = pd.concat([df_b, _lambda_a], axis=1)

    # 整理df_1b: 去除不合理数据
    index = (df.delta == 1) & (df_b.y < 10)
    df_1b = df_b[index].copy()
    r_1b = r[index]


def cal_z_hat():
    global df_1b, z_hat

    z_hat = df_1b.m / df_1b.lambda_a


def equ_b_1(b1, b2, compare):
    result = [
        np.sum(df_1b.x1.values - (df_1b.x1 * z_hat * np.exp(- df_1b[['x1', 'x2']] @ (b2 - b1)) @ compare) / (
                    z_hat @ compare)),
        np.sum(df_1b.x2.values - (df_1b.x2 * z_hat * np.exp(- df_1b[['x1', 'x2']] @ (b2 - b1)) @ compare) / (
                    z_hat @ compare))]
    return np.array(result) / SIZE


def equ_b_2(b1, b2, compare):
    result = [
        np.sum(df_1b.y * (df_1b.x1 - (df_1b.x1 * z_hat * np.exp(- df_1b[['x1', 'x2']] @ (b2 - b1)) @ compare) / (
                z_hat @ compare))),
        np.sum(df_1b.y * (df_1b.x2 - (df_1b.x2 * z_hat * np.exp(- df_1b[['x1', 'x2']] @ (b2 - b1)) @ compare) / (
                z_hat @ compare)))]
    return np.array(result) / SIZE


def raw_u_n(b, a1_hat):
    """Make sure calling df_b_transform() before this function"""

    global df, df_b, df_1b

    b1 = b[:2]
    b2 = b[-2:]
    df_b_transform(b1, a1_hat)
    cal_z_hat()

    compare = df_1b.y.values.reshape((-1, 1)) < df_1b.y.values.reshape((1, -1))

    result = np.array([equ_b_1(b1, b2, compare), equ_b_2(b1, b2, compare)])

    return flatten(result)


def tmp_u_n(a1_hat):
    def u_n(b):
        return raw_u_n(b, a1_hat)

    return u_n
