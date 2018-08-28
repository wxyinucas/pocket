import numpy as np
import pandas as pd
from p2.Simulation_Aug_print.utils import flatten
from p2.Simulation_Aug_print.true_value import SIZE


def var_load(_df, _r, _a):
    global df, r, a, a1, a2
    df = _df
    r = _r
    a = _a
    a1 = _a[:2]
    a2 = _a[-2:]


def pre_process():
    """
    a1: estimate parameter
    """
    global df, r, df_star, r_star, a1

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


def compute_lambda(XI):
    """
    a1: estimate parameter
    """
    global lambda_a, df, a

    # N
    n = np.sum(df.m)

    # Next to sort {sl}
    sl = flatten(r_star)

    # {rl}
    compare = df_star.y.values.reshape((1, -1)) < sl.reshape((-1, 1))
    rl = np.arange(n) + 1 - compare * XI @ df.m.values

    # index
    index = np.sum((1 - compare).T, axis=1)

    # vector {\hat{\Lambda}_n}
    rl = np.append(rl, np.inf)
    factor = 1 - XI / rl[tuple([index])]
    lambda_ = np.cumprod(factor[::-1])[::-1]

    # vector{\lambda(Y^*(a1))}
    lambda_a = lambda_


def compute_mu(XI):
    global mu_z, a1, a2

    mu_z = XI * df_star.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) @ (1 / lambda_a) / SIZE


def equ_a_1(XI):
    global a1, a2
    result = [XI * df.x1.values @ (df.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) / lambda_a - mu_z).values,
              XI * df.x2.values @ (df.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) / lambda_a - mu_z).values]
    return np.array(result) / SIZE


def equ_a_2(XI):
    """
    Also known as g_2(t;X, a)
    """
    global a1, a2
    r_sum = np.array([sum(row) for row in r])
    result = [r_sum * XI * df.x1.values @ (df.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) / lambda_a - mu_z).values,
              r_sum * XI * df.x2.values @ (df.m * np.exp(- df[['x1', 'x2']] @ (a2 - a1)) / lambda_a - mu_z).values]
    return np.array(result) / SIZE


def var_s_n(XI):
    pre_process()
    compute_lambda(XI)
    compute_mu(XI)
    result_vec = np.array([equ_a_1(XI), equ_a_2(XI)])

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
