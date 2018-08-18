import numpy as np
import pandas as pd
from p2.Simulation_Aug_cal.utils import flatten
from p2.Simulation_Aug_cal.true_value import SIZE


def load(_df, _r):
    global df, r
    df = _df
    r = _r


def pre_process(a: np.array):
    """
    a: estimate parameter
    """
    global df, r, df_star, r_star

    # df, r = generate(SIZE)

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
    """
    a: estimate parameter
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

    # vector{\lambda(Y^*(a))}
    lambda_a = lambda_[tuple([index])]
    # _lambda_a = pd.Series(lambda_a, name='lambda_a')
    # df = pd.concat([df, _lambda_a], axis=1)


def compute_mu(a):
    global mu_z

    mu_z = df_star.m @ (1 / lambda_a) / SIZE


def s_n(a):
    pre_process(a)
    compute_lambda(a)
    compute_mu(a)
    result = [df.x1.values @ (df.m / lambda_a - mu_z).values, df.x2.values @ (df.m / lambda_a - mu_z).values]
    return np.array(result) / SIZE


###################
#
#
# Beta
#
#
####################


def df_b_transform(b: np.array, a_hat: np.array):
    global df, df_b, df_1b
    df_b = df.copy()

    # e_star is vector {exp(x @ a)}
    e_star = np.exp(np.array([df.x1, df.x2]).T @ b)

    # justify df^*
    df_b.y = df.y * e_star
    df_b.c = df.c * e_star
    df_b.d = df.d * e_star

    # calculate \Lambda_n{\Y_i^*(\hat{a})}
    pre_process(a_hat)
    compute_lambda(a_hat)
    _lambda_a = pd.Series(lambda_a, name='lambda_a')
    df_b = pd.concat([df_b, _lambda_a], axis=1)

    # 整理df_1b: 去除不合理数据
    index = (df.delta == 1) & (df_b.y < 10)
    df_1b = df_b[index].copy()


def cal_z_hat():
    global df_1b, z_hat

    z_hat = df_1b.m / df_1b.lambda_a


def raw_u_n(b, a_hat):
    """Make sure calling df_b_transform() before this function"""

    global df, df_b, df_1b

    df_b_transform(b, a_hat)
    cal_z_hat()

    compare = df_1b.y.values.reshape((-1, 1)) < df_1b.y.values.reshape((1, -1))

    part11 = np.sum(df_1b.x1) / SIZE
    part21 = np.sum(df_1b.x2) / SIZE

    part12 = np.sum((df_1b.x1 * z_hat @ compare) / (z_hat @ compare)) / SIZE
    part22 = np.sum((df_1b.x2 * z_hat @ compare) / (z_hat @ compare)) / SIZE

    result = np.array([part11 - part12, part21 - part22])

    return result


def tmp_u_n(a_hat):
    def u_n(b):
        return raw_u_n(b, a_hat)

    return u_n
