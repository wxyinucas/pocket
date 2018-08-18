import numpy as np


def flatten(nest_list):
    """Flatten recurrent time to an array"""

    flatten_ = np.array([])

    for i in nest_list:
        flatten_ = np.append(flatten_, i)

    flatten_ = np.sort(flatten_)
    return flatten_


def exp_star_transform(df, a):

    # e_star is vector {exp(x @ a)}
    e_star = np.exp(np.array([df.x1, df.x2]).T @ a)

    # df_star contains {Y^*(a)}
    df_star = df.copy()
    df_star.y = df.y * e_star
    df_star.d = df.d * e_star
    df_star.c = df.c * e_star

    return df_star


def initial(x, scale=0.2):

    x = x + np.random.uniform(-scale, scale)

    return np.array(x)


def beta(result):

    for i in range(len(result)):
        result[i, 0] = result[i, 0] if (-0.9 < result[i, 0] < 1) else initial(-1, 0.025)
        result[i, 1] = result[i, 1] if (0.9 < result[i, 1] < 1) else initial(1, 0.015)

    return result

