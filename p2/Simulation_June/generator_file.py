# To generate one observation, and display it.

import numpy as np
import pandas as pd
import csv
import matplotlib as mlt


def setting(_alpha=[-1, 1], _beta=[-1, 1], _size=200):
    global alpha, beta, size
    alpha = np.array(_alpha)
    beta = np.array(_beta)
    size = _size


def generator():
    """
    The setting of generator is from cat book.
    :return:  Data frame  and recurrent time list
    """

    # Parameters
    # alpha = np.array([-1, 1])
    # beta = np.array([-1, 1])

    # Fixed terms
    lambda0 = 0.2
    h0 = 1 / 50
    tau = 10
    # size = 200

    # Covariances
    x1 = np.random.binomial(1, 0.95, size)
    # x1 = np.array([1] * size)
    x2 = np.random.uniform(0, 1, size)
    # x2 = np.array([1] * size)
    x = np.array([x1, x2]).T

    # Latent
    z = np.random.gamma(2, 5, size)

    # Censor time
    c = np.array([])
    for i in range(size):
        if x1[i] == 1:
            c = np.append(c,
                          np.random.exponential(10))
        else:
            c = np.append(c,
                          np.random.exponential(300 / z[i] ** 2))

    # failure time
    d = np.random.exponential(1 / (z * h0 *
                                   np.exp(x @ beta)),
                              size)

    # indicator
    delta = (d < c).astype(np.int32)
    y = d * delta + c * (1 - delta)
    temp = y < tau
    y = y * temp + (1 - temp) * tau
    # y == np.min(np.append(d, c).reshape([2,-1]).T, axis = 1)

    # recurrent time
    m = np.random.poisson(z * lambda0 *
                          np.exp(x @ alpha) * y,
                          size)
    r = []
    for i in range(size):
        r.append(np.random.uniform(0, y[i], m[i]))

    r = np.array(r)
    # r = np.random.uniform(0, y, m)
    # r.sort()

    names = ['x1', 'x2', 'z', 'c', 'd', 'delta', 'y', 'm']
    values = [x1, x2, z, c, d, delta, y, m]

    df = pd.DataFrame(values).T
    df.columns = names

    return df, r
