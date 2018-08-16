import numpy as np
import pandas as pd
from p2.Simulation_Aug.true_value import *


def generate(size):
    """
    While in generate procedure, we put PARAS in function,
        and set Covariances variables 'x'.

    Next, Latent variable 'x' was produced, then corresponding Censor time 'c',
        failure time 'd'. Thus Indicator 'delta' and Observed time 'y'.

    At last, we produce the Times of recurrent time 'm' happened and  exact
        When if happened 'r'.
    :param size: In one iteration, the number of samples which are observed.
    :return: pd.df which contains all information.
    """
    # Fixed terms
    lambda0 = 0.2
    h0 = 1 / 50
    tau = 10

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
                                   np.exp(x @ BETA0)),
                              size)

    # indicator
    delta = (d < c).astype(np.int32)
    y = d * delta + c * (1 - delta)
    temp = y < tau
    y = y * temp + (1 - temp) * tau

    # recurrent time
    m = np.random.poisson(z * lambda0 *
                          np.exp(x @ ALPHA0) * y,
                          size)
    r = []
    for i in range(size):
        r.append(np.random.uniform(0, y[i], m[i]))
    r = np.array(r)

    NAMES = ['x1', 'x2', 'z', 'c', 'd', 'delta', 'y', 'm']
    VALUES = [x1, x2, z, c, d, delta, y, m]

    df = pd.DataFrame(VALUES).T
    df.columns = NAMES

    return df, r


if __name__ == "__main__":
    print(generate(SIZE))
