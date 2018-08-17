from p2.Simulation_Aug.generator import generate
from p2.Simulation_Aug.value_estimator import s_n,  load, tmp_u_n
from p2.Simulation_Aug.true_value import SIZE
from p2.Simulation_Aug.utils import initial
from scipy.optimize import fsolve
from tqdm import tqdm

import numpy as np

# 估计a_hat
a_hat_seq = []
b_hat_seq = []
for _ in tqdm(range(20)):
    df, r = generate(SIZE)
    load(df, r)

    # alpha
    a_hat_cur = np.array(fsolve(s_n, np.array([initial(-1), initial(1)])))
    a_hat_seq = np.append(a_hat_seq, a_hat_cur)

    # beta
    u_n = tmp_u_n(a_hat_cur)
    b_hat_cur = np.array(fsolve(u_n, np.array([initial(-1), initial(1)])))
    b_hat_seq = np.append(b_hat_seq, b_hat_cur)


a_hat_seq = a_hat_seq.reshape([-1, 2])
b_hat_seq = b_hat_seq.reshape([-1, 2])
a_hat = np.mean(a_hat_seq, axis=0)
b_hat = np.mean(b_hat_seq, axis=0)
