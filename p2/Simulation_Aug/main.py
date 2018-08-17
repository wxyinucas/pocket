from p2.Simulation_Aug.generator import generate
from p2.Simulation_Aug.value_estimator import s_n, df_b_transform, load
from p2.Simulation_Aug.true_value import SIZE
from p2.Simulation_Aug.utils import initial
from scipy.optimize import fsolve
from tqdm import tqdm

import numpy as np

# 估计a_hat
a_hat_seq = []
for _ in tqdm(range(20)):
    df, r = generate(SIZE)
    load(df, r)

    # alpha
    a_hat_cur = np.array(fsolve(s_n, np.array([initial(-1), initial(1)])))
    a_hat_seq = np.append(a_hat_seq, a_hat_cur)

    # beta


a_hat_seq = a_hat_seq.reshape([-1, 2])
a_hat = np.mean(a_hat_seq, axis=0)
