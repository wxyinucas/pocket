from p2.Simulation_Aug.generator import generate
from p2.Simulation_Aug.value_estimator import s_n,  load, tmp_u_n
from p2.Simulation_Aug.true_value import SIZE, ALPHA1, BETA1, ALPHA2
from p2.Simulation_Aug.utils import initial
from scipy.optimize import fsolve
from tqdm import tqdm
from time import time


import numpy as np

# 估计a_hat
a_hat_seq = []
b_hat_seq = []

start_time = time()
for _ in tqdm(range(20)):
    df, r = generate(SIZE)
    load(df, r)

    # alpha
    a0 = np.append([ALPHA1, ALPHA2])
    a_hat_cur = np.array(fsolve(s_n, initial(a0)))
    a_hat_seq = np.append(a_hat_seq, a_hat_cur)

    # beta
    u_n = tmp_u_n(a_hat_cur[:2])
    b_hat_cur = np.array(fsolve(u_n, np.array(initial(BETA1))))
    b_hat_seq = np.append(b_hat_seq, b_hat_cur)

print(f'The running time is {time() - start_time:.2f}s.')
a_hat_seq = a_hat_seq.reshape([-1, 4])
b_hat_seq = b_hat_seq.reshape([-1, 2])
a_hat = np.mean(a_hat_seq, axis=0)
b_hat = np.mean(b_hat_seq, axis=0)


with open('./table.txt', 'a+') as f:
    tmp = np.append(a_hat, b_hat)
    string = [f'{num:.4f}' for num in tmp]

    print(*string, sep=' & ', file=f)
    f.seek(0)
    print(f.readlines())