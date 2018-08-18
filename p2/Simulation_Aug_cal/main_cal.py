from p2.Simulation_Aug_cal.generator import generate
from p2.Simulation_Aug_cal.value_estimator import s_n,  load, tmp_u_n
from p2.Simulation_Aug_cal.true_value import SIZE
from p2.Simulation_Aug_cal.utils import initial
from scipy.optimize import fsolve
from tqdm import tqdm
from time import time
from examples.simAnneal_dev import SimAnneal
from examples.simAnneal_dev import OptSolution

import sys
import numpy as np

# 估计a_hat
a_hat_seq = []
b_hat_seq = []
for _ in tqdm(range(10)):
    df, r = generate(SIZE)
    load(df, r)

    targ = SimAnneal()
    init = -sys.maxsize  # for minimum case
    # init = sys.maxsize # for maximum case
    xyRange = [[-2, 2], [-2, 2]]
    xRange = [[0, 10]]
    t_start = time()

    calculate = OptSolution(Markov_chain=1000, result=init, val_nd=[0, 0])
    output = calculate.solution(SA_newV=targ.newVar, SA_juge=targ.juge,
                                juge_text='max', ValueRange=xyRange, func=s_n)

    '''
    with open('out.dat', 'w') as f:
        for i in range(len(output)):
            f.write(str(output[i]) + '\n')
    '''
    t_end = time()
    print('Running %.4f seconds' % (t_end - t_start))

    # alpha
    a_hat_cur = np.array(output[0])
    a_hat_seq = np.append(a_hat_seq, a_hat_cur)

    # beta
    u_n = tmp_u_n(a_hat_cur)

    output = calculate.solution(SA_newV=targ.newVar, SA_juge=targ.juge,
                                juge_text='max', ValueRange=xyRange, func=u_n)
    b_hat_cur = np.array(output[0])
    b_hat_seq = np.append(b_hat_seq, b_hat_cur)


a_hat_seq = a_hat_seq.reshape([-1, 2])
b_hat_seq = b_hat_seq.reshape([-1, 2])
a_hat = np.mean(a_hat_seq, axis=0)
b_hat = np.mean(b_hat_seq, axis=0)
