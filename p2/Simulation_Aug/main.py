from generator import *
from value_estimator import *
from scipy.optimize import fsolve

PARAS = {'alpha0': [-1, 1],
         'alpha1': [-1, 1],
         'beta0': [-1, 1],
         'beta1': [-1, -1],
         'size': 200}


set_paras(**PARAS)
print('The parameters setting is as follows:')
show_paras(1)
print('-------------------------')


a_hat = np.array([np.nan, np.nan])
for _ in range(50):
    df, r = generate(PARAS['size'])
    s_n([-1, 1])