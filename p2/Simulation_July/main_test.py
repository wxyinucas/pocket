from generator import *

PARAS = {'alpha0': [-1, 1],
         'alpha1': [-1, 1],
         'beta0': [-1, 1],
         'beta1': [-1, -1]}

set_paras(**PARAS)
show_paras(1)

df_tmp, r_tmp = generate()