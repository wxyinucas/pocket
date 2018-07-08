from generator_file import *
from estimator_file import *
from scipy.optimize import fsolve

result_alpha = np.array([0, 0])
result_beta = np.array([0, 0])

for _ in range(200):
    df, r = generator()

    result_alpha = np.append(result_alpha, np.array(fsolve(s_n, [initial(-1, 0.02), initial(1, 0.02)])))
    lambda_hat = lam_a(result_alpha[-2:])
    z_hat = df.m / (lambda_hat + 0.001)
    result_beta = np.append(result_beta, np.array(fsolve(u_n, [initial(-1, 0.02), initial(1, 0.02)])))

result_alpha = result_alpha.reshape([-1, 2])
result_beta = beta(result_beta.reshape([-1, 2]))

np.mean(result_alpha, axis=0), np.mean(result_beta, axis=0)