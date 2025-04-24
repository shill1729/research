import matplotlib.pyplot as plt

from tda.toydata.pointclouds import generate_point_cloud_and_As
from solvers.kfunction import compute_K_fast,compute_K_hessian_fast, compute_K_gradient_fast, get_A_operations_fast
from solvers.scipy_solver import minimize_K
from tda.toydata.plotting import plot_K_surface

import numpy as np



n = 3
seed = 3
x, A_list = generate_point_cloud_and_As(n, seed)
A_array = np.array(A_list)
lambda_grid = np.linspace(0, 1, 30)
epsilon = 0.1
A_inv_array, x_Ainv_x = get_A_operations_fast(A_array, x)
K_func = lambda lam: compute_K_fast(lam, epsilon, x, A_inv_array, x_Ainv_x)
result = minimize_K(epsilon, x, A_array)
opt1 = np.array([result['lambda'][0], result['lambda'][1], result['K_min']])
fig = plot_K_surface(K_func, lambda_grid, [opt1])
plt.show()
