import numpy as np

from tda.ellipsoidal.solvers.kfunction import compute_K_gradient_fast, compute_K_fast, get_A_operations_fast


def cauchy_simplex_solver(eps, xs, A_list, max_iters=200, tol=1e-6):
    """
    Implements the Cauchy-Simplex algorithm for minimizing K(λ) over the probability simplex,
    using the simplified gradient.

    :param eps:
    :param xs:
    :param A_list:
    :param max_iters:
    :param tol:
    :return:
    """
    k, D = xs.shape
    A_inv_array, x_Ainv_x = get_A_operations_fast(A_list, xs)
    lmbd = np.ones(k) / k

    for t in range(max_iters):
        grad = compute_K_gradient_fast(lmbd, xs, A_inv_array)
        grad_mean = np.dot(lmbd, grad)
        direction = lmbd * (grad - grad_mean)
        step_size = min(1.0, 1.0 / np.max(np.abs(grad - grad_mean)))
        lmbd = lmbd - step_size * direction
        lmbd = np.maximum(lmbd, 1e-10)
        lmbd /= np.sum(lmbd)
        if np.linalg.norm(direction) < tol:
            result = {"lambda": lmbd, "K_min": compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x), "success": True}
            return result
    result = {"lambda": lmbd, "K_min": compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x), "success": False}
    return result
