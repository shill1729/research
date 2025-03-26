import numpy as np
from scipy.linalg import cho_solve
from tda.solvers.kfunction import compute_K_gradient_fast, compute_K_fast, compute_K_hessian_fast, get_A_operations_fast


def project_to_simplex_deprecated(v):
    """
    Project a vector v onto the probability simplex Δ_k.

    :param v:
    :return:
    """
    v_sorted = np.sort(v)[::-1]
    v_cumsum = np.cumsum(v_sorted)
    rho = np.where(v_sorted - (v_cumsum - 1) / (np.arange(len(v)) + 1) > 0)[0][-1]
    theta = (v_cumsum[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def project_to_simplex(v):
    u = v.copy()
    u_sum = u.sum()
    if u_sum <= 1.0 + 1e-6:  # Already on simplex
        return np.clip(u, 0, None)
    u -= (u_sum - 1.0) / len(u)
    return np.maximum(u, 0)


def projected_gradient_descent(eps, xs, A_list, step_size=0.001, max_iters=5000, tol=1e-10):
    """
    Minimize K(λ) using projected gradient descent onto the probability simplex.

    :param eps:
    :param xs:
    :param A_list:
    :param A_inv_array:
    :param x_Ainv_x:
    :param step_size:
    :param max_iters:
    :param tol:
    :return:
    """
    k, D = xs.shape
    lmbd = np.ones(k) / k

    A_inv_array, x_Ainv_x = get_A_operations_fast(A_list, xs)
    for _ in range(max_iters):
        grad = compute_K_gradient_fast(lmbd, xs, A_inv_array)
        lmbd_new = project_to_simplex(lmbd - step_size * grad)
        if np.linalg.norm(lmbd_new - lmbd) < tol:
            break
        lmbd = lmbd_new

    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    # The centroid is given by a linear system B m = S
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    # try:
    #     m_lambda = np.linalg.solve(A_lambda_inv, S)
    # except np.linalg.LinAlgError:
    #     m_lambda = None
    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    try:
        L = np.linalg.cholesky(A_lambda_inv)
        m_lambda = cho_solve((L, True), S)  # Backsubstitution
    except np.linalg.LinAlgError:
        m_lambda = None

    return {
        'lambda': lmbd,
        'K_min': compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x),
        'm_lambda': m_lambda,
        'A_lambda': np.linalg.inv(A_lambda_inv) if m_lambda is not None else None
    }









