import numpy as np
from tda.solvers.kfunction import (
    compute_K_fast,
    compute_K_gradient_fast,
    get_A_operations_fast
)

def minimize_K_fw(eps, xs, A_array, max_iter=100, tol=1e-9):
    """
    Minimizes K(λ) = eps^2 - C(λ) over the probability simplex using
    the Frank–Wolfe algorithm. Uses the “fast” routines from tda.solvers.kfunction.

    :param eps: float
        The radius/persistence parameter.
    :param xs: ndarray, shape (k, d)
        The centers x_i.
    :param A_array: ndarray, shape (k, d, d)
        Stack of covariance matrices A(x_i) for each center.
    :param max_iter: int
        Maximum number of FW iterations.
    :param tol: float
        Tolerance for objective change to declare convergence.
    :return: dict with keys
        'lambda'    : ndarray (k,), the final weights on the simplex.
        'K_min'     : float, final value of K(λ).
        'm_lambda'  : ndarray (d,), the m(λ) for the optimal λ.
        'A_lambda'  : ndarray (d, d), the A_lambda = inv(sum_i λ_i A_inv_i) if computable, else None.
        'success'   : bool, whether convergence was reached.
        'message'   : str, info on convergence or iteration limit.
        'history'   : list of float, K(λ) at each iteration.
    """
    k, d = xs.shape

    # Precompute A_inv_array and x_Ainv_x for all i
    A_inv_array, x_Ainv_x = get_A_operations_fast(A_array, xs)

    # Initialize λ uniformly on the simplex
    lmbd = np.ones(k) / k

    history = []
    prev_obj = np.inf
    converged = False

    for t in range(max_iter):
        # Compute gradient ∇K(λ) = [∂K/∂λ_i] using fast routine
        grad = compute_K_gradient_fast(lmbd, xs, A_inv_array)
        # Linear minimization oracle over simplex: pick vertex minimizing gradient entry
        j = np.argmin(grad)

        # Step size γ = 2 / (t + 2)
        gamma = 2.0 / (t + 2.0)

        # Update λ ← (1 - γ)λ + γ e_j
        lmbd = (1 - gamma) * lmbd
        lmbd[j] += gamma

        # Evaluate objective K(λ) using fast routine
        obj = compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x)
        history.append(obj)

        # Check for convergence: change in objective < tol
        if abs(prev_obj - obj) < tol:
            converged = True
            break
        prev_obj = obj

    # After FW, compute m(λ) and A_lambda if possible
    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))  # sum_i λ_i A_inv_i
    # Compute S = sum_i λ_i (A_inv_i @ x_i)
    Ax = np.matmul(A_inv_array, xs[..., None]).squeeze(-1)  # shape (k, d)
    S = (lmbd[:, None] * Ax).sum(axis=0)  # shape (d,)

    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
        A_lambda = np.linalg.inv(A_lambda_inv)
    except np.linalg.LinAlgError:
        m_lambda = None
        A_lambda = None

    message = "Converged" if converged else f"Max iterations ({max_iter}) reached"

    return {
        'lambda': lmbd,
        'K_min': history[-1] if history else compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x),
        'm_lambda': m_lambda,
        'A_lambda': A_lambda,
        'success': converged,
        'message': message,
        'history': history
    }


def ellipsoidal_intersection_fw(eps, xs, A_array, **fw_kwargs):
    """
    Returns True if the ellipsoidal balls (with centers xs and covariance array A_array)
    intersect (i.e., if the minimized K(λ) > 0) using the Frank–Wolfe solver.

    :param eps: float
        The radius/persistence parameter.
    :param xs: ndarray, shape (k, d)
        The centers x_i.
    :param A_array: ndarray, shape (k, d, d)
        Stack of covariance matrices A(x_i) for each center.
    :param fw_kwargs: additional keyword arguments passed to minimize_K_fw.
    :return: bool
        True if K_min > 0 (nonempty intersection), False otherwise.
    """
    result = minimize_K_fw(eps, xs, A_array, **fw_kwargs)
    return result['K_min'] > 0


# Example usage:
# eps = 1.0
# xs = np.random.randn(5, 3)          # 5 centers in 3D
# A_array = np.stack([np.eye(3) for _ in range(5)], axis=0)  # e.g., identity covariances
#
# result_fw = minimize_K_fw(eps, xs, A_array, max_iter=200, tol=1e-8)
# print("λ (FW):", result_fw['lambda'])
# print("K_min (FW):", result_fw['K_min'])
# print("Converged (FW):", result_fw['success'], result_fw['message'])
#
# has_intersection = ellipsoidal_intersection_fw(eps, xs, A_array, max_iter=200, tol=1e-8)
# print("Intersection exists (FW)?", has_intersection)
