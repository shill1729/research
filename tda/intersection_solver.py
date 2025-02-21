import numpy as np
from scipy.optimize import minimize


def compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x):
    """
    Computes K(λ) = eps^2 - C(λ), where
       A_lambda_inv = sum_i λ_i A_inv_array[i]
       S = sum_i λ_i (A_inv_array[i] @ xs[i])
       m(λ) is obtained by solving A_lambda_inv @ m = S,
       and C(λ) = (sum_i λ_i x_Ainv_x[i]) - m(λ)^T S.
    """
    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        print("Non PD!")
        return 1e10  # Penalty for singularity
    quad_term = m_lambda.dot(S)  # equals m_lambda^T A_lambda_inv m_lambda
    sum_term = np.dot(lmbd, x_Ainv_x)
    return eps ** 2 - (sum_term - quad_term)


def compute_K_gradient(lmbd, xs, A_inv_array):
    """
    Computes the gradient ∇K(λ) using the simplified formula:
      ∂K/∂λ_j = -(x_j - m(λ))^T A_j^{-1} (x_j - m(λ))
    where m(λ) is computed from the weighted combination of A_inv_array.
    """
    # Compute aggregated inverse and weighted sum
    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        return np.zeros_like(lmbd)

    # Compute (x_j - m(λ)) for each j
    diff = xs - m_lambda  # broadcasting m_lambda to each row of xs
    # For each j, compute (x_j - m(λ))^T A_inv_array[j] (x_j - m(λ)):
    grad = -np.einsum('ij,ijk,ik->i', diff, A_inv_array, diff)
    return grad


def minimize_K(eps, xs, A_list=None, solver="SLSQP"):
    """
    Minimizes K(λ) = eps^2 - C(λ) over the probability simplex using the simplified gradient.
    """
    k, D = xs.shape
    if A_list is None:
        A_list = [np.eye(D) for _ in range(k)]

    # Precompute each A_i^{-1} and x_i^T A_i^{-1} x_i.
    A_inv_list = []
    x_Ainv_x = np.zeros(k)
    for i in range(k):
        A_inv = np.linalg.inv(A_list[i])
        A_inv_list.append(A_inv)
        x_Ainv_x[i] = xs[i].T @ A_inv @ xs[i]
    # Stack the inverses into a single array for vectorized operations.
    A_inv_array = np.stack(A_inv_list, axis=0)

    # Define the objective and its gradient.
    def obj(lmbd, eps, xs, A_inv_array, x_Ainv_x):
        return compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x)

    def jac(lmbd, eps, xs, A_inv_array, x_Ainv_x):
        return compute_K_gradient(lmbd, xs, A_inv_array)

    lmbd0 = np.ones(k) / k  # initial uniform guess
    constraints = {'type': 'eq', 'fun': lambda lmbd: np.sum(lmbd) - 1}
    bounds = [(0, 1) for _ in range(k)]

    res = minimize(obj, lmbd0, args=(eps, xs, A_inv_array, x_Ainv_x), method=solver, jac=jac, bounds=bounds,
                   constraints=constraints, tol=1e-6)

    opt_lmbd = res.x
    # Recompute m(λ) at the optimum.
    A_lambda_inv = np.tensordot(opt_lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(opt_lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        m_lambda = None
    A_lambda = np.linalg.inv(A_lambda_inv) if m_lambda is not None else None

    return {'lambda': opt_lmbd, 'K_min': res.fun, 'm_lambda': m_lambda,
            'A_lambda': A_lambda, 'success': res.success, 'message': res.message}


def ellipsoidal_intersection(eps, xs, A_list):
    """
    Returns True if the ellipsoidal balls (with centers xs and matrices A_list)
    intersect (i.e. if the minimized K(λ) > 0), and False otherwise.
    """
    result = minimize_K(eps, xs, A_list)
    return result["K_min"] > 0


def cauchy_simplex_minimize_K(eps, xs, A_list, max_iters=200, tol=1e-6):
    """
    Implements the Cauchy-Simplex algorithm for minimizing K(λ) over the probability simplex,
    using the simplified gradient.
    """
    k, D = xs.shape
    A_inv_list = [np.linalg.inv(A_list[i]) for i in range(k)]
    A_inv_array = np.stack(A_inv_list, axis=0)
    x_Ainv_x = np.array([xs[i].T @ A_inv_array[i] @ xs[i] for i in range(k)])
    lmbd = np.ones(k) / k

    for t in range(max_iters):
        grad = compute_K_gradient(lmbd, xs, A_inv_array)
        grad_mean = np.dot(lmbd, grad)
        direction = lmbd * (grad - grad_mean)
        step_size = min(1.0, 1.0 / np.max(np.abs(grad - grad_mean)))
        lmbd = lmbd - step_size * direction
        lmbd = np.maximum(lmbd, 1e-10)
        lmbd /= np.sum(lmbd)
        if np.linalg.norm(direction) < tol:
            return {"lambda": lmbd, "K_min": compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x), "success": True}

    return {"lambda": lmbd, "K_min": compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x), "success": False}