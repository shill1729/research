import numpy as np
from scipy.optimize import minimize


def project_to_simplex(v):
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

    A_inv_array, x_Ainv_x = get_A_operations(A_list, xs)
    for _ in range(max_iters):
        grad = compute_K_gradient(lmbd, xs, A_inv_array)
        lmbd_new = project_to_simplex(lmbd - step_size * grad)
        if np.linalg.norm(lmbd_new - lmbd) < tol:
            break
        lmbd = lmbd_new

    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    # The centroid is given by a linear system B m = S
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        m_lambda = None

    return {
        'lambda': lmbd,
        'K_min': compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x),
        'm_lambda': m_lambda,
        'A_lambda': np.linalg.inv(A_lambda_inv) if m_lambda is not None else None
    }


def compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x):
    """
    Computes K(λ) = eps^2 - C(λ), where
       A_lambda_inv = sum_i λ_i A_inv_array[i]
       S = sum_i λ_i (A_inv_array[i] @ xs[i])
       m(λ) is obtained by solving A_lambda_inv @ m = S,
       and C(λ) = (sum_i λ_i x_Ainv_x[i]) - m(λ)^T S.

    :param lmbd: convex combination coefficients
    :param eps: the radius/persistence parameter
    :param xs: the centers, the pair, the tuple, triple
    :param A_inv_array: the tensor of inverse A inverse.
    :param x_Ainv_x: the quadratic form at the centers.
    :return:
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

    :param lmbd:
    :param xs:
    :param A_inv_array:
    :return:
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


def get_A_operations(A_list, xs):
    """

    :param A_list:
    :param xs:
    :return:
    """
    # Precompute each A_i^{-1} and x_i^T A_i^{-1} x_i.
    k, _ = xs.shape
    A_inv_list = []
    x_Ainv_x = np.zeros(k)
    for i in range(k):
        A_inv = np.linalg.inv(A_list[i])
        A_inv_list.append(A_inv)
        x_Ainv_x[i] = xs[i].T @ A_inv @ xs[i]
    # Stack the inverses into a single array for vectorized operations.
    A_inv_array = np.stack(A_inv_list, axis=0)
    return A_inv_array, x_Ainv_x


def minimize_K(eps, xs, A_list=None, solver="SLSQP"):
    """
    Minimizes K(λ) = eps^2 - C(λ) over the probability simplex using the simplified gradient.

    :param eps:
    :param xs:
    :param A_list:
    :param solver:
    :return:
    """
    k, D = xs.shape
    if A_list is None:
        A_list = [np.eye(D) for _ in range(k)]

    A_inv_array, x_Ainv_x = get_A_operations(A_list, xs)

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

    :param eps:
    :param xs:
    :param A_list:
    :return:
    """
    result = minimize_K(eps, xs, A_list)
    return result["K_min"] > 0


def cauchy_simplex_minimize_K(eps, xs, A_list, max_iters=200, tol=1e-6):
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
            result = {"lambda": lmbd, "K_min": compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x), "success": True}
            return result
    result = {"lambda": lmbd, "K_min": compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x), "success": False}
    return result
