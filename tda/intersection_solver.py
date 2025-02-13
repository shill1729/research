import numpy as np
from scipy.optimize import minimize


def compute_K(lmbd, eps, xs, A_inv_list, x_Ainv_x):
    """
    Given:
      - lmbd: a numpy array of shape (k,), the weights (must sum to 1, nonnegative)
      - eps: the scale parameter epsilon (a scalar)
      - xs: a numpy array of shape (k, d) holding the centers x_i.
      - A_inv_list: a list of k (d x d) arrays, each being the inverse of A_i.
      - x_Ainv_x: a list of k scalars, each equal to x_i^T A_i^{-1} x_i.

    This function computes:
       A_lambda^{-1} = sum_i lmbd_i * A_i^{-1}
       m_lambda = (A_lambda^{-1})^{-1} ( sum_i lmbd_i * (A_i^{-1} @ x_i) )
       C(lmbd) = sum_i lmbd_i * (x_i^T A_i^{-1} x_i) - m_lambda^T A_lambda^{-1} m_lambda
       K(lmbd) = eps^2 - C(lmbd)
    """
    k = len(lmbd)
    D = xs.shape[1]

    # Build A_lambda_inv and the weighted sum S = sum_i lambda_i A_i^{-1} x_i.
    A_lambda_inv = np.zeros((D, D))
    S = np.zeros(D)
    for i in range(k):
        A_lambda_inv += lmbd[i] * A_inv_list[i]
        S += lmbd[i] * (A_inv_list[i] @ xs[i])

    # Compute A_lambda = (A_lambda_inv)^{-1}
    try:
        A_lambda = np.linalg.inv(A_lambda_inv)
    except np.linalg.LinAlgError:
        # If the matrix is (nearly) singular, return a large penalty value.
        return 1e10

    # Compute m_lambda = A_lambda * S
    m_lambda = A_lambda @ S

    # Compute sum_i lambda_i * (x_i^T A_i^{-1} x_i)
    sum_term = np.dot(lmbd, x_Ainv_x)

    # Compute m_lambda^T A_lambda_inv m_lambda (note that A_lambda_inv is already computed)
    quad_term = m_lambda.T @ A_lambda_inv @ m_lambda

    # Now, C(lmbd) = sum_term - quad_term.
    C_lmbd = sum_term - quad_term

    # Then K(lmbd) = eps^2 - C(lmbd)
    return eps ** 2 - C_lmbd


def minimize_K(eps, xs, A_list=None):
    """
    Minimize the function K(lmbd) = eps^2 - C(lmbd) over the probability simplex
    where C(lmbd) is computed using the centers xs and the matrices A_i.

    Parameters:
      eps : float
         The fixed scale parameter epsilon.
      xs : numpy array of shape (k,D)
         The centers (each x_i is in R^D).
      A_list : list of k (D x D) arrays, optional
         The positive–definite matrices A_i. If None, each A_i is taken to be the identity.

    Returns:
      result: dict with keys:
          'lambda'  : the minimizing vector (of length k)
          'K_min'   : the minimum value of K
          'm_lambda': the corresponding m_lambda (computed from the optimal lambda)
          'A_lambda': the corresponding A_lambda (computed from the optimal lambda)
          'success' : boolean flag from the optimizer.
    """
    k, D = xs.shape

    # If no matrices are provided, use identity.
    if A_list is None:
        A_list = [np.eye(D) for _ in range(k)]

    # Precompute the inverses and the quadratic terms for each x.
    A_inv_list = []
    x_Ainv_x = []
    for i in range(k):
        # TODO: efficient inverse computations, like EVD and taking the inverse of the diag component since A is PD.
        # Compute the inverse of A_i
        A_inv = np.linalg.inv(A_list[i])
        A_inv_list.append(A_inv)
        x_Ainv_x.append(xs[i].T @ A_inv @ xs[i])

    # Define the objective function to be minimized.
    def obj(lmbd):
        return compute_K(lmbd, eps, xs, A_inv_list, x_Ainv_x)

    # Initial guess: uniform distribution on the simplex.
    lmbd0 = np.ones(k) / k

    # Define constraints: sum(lmbd) == 1 and lmbd_i >= 0.
    constraints = {'type': 'eq', 'fun': lambda lmbd: np.sum(lmbd) - 1}
    bounds = [(0, 1) for _ in range(k)]

    # Call the SLSQP optimizer.
    # solver = "SLSQP"
    res = minimize(obj, lmbd0, method='SLSQP', bounds=bounds, constraints=constraints, tol=10**-6)

    # Compute additional quantities at the optimum.
    opt_lmbd = res.x
    # Recompute A_lambda_inv and m_lambda.
    A_lambda_inv = np.zeros((D, D))
    S = np.zeros(D)
    for i in range(k):
        A_lambda_inv += opt_lmbd[i] * A_inv_list[i]
        S += opt_lmbd[i] * (A_inv_list[i] @ xs[i])
    A_lambda = np.linalg.inv(A_lambda_inv)
    m_lambda = A_lambda @ S

    return {'lambda': opt_lmbd, 'K_min': res.fun, 'm_lambda': m_lambda,
            'A_lambda': A_lambda, 'success': res.success, 'message': res.message}


def ellipsoidal_intersection(eps, xs, A_list):
    """
    Returns True if the ellipsoidal balls with centers in xs (and with matrices in A_list)
    intersect (i.e. if the minimized K(λ) > 0) and False otherwise.
    """
    result = minimize_K(eps, xs, A_list)
    if result["K_min"] > 0:
        return True
    else:
        return False



