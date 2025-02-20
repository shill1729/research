import numpy as np
import time
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


def minimize_K(eps, xs, A_list=None, solver="SLSQP"):
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
        start_time = time.perf_counter()
        K = compute_K(lmbd, eps, xs, A_inv_list, x_Ainv_x)
        end_time = time.perf_counter()
        # print("Objective function time = "+str(end_time-start_time))
        return K

    def jac(lmbd):
        return compute_K_gradient(lmbd, xs, A_inv_list, x_Ainv_x)

    # Initial guess: uniform distribution on the simplex.
    lmbd0 = np.ones(k) / k

    # Define constraints: sum(lmbd) == 1 and lmbd_i >= 0.
    constraints = {'type': 'eq', 'fun': lambda lmbd: np.sum(lmbd) - 1}
    bounds = [(0, 1) for _ in range(k)]

    # Call the SLSQP optimizer.
    start_time = time.perf_counter()
    res = minimize(obj, lmbd0, method=solver, jac=jac, bounds=bounds, constraints=constraints, tol=10 ** -6)
    end_time = time.perf_counter()
    # print("Minimizer time (does not include inverting A) = "+str(end_time-start_time))
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


def compute_K_gradient(lmbd, xs, A_inv_list, x_Ainv_x):
    """
    Compute the exact gradient ∇K(λ) using the formula:
    ∂K/∂λ_j = -[x_j^T A(x_j)^{-1} x_j - 2 x_j^T A(x_j)^{-1} m_λ + m_λ^T A(x_j)^{-1} m_λ].

    Parameters:
      lmbd      : array of lambda values (length k)
      xs        : array of x_i vectors (shape (k, D))
      A_inv_list: list of matrices A(x_i)^{-1} for i=1,...,k (each of shape (D,D))
      x_Ainv_x  : precomputed array where x_Ainv_x[j] = xs[j].T @ A_inv_list[j] @ xs[j]

    Returns:
      The gradient ∇K(λ) as a numpy array of length k.
    """
    import numpy as np

    k = len(lmbd)
    D = xs.shape[1]

    # Build A_lambda_inv and the weighted sum S
    A_lambda_inv = np.zeros((D, D))
    S = np.zeros(D)
    for i in range(k):
        A_lambda_inv += lmbd[i] * A_inv_list[i]
        S += lmbd[i] * (A_inv_list[i] @ xs[i])

    # Compute A_lambda and m_lambda
    A_lambda = np.linalg.inv(A_lambda_inv)
    m_lambda = A_lambda @ S

    # Compute gradient of C(λ) and then return -gradient for K(λ)
    grad_C = np.zeros(k)
    for j in range(k):
        term1 = x_Ainv_x[j]  # x_j^T A(x_j)^{-1} x_j
        term2 = 2 * (xs[j].T @ A_inv_list[j] @ m_lambda)  # 2 x_j^T A(x_j)^{-1} m_lambda
        term3 = m_lambda.T @ A_inv_list[j] @ m_lambda  # m_lambda^T A(x_j)^{-1} m_lambda
        grad_C[j] = term1 - term2 + term3
    return -grad_C  # Because ∇K(λ) = -∇C(λ)


# Implementing the Cauchy-Simplex (CS) algorithm for minimizing K(lambda) over the probability simplex
def cauchy_simplex_minimize_K(eps, xs, A_list, max_iters=1000, tol=1e-6):
    k, D = xs.shape

    # Compute inverses and quadratic terms
    A_inv_list = [np.linalg.inv(A_list[i]) for i in range(k)]
    x_Ainv_x = [xs[i].T @ A_inv_list[i] @ xs[i] for i in range(k)]

    # Define the objective function
    def obj(lmbd):
        return compute_K(lmbd, eps, xs, A_inv_list, x_Ainv_x)

    # Initialize with uniform probability distribution
    lmbd = np.ones(k) / k

    for t in range(max_iters):
        grad = compute_K_gradient(lmbd, xs, A_inv_list, x_Ainv_x)
        # Compute Cauchy-Simplex update
        grad_mean = np.dot(lmbd, grad)
        direction = lmbd * (grad - grad_mean)
        step_size = min(1.0, 1.0 / np.max(np.abs(grad - grad_mean)))  # Ensure positivity
        lmbd = lmbd - step_size * direction

        # Normalize to remain in the simplex
        lmbd = np.maximum(lmbd, 1e-10)  # Avoid numerical issues
        lmbd /= np.sum(lmbd)

        # Check for convergence
        if np.linalg.norm(direction) < tol:
            return {"lambda": lmbd, "K_min": obj(lmbd), "success": True}

    return {"lambda": lmbd, "K_min": obj(lmbd), "success": False}

