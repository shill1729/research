import numpy as np
from scipy.optimize import minimize, LinearConstraint
from tda.ellipsoidal.solvers.kfunction import compute_K_gradient_fast, compute_K_fast, compute_K_hessian_fast, get_A_operations_fast


def minimize_K(eps, xs, A_array=None, solver="SLSQP"):
    """
    Minimizes K(λ) = eps^2 - C(λ) over the probability simplex using the simplified gradient.

    :param eps:
    :param xs:
    :param A_list:
    :param solver:
    :return:
    """
    k, D = xs.shape
    if A_array is None:
        raise ValueError("Must past an array of matrices of shape (k, d, d)")

    A_inv_array, x_Ainv_x = get_A_operations_fast(A_array, xs)

    # Define the objective and its gradient.
    def obj(lmbd, eps, xs, A_inv_array, x_Ainv_x):
        return compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x)

    def jac(lmbd, eps, xs, A_inv_array, x_Ainv_x):
        return compute_K_gradient_fast(lmbd, xs, A_inv_array)

    def hess(lmbd, eps, xs, A_inv_array, x_Ainv_x):
        return compute_K_hessian_fast(lmbd, xs, A_inv_array)

    lmbd0 = np.ones(k) / k  # initial uniform guess
    constraints = {'type': 'eq', 'fun': lambda lmbd: np.sum(lmbd) - 1}
    bounds = [(0., 1.) for _ in range(k)]

    # Trust-constr needs different constraint format/type
    A_constr = np.ones((1, k))  # Row vector of ones
    lb = ub = np.array([1])  # Equality constraint
    constraint_trust = LinearConstraint(A_constr, lb, ub)
    if solver == "SLSQP":
        res = minimize(obj, lmbd0, args=(eps, xs, A_inv_array, x_Ainv_x), method=solver, jac=jac, hess=None,
                       bounds=bounds,
                       constraints=constraints, tol=1e-9)
    elif solver == "trust-constr":
        res = minimize(obj, lmbd0, args=(eps, xs, A_inv_array, x_Ainv_x), method=solver, jac=jac, hess=hess,
                       bounds=bounds,
                       # Expected type complaint here because I have scipy 1.13.1,
                       # 1.15.2 documentation would accept this but 1.13 doesn't yet.
                       constraints=[constraint_trust], tol=1e-9)
    else:
        raise ValueError("Only ineq+eq constrained solvers are SLSQP and trust-constr")
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
    intersect (i.e. if the minimized K(λ) >= 0), and False otherwise.

    :param eps:
    :param xs:
    :param A_list:
    :return:
    """
    result = minimize_K(eps, xs, A_list)
    return result["K_min"] >= 0
