# TODO: Does not work
import numpy as np
import itertools
from scipy.linalg import solve


def get_A_operations(A_list, xs):
    """
    Precompute A_i^{-1} and x_i^T A_i^{-1} x_i for all i.
    """
    k, D = xs.shape
    A_inv_list = [np.linalg.inv(A_list[i]) for i in range(k)]
    A_inv_array = np.stack(A_inv_list, axis=0)
    x_Ainv_x = np.array([xs[i].T @ A_inv_array[i] @ xs[i] for i in range(k)])
    return A_inv_array, x_Ainv_x


def solve_lambda_and_m(S, A_inv_array, xs):
    """
    Solve for lambda given a subset S of active indices.
    Enforce equal Mahalanobis distances for all active indices.
    """
    S = list(S)
    k_active = len(S)
    A_inv_S = A_inv_array[S]
    xs_S = xs[S]

    # Compute B(lambda)^{-1} and weighted sum S
    B_inv = np.sum(A_inv_S, axis=0)
    S_vec = np.sum(A_inv_S @ xs_S[:, :, None], axis=0).squeeze()

    try:
        m_lambda = solve(B_inv, S_vec)
    except np.linalg.LinAlgError:
        return None, None, None  # Singular case

    # Compute distances from m_lambda to x_i in S
    dists = np.array([((m_lambda - xs[i]).T @ A_inv_array[i] @ (m_lambda - xs[i])) for i in S])

    if not np.allclose(dists, dists[0]):
        return None, None, None  # Inconsistent distances

    alpha = dists[0]  # Common squared Mahalanobis distance

    # Solve for lambda
    C_matrix = np.array([[np.trace(A_inv_array[i] @ A_inv_array[j]) for j in S] for i in S])
    b_vector = np.ones(k_active)
    try:
        lambda_S = solve(C_matrix, b_vector)
        lambda_S /= np.sum(lambda_S)  # Normalize
    except np.linalg.LinAlgError:
        return None, None, None  # Singular case

    # Assign lambda_i = 0 for inactive indices
    lambda_full = np.zeros(A_inv_array.shape[0])
    lambda_full[S] = lambda_S

    return lambda_full, m_lambda, alpha


def face_enumeration_solver(eps, xs, A_list):
    """
    Face enumeration solver for min_lambda K(lambda) on the probability simplex.
    """
    k, D = xs.shape
    A_inv_array, x_Ainv_x = get_A_operations(A_list, xs)
    best_lambda, best_K = None, np.inf

    # Try all subsets S of {1,2,...,k}
    for S in itertools.chain.from_iterable(itertools.combinations(range(k), r) for r in range(1, k + 1)):
        lambda_S, m_lambda, alpha = solve_lambda_and_m(S, A_inv_array, xs)
        if lambda_S is None:
            continue

        # Check feasibility: all inactive distances should be <= alpha
        feasible = all(
            ((m_lambda - xs[j]).T @ A_inv_array[j] @ (m_lambda - xs[j])) <= alpha
            for j in range(k) if j not in S
        )

        if feasible:
            K_lambda = eps ** 2 - (
                    np.dot(lambda_S, x_Ainv_x) - m_lambda.T @ np.sum(lambda_S[:, None, None] * A_inv_array,
                                                                     axis=0) @ m_lambda)
            if K_lambda < best_K:
                best_lambda, best_K = lambda_S, K_lambda

    return {"lambda": best_lambda, "K_min": best_K}


def active_set_pivoting_solver(eps, xs, A_list, max_iters=100):
    """
    Active-set pivoting solver for min_lambda K(lambda).
    """
    k, D = xs.shape
    A_inv_array, x_Ainv_x = get_A_operations(A_list, xs)
    S = {np.argmin(x_Ainv_x)}  # Start with the minimum-distance point
    lambda_opt, best_K = None, np.inf

    for _ in range(max_iters):
        lambda_S, m_lambda, alpha = solve_lambda_and_m(S, A_inv_array, xs)
        if lambda_S is None:
            break

        # Compute distances for inactive indices
        inactive_dists = {
            j: ((m_lambda - xs[j]).T @ A_inv_array[j] @ (m_lambda - xs[j])) for j in range(k) if j not in S
        }

        # Check if all inactive indices are feasible
        if all(d <= alpha for d in inactive_dists.values()):
            K_lambda = eps ** 2 - (
                    np.dot(lambda_S, x_Ainv_x) - m_lambda.T @ np.sum(lambda_S[:, None, None] * A_inv_array,
                                                                     axis=0) @ m_lambda)
            if K_lambda < best_K:
                lambda_opt, best_K = lambda_S, K_lambda
            break

        # Pivot in the worst offending index
        worst_idx = max(inactive_dists, key=inactive_dists.get)
        S.add(worst_idx)

    return {"lambda": lambda_opt, "K_min": best_K}
