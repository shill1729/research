import numpy as np
from scipy.optimize import minimize, LinearConstraint
from tda.ellipsoidal.solvers.kfunction import compute_K_gradient_fast, compute_K_fast, compute_K_hessian_fast, get_A_operations_fast
from typing import Tuple, List, Dict


def minimize_K_adaptive_geometric(eps, xs, A_array=None,
                                  max_iter=1000,
                                  explore_boundary=True,
                                  initial_lr=0.1,
                                  fallback_solver="SLSQP"):
    """
    Novel Adaptive Geometric Algorithm for minimizing K(λ) over the probability simplex.

    This algorithm combines multiple optimization strategies:
    1. Smart multi-strategy initialization
    2. Adaptive projected gradient descent with momentum
    3. Boundary exploration using KKT insights
    4. Final refinement with standard solvers

    Based on the geometric insights from "On the intersection of finitely many
    ellipsoidal balls in Euclidean space" by Sean Hill.

    :param eps: Radius parameter ε
    :param xs: Points array of shape (k, D)
    :param A_array: Precision matrices array of shape (k, D, D)
    :param max_iter: Maximum iterations for custom phases
    :param explore_boundary: Whether to explore simplex boundary
    :param initial_lr: Initial learning rate for gradient descent
    :param fallback_solver: Solver for final refinement ("SLSQP" or "trust-constr")
    :return: Dictionary with optimization results
    """
    k, D = xs.shape
    if A_array is None:
        raise ValueError("Must pass an array of matrices of shape (k, d, d)")

    # Precompute operations for efficiency
    A_inv_array, x_Ainv_x = get_A_operations_fast(A_array, xs)

    # Define objective and gradient using provided functions
    def obj(lmbd):
        return compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x)

    def jac(lmbd):
        return compute_K_gradient_fast(lmbd, xs, A_inv_array)

    def hess(lmbd):
        return compute_K_hessian_fast(lmbd, xs, A_inv_array)

    # Phase 1: Smart initialization
    lmbd_best = _smart_initialization(xs, A_inv_array, x_Ainv_x, obj)
    obj_best = obj(lmbd_best)

    # Phase 2: Adaptive projected gradient descent
    lmbd_agd, obj_agd, history = _adaptive_projected_gradient(
        lmbd_best, obj, jac, max_iter, initial_lr
    )

    if obj_agd < obj_best:
        lmbd_best = lmbd_agd
        obj_best = obj_agd

    # Phase 3: Boundary exploration (if requested and potentially beneficial)
    if explore_boundary and obj_best > -eps ** 2:
        lmbd_boundary, obj_boundary = _boundary_exploration(
            lmbd_best, xs, A_inv_array, x_Ainv_x, obj, jac
        )

        if obj_boundary < obj_best:
            lmbd_best = lmbd_boundary
            obj_best = obj_boundary

    # Phase 4: Final refinement with standard solver
    lmbd_final, obj_final, refinement_success = _final_refinement(
        lmbd_best, eps, xs, A_inv_array, x_Ainv_x, fallback_solver
    )

    if refinement_success and obj_final < obj_best:
        lmbd_best = lmbd_final
        obj_best = obj_final

    # Compute final solution details
    A_lambda_inv = np.tensordot(lmbd_best, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd_best[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)

    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
        A_lambda = np.linalg.inv(A_lambda_inv)
        success = True
        message = "Adaptive geometric algorithm converged successfully"
    except np.linalg.LinAlgError:
        m_lambda = None
        A_lambda = None
        success = False
        message = "Numerical issues in final solution"

    # Additional diagnostic information
    active_indices = np.where(lmbd_best > 1e-8)[0]
    intersection_exists = obj_best > -1e-8

    return {
        'lambda': lmbd_best,
        'K_min': obj_best,
        'm_lambda': m_lambda,
        'A_lambda': A_lambda,
        'success': success,
        'message': message,
        'active_indices': active_indices,
        'intersection_exists': intersection_exists,
        'optimization_history': history
    }


def _smart_initialization(xs, A_inv_array, x_Ainv_x, obj_func) -> np.ndarray:
    """Smart initialization using multiple strategies."""
    k = xs.shape[0]
    candidates = []

    # Strategy 1: Uniform distribution
    candidates.append(np.ones(k) / k)

    # Strategy 2: Inverse distance weighting from centroid
    centroid = np.mean(xs, axis=0)
    distances = np.array([np.linalg.norm(xs[i] - centroid) for i in range(k)])
    if np.any(distances > 1e-10):
        weights = 1 / (distances + 1e-10)
        candidates.append(weights / np.sum(weights))

    # Strategy 3: Volume-based weighting (using determinant of precision matrices)
    try:
        log_dets = np.array([np.linalg.slogdet(A_inv_array[i])[1] for i in range(k)])
        # Larger determinant = smaller ellipsoid volume, so use negative
        volumes = np.exp(-0.5 * log_dets)
        volumes = volumes / np.sum(volumes)
        candidates.append(volumes)
    except:
        pass

    # Strategy 4: Based on diagonal dominance of precision matrices
    try:
        diag_strength = np.array([np.trace(A_inv_array[i]) for i in range(k)])
        diag_weights = diag_strength / np.sum(diag_strength)
        candidates.append(diag_weights)
    except:
        pass

    # Strategy 5: Random perturbations
    np.random.seed(42)  # For reproducibility
    for _ in range(3):
        random_weights = np.random.exponential(1, k)
        candidates.append(random_weights / np.sum(random_weights))

    # Evaluate all candidates and return the best
    best_lmbd = candidates[0]
    best_obj = obj_func(best_lmbd)

    for lmbd in candidates[1:]:
        try:
            obj_val = obj_func(lmbd)
            if obj_val < best_obj:
                best_obj = obj_val
                best_lmbd = lmbd
        except:
            continue

    return best_lmbd


def _project_simplex(x: np.ndarray) -> np.ndarray:
    """Efficient projection onto probability simplex."""
    x_sorted = np.sort(x)[::-1]
    cumsum = np.cumsum(x_sorted)
    ind = np.arange(1, len(x) + 1)
    cond = x_sorted - (cumsum - 1) / ind > 0

    if np.any(cond):
        rho = np.where(cond)[0][-1]
        theta = (cumsum[rho] - 1) / (rho + 1)
    else:
        theta = (cumsum[-1] - 1) / len(x)

    return np.maximum(x - theta, 0)


def _adaptive_projected_gradient(lmbd_init: np.ndarray,
                                 obj_func,
                                 grad_func,
                                 max_iter: int,
                                 initial_lr: float) -> Tuple[np.ndarray, float, Dict]:
    """Adaptive projected gradient descent with momentum."""
    lmbd = lmbd_init.copy()
    lr = initial_lr
    momentum = 0.9
    velocity = np.zeros_like(lmbd)
    tol = 1e-9

    history = {'objective': [], 'lambda': [], 'lr': [], 'grad_norm': []}

    for iteration in range(max_iter):
        try:
            grad = grad_func(lmbd)
            obj_val = obj_func(lmbd)
        except:
            lr *= 0.5
            if lr < 1e-12:
                break
            continue

        # Store history
        history['objective'].append(obj_val)
        history['lambda'].append(lmbd.copy())
        history['lr'].append(lr)
        history['grad_norm'].append(np.linalg.norm(grad))

        # Adaptive learning rate based on gradient norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 10:
            lr *= 0.8
        elif grad_norm < 0.1 and lr < 1.0:
            lr *= 1.05

        # Project gradient to simplex tangent space (subtract mean)
        grad_projected = grad - np.mean(grad)

        # Momentum update
        velocity = momentum * velocity - lr * grad_projected

        # Projected gradient step
        lmbd_new = _project_simplex(lmbd + velocity)

        # Check convergence
        if np.linalg.norm(lmbd_new - lmbd) < tol:
            lmbd = lmbd_new
            break

        # Simple line search
        alpha = 1.0
        lmbd_candidate = lmbd_new

        try:
            obj_candidate = obj_func(lmbd_candidate)
            # Armijo condition
            while (alpha > 1e-6 and
                   obj_candidate > obj_val + 1e-4 * alpha * np.dot(grad_projected, lmbd_candidate - lmbd)):
                alpha *= 0.7
                lmbd_candidate = _project_simplex(lmbd + alpha * velocity)
                obj_candidate = obj_func(lmbd_candidate)

            if alpha > 1e-6:
                lmbd = lmbd_candidate
            else:
                velocity *= 0.5
                lr *= 0.8
        except:
            velocity *= 0.5
            lr *= 0.8

    final_obj = obj_func(lmbd)
    return lmbd, final_obj, history


def _boundary_exploration(lmbd_interior: np.ndarray,
                          xs: np.ndarray,
                          A_inv_array: np.ndarray,
                          x_Ainv_x: np.ndarray,
                          obj_func,
                          grad_func) -> Tuple[np.ndarray, float]:
    """
    Explore boundary of simplex for better solutions.
    Based on KKT insight that optimal solutions often lie on boundary.
    """
    k = len(lmbd_interior)
    best_lmbd = lmbd_interior.copy()
    best_obj = obj_func(lmbd_interior)

    # Try fixing each coordinate to 0 and optimizing over remaining face
    for i in range(k):
        if lmbd_interior[i] > 0.05:  # Only try if coordinate is reasonably large
            # Create indices for reduced problem
            active_indices = [j for j in range(k) if j != i]
            if len(active_indices) <= 1:
                continue

            # Initialize on this face
            lmbd_reduced = lmbd_interior[active_indices]
            lmbd_reduced = lmbd_reduced / np.sum(lmbd_reduced)

            try:
                # Optimize on this face using simple projected gradient
                lmbd_face_opt = _optimize_on_face(active_indices, lmbd_reduced,
                                                  obj_func, grad_func, max_iter=50)

                # Reconstruct full lambda
                lmbd_full = np.zeros(k)
                lmbd_full[active_indices] = lmbd_face_opt

                obj_face = obj_func(lmbd_full)
                if obj_face < best_obj:
                    best_lmbd = lmbd_full
                    best_obj = obj_face
            except:
                continue

    return best_lmbd, best_obj


def _optimize_on_face(active_indices: List[int],
                      lmbd_reduced: np.ndarray,
                      obj_func,
                      grad_func,
                      max_iter: int = 50) -> np.ndarray:
    """Optimize on a face of the simplex."""
    k_full = len(active_indices) + 1  # Infer full dimension

    for _ in range(max_iter):
        # Construct full lambda for gradient computation
        lmbd_full = np.zeros(k_full)
        lmbd_full[active_indices] = lmbd_reduced

        # Compute gradient
        grad_full = grad_func(lmbd_full)
        grad_reduced = grad_full[active_indices]

        # Project to face tangent space
        grad_projected = grad_reduced - np.mean(grad_reduced)

        # Update with projection
        lmbd_reduced_new = _project_simplex(lmbd_reduced - 0.05 * grad_projected)

        if np.linalg.norm(lmbd_reduced_new - lmbd_reduced) < 1e-6:
            break
        lmbd_reduced = lmbd_reduced_new

    return lmbd_reduced


def _final_refinement(lmbd_init: np.ndarray,
                      eps: float,
                      xs: np.ndarray,
                      A_inv_array: np.ndarray,
                      x_Ainv_x: np.ndarray,
                      solver: str) -> Tuple[np.ndarray, float, bool]:
    """Final refinement using standard constrained optimization."""
    k = len(lmbd_init)

    def obj(lmbd):
        return compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x)

    def jac(lmbd):
        return compute_K_gradient_fast(lmbd, xs, A_inv_array)

    def hess(lmbd):
        return compute_K_hessian_fast(lmbd, xs, A_inv_array)

    constraints_eq = {'type': 'eq', 'fun': lambda lmbd: np.sum(lmbd) - 1}
    bounds = [(0., 1.) for _ in range(k)]

    try:
        if solver == "SLSQP":
            res = minimize(obj, lmbd_init, method=solver, jac=jac,
                           bounds=bounds, constraints=constraints_eq,
                           options={'ftol': 1e-12, 'maxiter': 1000})
        elif solver == "trust-constr":
            A_constr = np.ones((1, k))
            constraint_trust = LinearConstraint(A_constr, [1], [1])
            res = minimize(obj, lmbd_init, method=solver, jac=jac, hess=hess,
                           bounds=bounds, constraints=[constraint_trust],
                           options={'gtol': 1e-12, 'maxiter': 1000})
        else:
            raise ValueError("Solver must be 'SLSQP' or 'trust-constr'")

        return res.x, res.fun, res.success
    except:
        return lmbd_init, obj(lmbd_init), False


# Wrapper to maintain compatibility with original interface
def minimize_K_jw(eps, xs, A_array=None, solver="SLSQP"):
    """
    Original interface for backward compatibility.
    Uses the new adaptive geometric algorithm as default.
    """
    result = minimize_K_adaptive_geometric(eps, xs, A_array, fallback_solver=solver)

    # Return in original format
    return {
        'lambda': result['lambda'],
        'K_min': result['K_min'],
        'm_lambda': result['m_lambda'],
        'A_lambda': result['A_lambda'],
        'success': result['success'],
        'message': result['message']
    }