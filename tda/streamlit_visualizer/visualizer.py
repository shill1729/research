"""
Ellipsoidal Ball Intersection via Convex Optimization of K(λ)

This module implements the formulas for K and optimization-based method for testing
intersection of ellipsoidal balls in ℝ^D, as developed in the accompanying paper write-up.

Given a finite set of points x₁, ..., x_k ∈ ℝ^D and associated positive-definite matrices
A₁, ..., A_k defining ellipsoidal balls E_ε(x_i) = { y ∈ ℝ^D : ‖y - x_i‖_{A_i^{-1}} ≤ ε },
we compute whether the intersection ⋂ E_ε(x_i) is non-empty by minimizing an associated
convex function K(λ) over the standard k-simplex Δ^k. The function K(λ) is derived from a
weighted average of squared Mahalanobis distances and admits closed-form expressions for
its value, gradient, and Hessian, although the Hessian is not used below explicitly.

This code includes:
- Computation of K(λ), its gradient, and curvature-based bounds
- Solvers (projected gradient descent and constrained minimization) for minimizing K
- Plotting utilities to visualize ellipsoids, centroids, and the K-surface
- A Streamlit interface for dynamic exploration over parameterized input curves

To run this streamlit app, you need to run in terminal:

'streamlit run ./visualizer.py'

"""
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import sympy as sp

from scipy.optimize import minimize, minimize_scalar
from matplotlib.patches import Ellipse

#=============================================================================================
# 1. Functions for computing $K( \lambda ) $, and its Gradient
#=============================================================================================
def precompute_matrix_operations(centers, pd_matrix_list):
    """
    Compute the inverses A_i^{-1} of each precision matrix A_i in pd_matrix_list, stack these into an array,
    and compute the quadratic forms x_i^T A_i^{-1} x_i for each center x_i.

    :param centers: ndarray of shape (k, D), representing the centers x₁, ..., x_k ∈ ℝᴰ
    :param pd_matrix_list: list of k positive-definite (D, D) ndarrays, A₁, ..., A_k

    :return: (A_inv_list, A_inv_array, x_Ainv_x)
             - A_inv_list: list of A_i^{-1}
             - A_inv_array: ndarray of shape (k, D, D) stacking all A_i^{-1}
             - x_Ainv_x: ndarray of shape (k,), containing x_i^T A_i^{-1} x_i for each i
    """
    k, D = centers.shape
    A_inv_list = [np.linalg.inv(pd_matrix_list[i]) for i in range(k)]
    A_inv_array = np.stack(A_inv_list, axis=0)
    # Computing the quadratic form: x^ T A^{-1} x
    x_Ainv_x = np.array([centers[i].T @ A_inv_list[i] @ centers[i] for i in range(k)])
    return A_inv_list, A_inv_array, x_Ainv_x

def compute_B_inv_and_m_lambda(lmbd, A_inv_array, x):
    """
    Given λ ∈ Δ^k, compute the barycentric precision matrix B(λ)^{-1} = Σ_i λ_i A_i^{-1},
    the weighted sum S = Σ_i λ_i A_i^{-1} x_i, and the centroid m(λ) = B^{-1}(λ) S.

    :param lmbd: ndarray of shape (k,), a point in the probability simplex
    :param A_inv_array: ndarray of shape (k, D, D), each A_i^{-1}
    :param x: ndarray of shape (k, D), the center points x₁,...,x_k ∈ ℝᴰ

    :return: (B_lambda_inv, m_lambda, S)
             - B_lambda_inv: the matrix B(λ)^{-1}
             - m_lambda: the centroid m(λ)
             - S: the weighted sum of A_i^{-1} x_i
    """
    B_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, x), axis=0)
    m_lambda = np.linalg.solve(B_lambda_inv, S)
    return B_lambda_inv, m_lambda, S

def compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x):
    """
    Compute the function K(λ) = ε² − C(λ), where
      C(λ) = Σ λ_i x_i^T A_i^{-1} x_i − m(λ)^T B(λ)^{-1} m(λ)
    as described in Lemma 4 of the paper.

    :param lmbd: ndarray of shape (k,), λ ∈ Δ^k
    :param eps: float, the ε parameter in the ellipsoidal intersection test
    :param xs: ndarray of shape (k, D), the centers x_i
    :param A_inv_array: ndarray of shape (k, D, D), the precision matrices A_i^{-1}
    :param x_Ainv_x: ndarray of shape (k,), containing x_i^T A_i^{-1} x_i

    :return: scalar value K(λ)
    """
    B_lambda_inv, m_lambda, S = compute_B_inv_and_m_lambda(lmbd, A_inv_array, xs)
    quad_term = m_lambda.dot(S)
    sum_term = np.dot(lmbd, x_Ainv_x)
    return eps ** 2 - (sum_term - quad_term)

def compute_K_gradient(lmbd, xs, A_inv_array):
    """
    Compute the gradient ∇K(λ), where
      ∂K/∂λ_j = −‖m(λ) − x_j‖²_{A_j^{-1}}.

    :param lmbd: ndarray of shape (k,), λ ∈ Δ^k
    :param xs: ndarray of shape (k, D), center points x_i
    :param A_inv_array: ndarray of shape (k, D, D), the precision matrices A_i^{-1}

    :return: ndarray of shape (k,), the gradient of K(λ)
    """
    B_lambda_inv, m_lambda, S = compute_B_inv_and_m_lambda(lmbd, A_inv_array, xs)
    diff = m_lambda - xs  # (x_j - m(λ)) for each j
    grad = -np.einsum('ij,ijk,ik->i', diff, A_inv_array, diff)
    return grad

#================================================================================================
# 2. Optimization functions.
#================================================================================================
def project_to_prob_simplex(v):
    """
    Project a real vector v ∈ ℝᵏ onto the k-dimensional probability simplex Δ^k.

    :param v: ndarray of shape (k,), input vector
    :return: ndarray of shape (k,), projection of v onto Δ^k
    """
    v_sorted = np.sort(v)[::-1]
    v_cumsum = np.cumsum(v_sorted)
    rho = np.where(v_sorted - (v_cumsum - 1) / (np.arange(len(v)) + 1) > 0)[0][-1]
    theta = (v_cumsum[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def projected_gradient_descent(eps, xs, A_inv_array, x_Ainv_x, step_size=0.001, max_iters=5000, tol=1e-10):
    """
    Perform projected gradient descent to minimize K(λ) over the simplex Δ^k.

    :param eps: float, the ε parameter
    :param xs: ndarray of shape (k, D), centers x_i
    :param A_inv_array: ndarray of shape (k, D, D), precision matrices A_i^{-1}
    :param x_Ainv_x: ndarray of shape (k,), x_i^T A_i^{-1} x_i
    :param step_size: float, learning rate
    :param max_iters: int, maximum number of iterations
    :param tol: float, stopping tolerance on λ-update norm

    :return: dictionary containing
             - 'lambda': minimizing λ
             - 'K_min': minimum K(λ)
             - 'm_lambda': centroid m(λ)
             - 'B_lambda_inv': matrix B(λ)^{-1}
    """
    k, D = xs.shape
    lmbd = np.ones(k) / k

    for _ in range(max_iters):
        grad = compute_K_gradient(lmbd, xs, A_inv_array)
        lmbd_new = project_to_prob_simplex(lmbd - step_size * grad)
        if np.linalg.norm(lmbd_new - lmbd) < tol:
            break
        lmbd = lmbd_new
    # Compute the optimal precicision and centroid
    B_lambda_inv, m_lambda, S = compute_B_inv_and_m_lambda(lmbd, A_inv_array, xs)
    return {
        'lambda': lmbd,
        'K_min': compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x),
        'm_lambda': m_lambda,
        'B_lambda_inv': B_lambda_inv if m_lambda is not None else None
    }

# This minimzer uses scipy's routines.
def minimize_K(eps, xs, A_inv_array, x_Ainv_x, solver="SLSQP"):
    """
    Minimize K(λ) over λ ∈ Δ^k using either projected gradient descent ("pga") or
    a constrained solver from scipy (e.g., "SLSQP").

    :param eps: float, ε parameter
    :param xs: ndarray of shape (k, D), centers x_i
    :param A_inv_array: ndarray of shape (k, D, D), A_i^{-1}
    :param x_Ainv_x: ndarray of shape (k,), x_i^T A_i^{-1} x_i
    :param solver: str, either "pga" or a scipy.optimize.minimize solver name

    :return: dictionary containing keys 'lambda', 'K_min', 'm_lambda', 'B_lambda_inv', and
    optimization metadata
    """
    k, D = xs.shape

    # If the user has passed one of the custom solvers, just call them and return.
    if solver == "pga":
        return projected_gradient_descent(eps, xs, A_inv_array, x_Ainv_x)
    # Otherwise, we move on to use the available scipy solvers.

    # The gradient needs the same function signature as the objective function, so
    # we make a wrapper for convenience
    def jac(lmbd, eps, xs, A_inv_array, x_Ainv_x):
        return compute_K_gradient(lmbd, xs, A_inv_array)

    lmbd0 = np.ones(k) / k
    constraints = {'type': 'eq', 'fun': lambda lmbd: np.sum(lmbd) - 1}
    bounds = [(0, 1) for _ in range(k)]

    res = minimize(compute_K, lmbd0, args=(eps, xs, A_inv_array, x_Ainv_x), method=solver,
                   jac=jac, bounds=bounds, constraints=constraints, tol=1e-10,
                   options={"maxiter": 5000})

    opt_lmbd = res.x
    B_lambda_inv, m_lambda, S = compute_B_inv_and_m_lambda(opt_lmbd, A_inv_array, xs)
    return {'lambda': opt_lmbd, 'K_min': res.fun, 'm_lambda': m_lambda,
            'B_lambda_inv': B_lambda_inv, 'success': res.success, 'message': res.message}

def minimize_K_boundary(eps, xs, A_inv_array, x_Ainv_x):
    """
    Minimize K(λ) over the 1D edges of the simplex Δ^k by searching all line segments
    between pairs of basis vectors. Uses bounded scalar optimization.

    :param eps: float, ε parameter
    :param xs: ndarray of shape (k, D), centers x_i
    :param A_inv_array: ndarray of shape (k, D, D), A_i^{-1}
    :param x_Ainv_x: ndarray of shape (k,), x_i^T A_i^{-1} x_i

    :return: dictionary with best boundary λ, minimum K, m(λ), B(λ)^{-1}, edge info, and solver status
    """
    k, D = xs.shape
    best = {
        'K_min': np.inf,
        'lambda': None,
        'm_lambda': None,
        'B_lambda_inv': None,
        'edge': None,
        'success': False,
        'message': ''
    }

    # helper to build λ from (i,j,t)
    def make_lambda(i, j, t):
        lam = np.zeros(k)
        lam[i] = t
        lam[j] = 1 - t
        return lam

    # objective along the edge (i,j)
    def K_edge(t, i, j):
        lam = make_lambda(i, j, t)
        return compute_K(lam, eps, xs, A_inv_array, x_Ainv_x)

    for i in range(k):
        for j in range(i+1, k):
            # minimize on t in [0,1]
            res = minimize_scalar(
                K_edge,
                args=(i,j),
                bounds=(0.0, 1.0),
                method='bounded',
                options={'xatol':1e-8}
            )
            if not res.success:
                print("Optimizer failed to converge")
                pass

            if res.fun < best['K_min']:
                # reconstruct the best λ, m(λ) and A(λ)
                t_opt  = res.x
                lam_opt = make_lambda(i, j, t_opt)
                # B(λ)^{-1} = sum λ_i A_i^{-1}
                B_lambda_inv, m_opt, S = compute_B_inv_and_m_lambda(lam_opt, A_inv_array, xs)
                best.update({
                    'K_min':    res.fun,
                    'lambda':   lam_opt,
                    'm_lambda': m_opt,
                    'B_lambda_inv': B_lambda_inv,
                    'edge':     (i,j),
                    'success':  res.success,
                    'message':  res.message
                })

    return best

#======================================================================================================
# 3. Functions for estimating curvature bounds for our short-cut check:
#======================================================================================================
def compute_precision_bounds(A_inv_list):
    """
    Compute α = min_i λ_max(A_i^{-1}) and β = max_i λ_max(A_i^{-1}) over all precision matrices.
    Used in curvature bounds for sufficient conditions of intersection.

    :param A_inv_list: list of (D,D) precision matrices A_i^{-1}

    :return: (alpha, beta), lower and upper spectral bounds
    """
    max_eigs = []
    for A_inv in A_inv_list:
        eigs = np.linalg.eigvals(A_inv)
        max_eigs.append(np.max(np.real(eigs)))  # ensure real parts
    alpha = np.min(max_eigs)
    beta = np.max(max_eigs)
    return alpha, beta

def compute_mahalanobis_distances(m_lambda, points, A_inv_list):
    """
    Compute the Mahalanobis distance from m(λ) to each x_i using precision matrix A_i^{-1}.

    :param m_lambda: ndarray of shape (D,), the centroid m(λ)
    :param points: ndarray of shape (k, D), the centers x_i
    :param A_inv_list: list of (D,D) precision matrices A_i^{-1}

    :return: ndarray of shape (k,), the Mahalanobis distances d_j = sqrt((m − x_j)^T A_j^{-1} (m − x_j))
    """
    djs = []
    for i in range(len(points)):
        diff = m_lambda - points[i]
        d_j = diff.T @ A_inv_list[i] @ diff
        djs.append(np.sqrt(d_j))
    return np.array(djs)

#====================================================================================================
# 4. "Point cloud" generation functions:
# These functions parse strings into sympy expressions. The user defines a parameterization of a curve
# and then we compute some arbitrarily chosen covariance matrices that vary along the curve which
# define ellipses in the plane with semi-axis tangent to the curve, all of which are converted to
# numpy functions or arrays.
#====================================================================================================
def parse_functions(func_str):
    """
    Parse a string of the form 'x(t), y(t)' into sympy expressions fx, fy and symbol t. Here, 'x(t)' and
    'y(t)' can be any smooth functions of t. To be sure what is passed is literally a string, e.g.

    func_str = 'sin(t), t*cos(t)'

    is a valid input to pass.

    :param func_str: str, input like "cos(t), sin(t)"
    :return: (fx, fy, t), where fx and fy are sympy expressions, and t is the sympy symbol
    """
    t = sp.symbols('t')
    try:
        funcs = func_str.split(',')
        if len(funcs) != 2:
            raise ValueError("Please provide two functions separated by a comma.")
        fx = sp.sympify(funcs[0].strip())
        fy = sp.sympify(funcs[1].strip())
        return fx, fy, t
    except Exception as e:
        st.error(f"Error parsing functions: {e}")
        return None, None, t


# An arbitrary chosen Precision field that determines the orientation and scale of the ellipse at
# each point of the point cloud $\{x_1, x_2, x_3\}$
def compute_precision_matrix(fx, fy, t_sym, t_val):
    """
    Given a parametric curve (fx, fy) in sympy form, and a scalar t_val, compute:
    - the point p(t_val)
    - the tangent vector at p(t_val)
    - the normal vector
    - construct an ellipse-aligned positive-definite matrix A using tangent and normal directions

    :param fx: sympy expression for x(t)
    :param fy: sympy expression for y(t)
    :param t_sym: sympy symbol t
    :param t_val: float, the value of t

    :return: (p_val, A), point and (2,2) positive-definite matrix
    """
    p = sp.Matrix([fx, fy])
    p_val = np.array([float(p.subs(t_sym, t_val)[0]), float(p.subs(t_sym, t_val)[1])])
    dp_dt = sp.Matrix([sp.diff(fx, t_sym), sp.diff(fy, t_sym)])
    tangent_sym = dp_dt.subs(t_sym, t_val)
    tangent = np.array([float(tangent_sym[0]), float(tangent_sym[1])])
    if np.linalg.norm(tangent) == 0:
        tangent = np.array([1.0, 0.0])
    else:
        tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    sigma_t, sigma_n = 0.9 * p_val[0] ** 2, 0.1 * np.sin(p_val[1] * p_val[0]) ** 2 + 0.01
    Q = np.column_stack((tangent, normal))
    D = np.diag([sigma_t, sigma_n])
    A = Q @ D @ Q.T
    return p_val, A

#===================================================================================================
# 5. Plotting-helper functions
#===================================================================================================
def ellipse_patch(center, Ainv, radius=1.0, edgecolor='black', linestyle="-"):
    """
    Create a matplotlib Ellipse patch from a 2D inverse precision matrix Ainv.

    The resulting ellipse has center `center`, semi-axes determined by the inverse square roots
    of the eigenvalues of Ainv scaled by `radius`, and orientation determined by its eigenvectors.

    :param center: ndarray of shape (2,), the center of the ellipse
    :param Ainv: ndarray of shape (2, 2), the inverse precision matrix A^{-1}
    :param radius: float, scaling factor for the ellipse radii (default: 1.0)
    :param edgecolor: str, color of the ellipse edge (default: 'black')
    :param linestyle: str, line style for the ellipse (default: '-')

    :return: matplotlib.patches.Ellipse object
    """
    vals, vecs = np.linalg.eigh(Ainv)
    sorted_indices = np.argsort(vals)  # ascending order
    vals = vals[sorted_indices]
    vecs = vecs[:, sorted_indices]
    semi_axis_major = radius / np.sqrt(vals[0])
    semi_axis_minor = radius / np.sqrt(vals[1])
    width = 2 * semi_axis_major
    height = 2 * semi_axis_minor
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return Ellipse(xy=center, width=width, height=height, angle=angle,
                   edgecolor=edgecolor, facecolor='none', linestyle=linestyle, fill=False)


def plot_curve_and_ellipses(fx, fy, points, A_inv_matrices, epsilon, m_lambda, result):
    """
    Plot the parameterized curve p(t), its associated ellipsoidal balls at three points,
    and the ellipse centered at m(λ*) with radius √K(λ*), when K(λ*) > 0.

    Each ellipse is drawn using its local inverse precision matrix and scaled by ε and δ_max.
    Also marks the optimal centroid m(λ*) and computes bounding box with 20% margin.

    :param fx: sympy expression for x(t)
    :param fy: sympy expression for y(t)
    :param points: ndarray of shape (3, 2), center points on the curve
    :param A_inv_matrices: list of (2,2) inverse precision matrices for each point
    :param epsilon: float, ε radius for the ellipsoidal balls
    :param m_lambda: ndarray of shape (2,), the centroid m(λ*)
    :param result: dict containing keys 'K_min' and 'B_lambda_inv' for final ellipse if K_min > 0
    :return: matplotlib.figure.Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    t_vals = np.linspace(0, 2 * np.pi, 300)

    # Evaluate the curve
    curve_x = sp.lambdify(t_sym, fx, 'numpy')(t_vals)
    curve_y = sp.lambdify(t_sym, fy, 'numpy')(t_vals)
    ax.plot(curve_x, curve_y, 'gray', label='Curve p(t)')

    # Compute Mahalanobis distances -> alpha_star
    djs = compute_mahalanobis_distances(m_lambda, points, A_inv_matrices)
    alpha_star = float(np.max(djs))

    colors = ['red', 'green', 'blue']
    for i, (pt, A_inv) in enumerate(zip(points, A_inv_matrices)):
        ax.plot(pt[0], pt[1], marker='o', color=colors[i], label=f'p(t{i + 1})')
        ax.add_patch(ellipse_patch(pt, A_inv, epsilon, edgecolor=colors[i]))
        ax.add_patch(ellipse_patch(pt, A_inv, alpha_star, edgecolor='black', linestyle='--'))

    # Mark m(λ*)
    ax.plot(m_lambda[0], m_lambda[1], marker='x', markersize=10, color='black', label='m(λ*)')
    if result["K_min"] > 0:
        ax.add_patch(ellipse_patch(
            center=m_lambda,
            Ainv=result["B_lambda_inv"],
            radius=np.sqrt(result["K_min"]),
            edgecolor='purple',
            linestyle='--'
        ))


    ax.set_aspect('equal')
    ax.legend()

    # --- Stable axis limits: gather all relevant x,y points ---
    all_x = np.concatenate([curve_x, points[:, 0], [m_lambda[0]]])
    all_y = np.concatenate([curve_y, points[:, 1], [m_lambda[1]]])

    # Compute bounding box + margin
    minx, maxx = np.min(all_x), np.max(all_x)
    miny, maxy = np.min(all_y), np.max(all_y)
    dx = maxx - minx
    dy = maxy - miny
    margin_x = 0.2 * dx  # 20% margin
    margin_y = 0.2 * dy  # 20% margin

    ax.set_xlim(minx - margin_x, maxx + margin_x)
    ax.set_ylim(miny - margin_y, maxy + margin_y)

    return fig


def plot_K_surface(K_func, lambda_grid, opt_pts, b_star=None, num_t=200, elev=30, azim=135, plot_path=False):
    """
    Generate a 3D surface plot of K(λ) over the 2-simplex Δ³ using a grid of (λ₁, λ₂) values.

    This function also marks the global and boundary optima on the surface and optionally draws
    a curve from b* to a simplex vertex representing a 1D path in λ-space.

    :param K_func: callable, takes a 3-vector λ and returns K(λ)
    :param lambda_grid: 1D ndarray, grid values for λ₁ and λ₂ in [0, 1] to form mesh
    :param opt_pts: list of two 3-element arrays (global and boundary minima) with λ and K(λ)
    :param b_star: optional ndarray of shape (3,), boundary minimizer λ
    :param num_t: int, number of samples to interpolate path if plot_path is True (default: 200)
    :param elev: float, elevation angle for the 3D plot (default: 30)
    :param azim: float, azimuth angle for the 3D plot (default: 135)
    :param plot_path: bool, whether to draw the path from b* to a vertex (default: False)

    :return: matplotlib.figure.Figure object with the 3D surface plot
    """
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$\lambda_1$')
    ax.set_ylabel(r'$\lambda_2$')
    ax.set_zlabel(r'$K(\lambda)$')
    ax.set_title("Surface Plot of K(λ) over Δ³")
    ax.view_init(elev=elev, azim=azim)
    # --- surface over the simplex Δ²:
    L1, L2 = np.meshgrid(lambda_grid, lambda_grid, indexing="ij")
    K_vals = np.array([
        [ K_func(np.array([l1, l2, 1 - l1 - l2])) if l1 + l2 <= 1 else np.nan
          for l2 in lambda_grid ]
        for l1 in lambda_grid
    ])
    ax.plot_surface(L1, L2, K_vals,
                    cmap='inferno', edgecolor='none', alpha=0.2)

    # --- scatter any optima you already have, with legend ---
    (l1, l2, l3) = opt_pts[0]  # global minimum
    ax.scatter(l1, l2, l3, color="b", s=50, label="Global minimum")

    (l1, l2, l3) = opt_pts[1]  # boundary minimum
    ax.scatter(l1, l2, l3, color="r", s=50, label="Boundary minimum")

    # Add legend
    ax.legend()

    # --- zero plane for reference:
    zero_plane = np.zeros_like(K_vals)
    ax.plot_surface(L1, L2, zero_plane,
                    color='gray', alpha=0.4, edgecolor='none')

    # Experimental feature! We plot the path from K(b^*) to another vertex.
    # --- now overlay the boundary‐to‐vertex curve if requested:
    if b_star is not None and plot_path:
        t = np.linspace(0, 1, num_t)
        if b_star[2]==0.:

            # λ(t) = ((1-t)b1, (1-t)b2, t)
            lambdas = np.vstack([
                (1 - t) * b_star[0],
                (1 - t) * b_star[1],
                          t
            ]).T
            K_line = np.array([K_func(lam) for lam in lambdas])
            ax.plot(lambdas[:,0], lambdas[:,1], K_line,
                    color='crimson', linewidth=2, label=r'$f1(t)=K(\lambda(t))$')
        elif b_star[1]==0.:
            lambdas = np.vstack([
                (1 - t) * b_star[0],
                t,
                (1 - t) * b_star[2],
            ]).T
            K_line = np.array([K_func(lam) for lam in lambdas])
            ax.plot(lambdas[:, 0], lambdas[:, 1], K_line,
                    color='crimson', linewidth=2, label=r'$f1(t)=K(\lambda(t))$')
        elif b_star[0]==0.:
            lambdas = np.vstack([
                t,
                (1 - t) * b_star[0],
                (1 - t) * b_star[2],
            ]).T
            K_line = np.array([K_func(lam) for lam in lambdas])
            ax.plot(lambdas[:, 0], lambdas[:, 1], K_line,
                    color='crimson', linewidth=2, label=r'$f1(t)=K(\lambda(t))$')
        ax.legend()

    return fig

#===================================================================================================
# 6. Streamlit UI-
#===================================================================================================
st.title("Intersections of Ellipsoidal Balls & K(λ) Surface")
st.sidebar.markdown(
    """Enter a parameterized curve p(t) = (x(t), y(t)) (e.g. `cos(t), sin(t)`),
and adjust sliders for three points and ε to visualize ellipsoidal intersections."""
)
func_str = st.sidebar.text_input("Enter x(t), y(t)", "cos(t)*(1.5+1.4*cos(2*t)^2), sin(t)*(1.5+1.4*cos(2*t)^2)")
solver = st.sidebar.radio("Solver", ["SLSQP", "pga"])
# Sidebar sliders for view angles
elev = st.sidebar.slider("Elevation angle for Surface", min_value=0, max_value=90, value=30)
azim = st.sidebar.slider("Azimuth angle for Surface", min_value=0, max_value=360, value=135)
fx, fy, t_sym = parse_functions(func_str)

if fx is not None and fy is not None:
    # Get three t values--these are the local coordinates for the k=3 points on the curve/manifold.
    t1, t2, t3 = [st.sidebar.slider(f"t{i}", 0.0, 2 * np.pi, i * np.pi / 2, step=0.01) for i in range(1, 4)]
    epsilon = st.sidebar.slider("ε", 0.01, 8.0, 1.0, step=0.01)
    plot_path = st.sidebar.toggle("Plot bd^*-to-vertex path (experimental)", False)
    ts = [t1, t2, t3]
    points, A_matrices = [], []
    for t_val in ts:
        pt, A = compute_precision_matrix(fx, fy, t_sym, t_val)
        points.append(pt)
        A_matrices.append(A)
    points = np.array(points)

    # Precompute inverses and quadratic forms for the K(λ) minimization.
    A_inv_list, A_inv_array, x_Ainv_x = precompute_matrix_operations(np.array(points), A_matrices)

    # Minimize K(λ)
    result = minimize_K(epsilon, points, A_inv_array, x_Ainv_x, solver)
    lambda_star = result["lambda"]
    m_lambda = result['m_lambda']

    # Plot the curve and ellipses as before.
    st.pyplot(plot_curve_and_ellipses(fx, fy, points, A_inv_list, epsilon, m_lambda, result))

    # --- New: Test Theorem 6 condition ---
    # 1. Compute K(b*) (minimum of K(λ) on the boundary)
    boundary_solver = minimize_K_boundary(epsilon, points, A_inv_array, x_Ainv_x)
    k_b_star = boundary_solver["K_min"]
    b_star = boundary_solver["lambda"]

    # Plot the K(λ) surface.
    lambda_grid = np.linspace(0, 1, 30)
    K_func = lambda lam: compute_K(lam, epsilon, points, A_inv_array, x_Ainv_x)
    opt1 = np.array([result['lambda'][0], result['lambda'][1], result['K_min']])
    opt2 = np.array([b_star[0], b_star[1], k_b_star])
    st.pyplot(plot_K_surface(K_func, lambda_grid, [opt1, opt2],
                             b_star=b_star, elev=elev, azim=azim, plot_path=plot_path))

    # Compute Mahalanobis distances → δ_max (here called alpha_star)
    djs = compute_mahalanobis_distances(m_lambda, points, A_inv_list)
    delta_max = float(np.max(djs))

    # 2. Compute upper bounds (α and β) from the precision matrices
    alpha_bound, beta_bound = compute_precision_bounds(A_inv_list)
    # 3. For the standard simplex in R^3, the diameter D satisfies D^2 = 2.
    # Then the condition of Theorem 6 becomes: b_star > (beta/alpha)* (delta_max)^2.
    threshold = 2 * (beta_bound / alpha_bound) * (delta_max ** 2)
    theorem6_holds = k_b_star > threshold

    # --- Display results ---
    st.markdown(f"""
    **Results:**
    - Minimizing $\\lambda^*$ = {result['lambda'].round(3)}
    - Minimum $K(\\lambda^*)$ (over Δ) = {result['K_min']:.3f}
    - $m(\\lambda^*)$ = {np.array(m_lambda).round(3)}
    - $\\epsilon^* = $ {delta_max}
     
    **Curvature Bound Check:**
    - Minimum $K(b^*)$ on the boundary, $K(b^*)$ = {k_b_star:.3f}
    - Minimizer $b^*$ on the boundary, $b^*$ = {b_star}
    - Curvature threshold $(\\beta/\\alpha)\\delta^2$ = {threshold:.3f}
    - **Condition:** $K(b^*) > 2(\\beta/\\alpha)\\delta^2$ is **{"satisfied" if theorem6_holds else "not satisfied"}**.
    
    If this condition holds then K(λ) > 0 for all λ ∈ Δ, and hence the ellipsoidal
    balls intersect.
    """)
else:
    st.stop()
