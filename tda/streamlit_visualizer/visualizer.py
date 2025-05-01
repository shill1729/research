import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import sympy as sp

from scipy.optimize import minimize, minimize_scalar
from matplotlib.patches import Ellipse

# TODO: finish doc-strings and maybe some additional comments

#=============================================================================================
# 1. Functions for computing $K( \lambda ) $, and its Gradient
#=============================================================================================
def precompute_matrix_operations(centers, pd_matrix_list):
    """
    Given points with shape (k, D) and a list of (D,D) matrices A_list (length k),
    compute each inverse A_i^{-1}, stack them into an array, and also compute
    the quadratic forms x_i^T A_i^{-1} x_i.

    :param centers:
    :param pd_matrix_list:
    :return:
    """
    k, D = centers.shape
    A_inv_list = [np.linalg.inv(pd_matrix_list[i]) for i in range(k)]
    A_inv_array = np.stack(A_inv_list, axis=0)
    # Computing the quadratic form: x^ T A^{-1} x
    x_Ainv_x = np.array([centers[i].T @ A_inv_list[i] @ centers[i] for i in range(k)])
    return A_inv_list, A_inv_array, x_Ainv_x

def compute_B_inv_and_m_lambda(lmbd, A_inv_array, x):
    """


    :param lmbd:
    :param A_inv_array:
    :param x:
    :return:
    """
    B_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, x), axis=0)
    m_lambda = np.linalg.solve(B_lambda_inv, S)
    return B_lambda_inv, m_lambda, S

def compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x):
    """
    Computes K(λ) = eps^2 - C(λ), where
       A_λ^{-1} = Σ_i λ_i A_i^{-1}
       S = Σ_i λ_i (A_i^{-1} @ xs[i])
       m(λ) is obtained by solving A_λ^{-1} m = S,
       and C(λ) = (Σ_i λ_i x_i^T A_i^{-1} x_i) - m(λ)^T S.

    :param lmbd:
    :param eps:
    :param xs:
    :param A_inv_array:
    :param x_Ainv_x:
    :return:
    """
    # B_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    # S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    # try:
    #     m_lambda = np.linalg.solve(B_lambda_inv, S)
    # except np.linalg.LinAlgError:
    #     print("Non PD!")
    #     return 1e10  # Penalty for singularity
    B_lambda_inv, m_lambda, S = compute_B_inv_and_m_lambda(lmbd, A_inv_array, xs)
    quad_term = m_lambda.dot(S)
    sum_term = np.dot(lmbd, x_Ainv_x)
    return eps ** 2 - (sum_term - quad_term)

def compute_K_gradient(lmbd, xs, A_inv_array):
    """
    Computes the gradient ∇K(λ) using the simplified formula:
      ∂K/∂λ_j = -(x_j - m(λ))^T A_j^{-1} (x_j - m(λ))

    :param lmbd:
    :param xs:
    :param A_inv_array:
    :return:
    """
    B_lambda_inv, m_lambda, S = compute_B_inv_and_m_lambda(lmbd, A_inv_array, xs)
    diff = m_lambda - xs  # (x_j - m(λ)) for each j
    grad = -np.einsum('ij,ijk,ik->i', diff, A_inv_array, diff)
    return grad

#================================================================================================
# 2. Optimization functions.
#================================================================================================
def project_simplex(v):
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


def projected_gradient_descent(eps, xs, A_inv_array, x_Ainv_x, step_size=0.001, max_iters=5000, tol=1e-10):
    """
    Minimize K(λ) using projected gradient descent onto the probability simplex.

    :param eps:
    :param xs:
    :param A_inv_array:
    :param x_Ainv_x:
    :param step_size:
    :param max_iters:
    :param tol:
    :return:
    """
    k, D = xs.shape
    lmbd = np.ones(k) / k

    for _ in range(max_iters):
        grad = compute_K_gradient(lmbd, xs, A_inv_array)
        lmbd_new = project_simplex(lmbd - step_size * grad)
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


def minimize_K(eps, xs, A_inv_array, x_Ainv_x, solver="pga"):
    """
    Minimizes K(λ) = eps^2 - C(λ) over the probability simplex using the simplified gradient.

    :param eps:
    :param xs:
    :param A_inv_array:
    :param x_Ainv_x:
    :param solver:
    :return:
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
    Find the minimum of K(λ) = eps^2 - C(λ) over just the 1D edges of Δ^k.
    Returns the best λ on any edge, its K-value, m(λ), A(λ), and solver info.
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
    """For each precision matrix A⁻¹, compute its maximum eigenvalue.
       Then set α = min(max_eigs) and β = max(max_eigs).
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

    :param m_lambda:
    :param points:
    :param A_inv_list:
    :return:
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

    :param func_str:
    :return:
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

    :param fx:
    :param fy:
    :param t_sym:
    :param t_val:
    :return:
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
    Given an inverse precision matrix Ainv, create an Ellipse patch.
    The semi-axis lengths are computed as radius/sqrt(eigenvalue) and doubled for full axis lengths.

    :param center:
    :param Ainv:
    :param radius:
    :param edgecolor:
    :param linestyle:
    :return:
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

    :param fx:
    :param fy:
    :param points:
    :param A_inv_matrices:
    :param epsilon:
    :param m_lambda:
    :param result:
    :return:
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
    :param K_func:      function taking a length‑3 array λ↦K(λ)
    :param lambda_grid: 1D array of λ₁,λ₂ values for meshing the simplex
    :param opt_pts:     list of optimizer points [ (λ1,λ2,λ3), ... ] to scatter
    :param b_star:      optional length‑3 array [b1*, b2*, 0] defining boundary minimizer
    :param num_t:       number of samples along t∈[0,1] for the red line
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

    # # --- scatter any optima you already have:
    # color = "b"
    # for (l1, l2, l3) in opt_pts:
    #     ax.scatter(l1, l2, l3, color=color, s=50)
    #     color = "r"

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
and adjust sliders for three points and ε to visualize ellipsoidal intersections.
Now also testing the sufficient condition from Theorem 6."""
)
func_str = st.sidebar.text_input("Enter x(t), y(t)", "cos(t)*(1.5+1.4*cos(2*t)^2), sin(t)*(1.5+1.4*cos(2*t)^2)")
solver = st.sidebar.radio("Solver", ["SLSQP", "pga"])
# Sidebar sliders for view angles
elev = st.sidebar.slider("Elevation angle", min_value=0, max_value=90, value=30)
azim = st.sidebar.slider("Azimuth angle", min_value=0, max_value=360, value=135)
fx, fy, t_sym = parse_functions(func_str)

if fx is not None and fy is not None:
    # Get three t values
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
