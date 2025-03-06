import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp
from matplotlib.patches import Ellipse
from scipy.optimize import minimize


def precompute_A_inverses(xs, A_list):
    """
    Given points xs (shape (k,D)) and a list of matrices A_list (length k),
    compute each inverse A_i^{-1}, stack them into an array, and also compute
    the quadratic forms x_i^T A_i^{-1} x_i.

    :param xs:
    :param A_list:
    :return:
    """
    k, D = xs.shape
    A_inv_list = [np.linalg.inv(A_list[i]) for i in range(k)]
    A_inv_array = np.stack(A_inv_list, axis=0)
    x_Ainv_x = np.array([xs[i].T @ A_inv_list[i] @ xs[i] for i in range(k)])
    return A_inv_list, A_inv_array, x_Ainv_x


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
    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        print("Non PD!")
        return 1e10  # Penalty for singularity
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
    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        return np.zeros_like(lmbd)
    diff = m_lambda - xs  # (x_j - m(λ)) for each j
    grad = -np.einsum('ij,ijk,ik->i', diff, A_inv_array, diff)
    return grad


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


def cauchy_simplex_minimize_K(eps, xs, A_list, A_inv_array, x_Ainv_x, max_iters=300, tol=1e-9):
    """
    Implements the Cauchy-Simplex algorithm for minimizing K(λ) over the probability simplex,
    using the simplified gradient.

    :param eps:
    :param xs:
    :param A_list:
    :param A_inv_array:
    :param x_Ainv_x:
    :param max_iters:
    :param tol:
    :return:
    """
    k, D = xs.shape
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
            opt_lmbd = lmbd
            A_lambda_inv = np.tensordot(opt_lmbd, A_inv_array, axes=([0], [0]))
            S = np.sum(opt_lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
            try:
                m_lambda = np.linalg.solve(A_lambda_inv, S)
            except np.linalg.LinAlgError:
                m_lambda = None
            return {"lambda": lmbd, "K_min": compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x),
                    "m_lambda": m_lambda, "success": True}
    # If no convergence:
    opt_lmbd = lmbd
    A_lambda_inv = np.tensordot(opt_lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(opt_lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        m_lambda = None
    return {"lambda": lmbd, "K_min": compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x),
            "m_lambda": m_lambda, "success": False}


def projected_gradient_descent(eps, xs, A_list, A_inv_array, x_Ainv_x, step_size=0.001, max_iters=5000, tol=1e-10):
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

    for _ in range(max_iters):
        grad = compute_K_gradient(lmbd, xs, A_inv_array)
        lmbd_new = project_simplex(lmbd - step_size * grad)
        if np.linalg.norm(lmbd_new - lmbd) < tol:
            break
        lmbd = lmbd_new

    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
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


def minimize_K(eps, xs, A_list, A_inv_array, x_Ainv_x, solver="pga"):
    """
    Minimizes K(λ) = eps^2 - C(λ) over the probability simplex using the simplified gradient.

    :param eps:
    :param xs:
    :param A_list:
    :param A_inv_array:
    :param x_Ainv_x:
    :param solver:
    :return:
    """
    k, D = xs.shape
    if A_list is None:
        A_list = [np.eye(D) for _ in range(k)]
    if solver == "pga":
        return projected_gradient_descent(eps, xs, A_list, A_inv_array, x_Ainv_x)
    elif solver == "cs":
        return cauchy_simplex_minimize_K(eps, xs, A_list, A_inv_array, x_Ainv_x)

    # For other solvers, precompute inverses once.

    def obj(lmbd, eps, xs, A_inv_array, x_Ainv_x):
        return compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x)

    def jac(lmbd, eps, xs, A_inv_array, x_Ainv_x):
        return compute_K_gradient(lmbd, xs, A_inv_array)

    lmbd0 = np.ones(k) / k
    constraints = {'type': 'eq', 'fun': lambda lmbd: np.sum(lmbd) - 1}
    bounds = [(0, 1) for _ in range(k)]

    res = minimize(obj, lmbd0, args=(eps, xs, A_inv_array, x_Ainv_x), method=solver,
                   jac=jac, bounds=bounds, constraints=constraints, tol=1e-10,
                   options={"maxiter": 5000})

    opt_lmbd = res.x
    A_lambda_inv = np.tensordot(opt_lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(opt_lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        m_lambda = None
    A_lambda = np.linalg.inv(A_lambda_inv) if m_lambda is not None else None

    return {'lambda': opt_lmbd, 'K_min': res.fun, 'm_lambda': m_lambda,
            'A_lambda': A_lambda, 'success': res.success, 'message': res.message}


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


def compute_mahalanobis_distances(m_lambda, points, A_inv_list):
    """
    Compute the Mahalanobis distances d_j = sqrt((m - x_j)^T A(x_j)^{-1} (m - x_j)) for each j.
    """
    djs = []
    for i in range(len(points)):
        diff = m_lambda - points[i]
        d_j = diff.T @ A_inv_list[i] @ diff
        djs.append(np.sqrt(d_j))
    return np.array(djs)


def plot_curve_and_ellipses(fx, fy, points, A_matrices, A_inv_matrices, epsilon, m_lambda):
    """

    :param fx:
    :param fy:
    :param points:
    :param A_matrices:
    :param A_inv_matrices:
    :param epsilon:
    :param m_lambda:
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


def plot_K_surface(K_func, lambda_grid, opt_pts):
    """

    :param K_func:
    :param lambda_grid:
    :param opt_pts:
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("λ1")
    ax.set_ylabel("λ2")
    ax.set_zlabel("K(λ)")
    ax.set_title("Surface Plot of K(λ) over Δ³")
    L1, L2 = np.meshgrid(lambda_grid, lambda_grid, indexing="ij")
    K_vals = np.array([[K_func(np.array([l1, l2, 1 - l1 - l2]))
                        if l1 + l2 <= 1 else np.nan for l2 in lambda_grid] for l1 in lambda_grid])
    ax.plot_surface(L1, L2, K_vals, cmap='inferno', edgecolor='none', alpha=0.2)
    for opt_pt in opt_pts:
        ax.scatter(opt_pt[0], opt_pt[1], opt_pt[2])
    zero_plane = np.zeros_like(K_vals)
    ax.plot_surface(L1, L2, zero_plane, color='gray', alpha=0.4, edgecolor='none')
    return fig


# --- New helper functions for Theorem 6 ---

def sample_boundary_K(K_func, num_samples=100):
    """Sample K(λ) along the three edges of the 3-simplex (with k = 3).
       The edges are:
         Edge 1: λ = [t, 1-t, 0]
         Edge 2: λ = [t, 0, 1-t]
         Edge 3: λ = [0, t, 1-t]
       Return the minimum K-value among all samples.
    """
    b_values = []
    ts = np.linspace(0, 1, num_samples)
    # Edge 1: λ = [t, 1-t, 0]
    for t in ts:
        lam = np.array([t, 1 - t, 0.0])
        b_values.append(K_func(lam))
    # Edge 2: λ = [t, 0, 1-t]
    for t in ts:
        lam = np.array([t, 0.0, 1 - t])
        b_values.append(K_func(lam))
    # Edge 3: λ = [0, t, 1-t]
    for t in ts:
        lam = np.array([0.0, t, 1 - t])
        b_values.append(K_func(lam))
    return np.min(b_values)


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


# --- Streamlit UI ---

st.title("Intersections of Ellipsoidal Balls & K(λ) Surface")
st.sidebar.markdown(
    """Enter a parameterized curve p(t) = (x(t), y(t)) (e.g. `cos(t), sin(t)`),
and adjust sliders for three points and ε to visualize ellipsoidal intersections.
Now also testing the sufficient condition from Theorem 6."""
)
func_str = st.sidebar.text_input("Enter x(t), y(t)", "cos(t)*(1.5+1.4*cos(2*t)^2), sin(t)*(1.5+1.4*cos(2*t)^2)")
solver = st.sidebar.radio("Solver", ["SLSQP", "cs", "pga"])
fx, fy, t_sym = parse_functions(func_str)

if fx is not None and fy is not None:
    # Get three t values
    t1, t2, t3 = [st.sidebar.slider(f"t{i}", 0.0, 2 * np.pi, i * np.pi / 2, step=0.01) for i in range(1, 4)]
    epsilon = st.sidebar.slider("ε", 0.01, 5.0, 1.0, step=0.05)
    ts = [t1, t2, t3]
    points, A_matrices = [], []
    for t_val in ts:
        pt, A = compute_precision_matrix(fx, fy, t_sym, t_val)
        points.append(pt)
        A_matrices.append(A)
    points = np.array(points)

    # Precompute inverses and quadratic forms for the K(λ) minimization.
    A_inv_list, A_inv_array, x_Ainv_x = precompute_A_inverses(np.array(points), A_matrices)

    # Minimize K(λ)
    result = minimize_K(epsilon, points, A_matrices, A_inv_array, x_Ainv_x, solver)
    m_lambda = result['m_lambda']

    # Plot the curve and ellipses as before.
    st.pyplot(plot_curve_and_ellipses(fx, fy, points, A_matrices, A_inv_list, epsilon, m_lambda))

    # Plot the K(λ) surface.
    lambda_grid = np.linspace(0, 1, 30)
    K_func = lambda lam: compute_K(lam, epsilon, points, A_inv_array, x_Ainv_x)
    opt1 = np.array([result['lambda'][0], result['lambda'][1], result['K_min']])
    st.pyplot(plot_K_surface(K_func, lambda_grid, [opt1]))

    # Compute Mahalanobis distances → δ_max (here called alpha_star)
    djs = compute_mahalanobis_distances(m_lambda, points, A_inv_list)
    delta_max = float(np.max(djs))

    # --- New: Test Theorem 6 condition ---
    # 1. Compute b* (minimum of K(λ) on the boundary)
    b_star = sample_boundary_K(K_func, num_samples=100)
    # 2. Compute curvature bounds (α and β) from the precision matrices
    alpha_bound, beta_bound = compute_precision_bounds(A_inv_list)
    # 3. For the standard simplex in R^3, the diameter D satisfies D^2 = 2.
    # Then the condition of Theorem 6 becomes: b_star > (beta/alpha)* (delta_max)^2.
    threshold = 2 * (beta_bound / alpha_bound) * (delta_max ** 2)
    theorem6_holds = b_star > threshold

    # --- Display results ---
    st.markdown(f"""
    **Results:**
    - Minimizing $\\lambda^*$ = {result['lambda'].round(3)}
    - Minimum $K(\\lambda^*)$ (over Δ) = {result['K_min']:.3f}
    - $m(\\lambda^*)$ = {np.array(m_lambda).round(3)}
    - $\\epsilon^*$ (max Mahalanobis distance) = {delta_max:.3f}

    **Theorem 6 Check:**
    - Minimum $K(\\lambda)$ on the boundary, $b^*$ = {b_star:.3f}
    - Curvature threshold $(\\beta/\\alpha)\\delta^2$ = {threshold:.3f}
    - **Condition:** $b^* > 2(\\beta/\\alpha)\\delta^2$ is **{"satisfied" if theorem6_holds else "not satisfied"}**.

    According to Theorem 6, if the condition holds then K(λ) > 0 for all λ ∈ Δ, and hence the ellipsoidal
    balls intersect.
    """)
else:
    st.stop()

# # --- Streamlit UI ---
#
# st.title("Intersections ellipses and the surface")
# st.sidebar.markdown(
#     """Enter a parameterized curve p(t) = (x(t), y(t)) (e.g. `cos(t), sin(t)`),
# and adjust sliders for three points and ε to visualize ellipsoidal intersections."""
# )
# func_str = st.sidebar.text_input("Enter x(t), y(t)", "cos(t), sin(t)")
# # solver = st.sidebar.text_input("Enter solver:", "pga")
# solver = st.sidebar.radio("Solver", ["SLSQP", "cs", "pga"])
# fx, fy, t_sym = parse_functions(func_str)
#
# if fx is not None and fy is not None:
#     t1, t2, t3 = [st.sidebar.slider(f"t{i}", 0.0, 2 * np.pi, i * np.pi / 2, step=0.01) for i in range(1, 4)]
#     epsilon = st.sidebar.slider("ε", 0.01, 5.0, 1.0, step=0.05)
#     ts = [t1, t2, t3]
#     points, A_matrices = [], []
#     for t_val in ts:
#         pt, A = compute_precision_matrix(fx, fy, t_sym, t_val)
#         points.append(pt)
#         A_matrices.append(A)
#     points = np.array(points)
#     A_inv_list, A_inv_array, x_Ainv_x = precompute_A_inverses(np.array(points), A_matrices)
#     result = minimize_K(epsilon, points, A_matrices, A_inv_array, x_Ainv_x, solver)
#     m_lambda = result['m_lambda']
#
#     st.pyplot(plot_curve_and_ellipses(fx, fy, points, A_matrices, A_inv_list, epsilon, m_lambda))
#     lambda_grid = np.linspace(0, 1, 30)
#     K_func = lambda lam: compute_K(lam, epsilon, points, A_inv_array, x_Ainv_x)
#
#     opt1 = np.array([result['lambda'][0], result['lambda'][1], result['K_min']])
#
#     st.pyplot(plot_K_surface(K_func, lambda_grid, [opt1]))
#     # Compute Mahalanobis distances -> alpha_star
#     djs = compute_mahalanobis_distances(m_lambda, points, A_inv_list)
#     alpha_star = float(np.max(djs))
#     st.markdown(f"""
#     **Results:**
#     - Minimizing $\\lambda^*$ = {result['lambda'].round(3)}
#     - Minimum $K(\\lambda^*)$ = {result['K_min']:.3f}
#     - $m(\\lambda^*)$ = {np.array(m_lambda).round(3)}
#     - $\\epsilon^* =$ {alpha_star:.3f}
#     """)
# else:
#     st.stop()
