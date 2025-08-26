# streamlit_app.py
# Relativistic random motion: velocity diffusion on the unit future timelike hyperboloid
# and induced position process in Minkowski space.
#
# Implements the local (Poincaré ball) SDE for U_s in (𝔻, \tilde g):
#   dU = (1/4)(d-2)(1 - ||U||^2) U ds + (1/2)(1 - ||U||^2) dW
# with independent Brownian components (Itô). We then map to the ambient
# Minkowski 4-velocity via
#   φ(u) = ((1 + ||u||^2)/(1 - ||u||^2),  2u/(1 - ||u||^2))
# which lives on the unit future timelike hyperboloid g(V, V) = -1, and integrate
# dX_s = V_s ds to obtain the worldline X_s.
#
# Usage:
#   streamlit run streamlit_app.py
#
# Dependencies: streamlit, numpy, matplotlib

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D projection)

# ----------------------------
# Utility functions (math core)
# ----------------------------

def drift_u(u, d):
    """
    Drift for U in local coordinates (Itô):
        μ(u) = (1/4)(d - 2)(1 - ||u||^2) u
    u: (d,) array
    """
    r2 = np.dot(u, u)
    return 0.25 * (d - 2.0) * (1.0 - r2) * u


def sigma_scalar(u):
    """
    Scalar diffusion amplitude multiplying each independent Brownian component:
        σ(u) = (1/2)(1 - ||u||^2)
    """
    r2 = np.dot(u, u)
    return 0.5 * (1.0 - r2)


def project_to_open_ball(u, eps=1e-9):
    """
    Ensure u remains strictly inside the open unit ball: ||u|| < 1.
    If ||u|| >= 1 - eps, scale back to radius (1 - eps).
    """
    r = np.linalg.norm(u)
    rmax = 1.0 - eps
    if r >= rmax and r > 0.0:
        u = (rmax / r) * u
    return u


def phi_to_minkowski_velocity(u):
    """
    Map local velocity u ∈ 𝔻 (unit ball in R^d) to ambient Minkowski (t', x'):
        t' = (1 + ||u||^2) / (1 - ||u||^2)
        x' = 2 u / (1 - ||u||^2)
    Returns (tprime, xprime) with shapes () and (d,).
    Assumes ||u|| < 1.
    """
    r2 = np.dot(u, u)
    denom = 1.0 - r2
    tprime = (1.0 + r2) / denom
    xprime = (2.0 / denom) * u
    return tprime, xprime


def simulate_U_paths(d, S, N, n_paths, rng, u0_vec, clamp_eps=1e-9):
    """
    Euler–Maruyama simulation for U_s in the Poincaré ball coordinates.
    Returns U_paths with shape (N+1, n_paths, d).
    """
    dt = S / N
    sqrt_dt = np.sqrt(dt)
    U = np.zeros((N + 1, n_paths, d), dtype=np.float64)

    # Initialize each path with provided u0_vec, optionally perturbed per-path if desired.
    for k in range(n_paths):
        u0 = np.array(u0_vec, dtype=np.float64)
        u0 = project_to_open_ball(u0, eps=clamp_eps)
        U[0, k] = u0

    # Time stepping
    for n in range(N):
        # For each path, step independently
        for k in range(n_paths):
            u = U[n, k].copy()
            # Drift
            mu = drift_u(u, d)
            # Diffusion
            sig = sigma_scalar(u)
            dW = sqrt_dt * rng.normal(size=d)
            u_next = u + mu * dt + sig * dW
            u_next = project_to_open_ball(u_next, clamp_eps)
            U[n + 1, k] = u_next

    return U


def minkowski_velocity_from_U(U):
    """
    Vectorized mapping V = φ(U) for all times and paths.
    U: (N+1, n_paths, d)
    Returns V: (N+1, n_paths, d+1) where V[..., 0] = t', V[..., 1:] = x'.
    Also returns normalization error array err with shape (N+1, n_paths):
        err = ( -t'^2 + ||x'||^2 ) - ( -1 ) = -t'^2 + ||x'||^2 + 1
    """
    Np1, n_paths, d = U.shape
    V = np.zeros((Np1, n_paths, d + 1), dtype=np.float64)
    err = np.zeros((Np1, n_paths), dtype=np.float64)

    for n in range(Np1):
        for k in range(n_paths):
            u = U[n, k]
            tprime, xprime = phi_to_minkowski_velocity(u)
            V[n, k, 0] = tprime
            V[n, k, 1:] = xprime
            # Check Minkowski norm: g(V, V) = -1
            err[n, k] = (-tprime * tprime + np.dot(xprime, xprime)) + 1.0

    return V, err


def integrate_worldline(V, dt, X0=None):
    """
    Integrate dX_s = V_s ds using forward Euler:
        X_{n+1} = X_n + V_n dt
    V: (N+1, n_paths, d+1)
    dt: scalar
    X0: optional initial position (n_paths, d+1) or (d+1,)
         defaults to zeros.
    Returns X: (N+1, n_paths, d+1)
    """
    Np1, n_paths, D1 = V.shape
    if X0 is None:
        X0 = np.zeros((n_paths, D1), dtype=np.float64)
    else:
        X0 = np.array(X0, dtype=np.float64)
        if X0.ndim == 1:
            X0 = np.repeat(X0[None, :], n_paths, axis=0)

    X = np.zeros((Np1, n_paths, D1), dtype=np.float64)
    X[0] = X0
    for n in range(Np1 - 1):
        X[n + 1] = X[n] + V[n] * dt
    return X


def parse_vector_from_text(text, d, default=None):
    """
    Parse a comma-separated vector of length d from text input.
    """
    try:
        parts = [float(x.strip()) for x in text.split(",")]
        if len(parts) != d:
            raise ValueError
        vec = np.array(parts, dtype=np.float64)
        return vec
    except Exception:
        if default is None:
            return np.zeros(d, dtype=np.float64)
        return np.array(default, dtype=np.float64)


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Relativistic Random Motion Simulator", layout="wide")

st.title("Relativistic Random Motion: Velocity Diffusion on the Unit Hyperboloid")

with st.sidebar:
    st.header("Simulation Parameters")

    d = st.selectbox("Spatial dimension d", options=[1, 2, 3, 4, 5], index=1)
    total_time = st.number_input("Total proper time S", min_value=0.1, value=5.0, step=0.1, format="%.3f")
    N = st.number_input("Number of steps N", min_value=100, value=4000, step=100)
    n_paths = st.number_input("Number of paths", min_value=1, max_value=10, value=1, step=1)

    seed = st.number_input("Random seed (int)", min_value=0, value=12345, step=1)
    rng = np.random.default_rng(int(seed))

    st.markdown("**Initial local velocity $U_0 \\in \\mathbb{D}$ (comma-separated):**")
    default_u0_text = ",".join(["0"] * d)
    u0_text = st.text_input("U0", value=default_u0_text)
    u0_vec = parse_vector_from_text(u0_text, d)

    # Optional helper to set U0 via radius and direction along a coordinate axis.
    with st.expander("Quick set U0 by radius and axis"):
        r0 = st.slider("||U0|| (must be < 1)", min_value=0.0, max_value=0.99, value=float(np.linalg.norm(u0_vec)), step=0.01)
        axis_choice = st.selectbox("Axis for direction", options=[f"axis {j}" for j in range(d)] + ["random"], index=0)
        if st.button("Apply radius/axis to U0"):
            if axis_choice == "random":
                vec = rng.normal(size=d)
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 0:
                    vec = vec / vec_norm
                else:
                    vec = np.zeros(d)
                    vec[0] = 1.0
            else:
                j = int(axis_choice.split()[-1])
                vec = np.zeros(d)
                vec[j] = 1.0
            u0_vec = r0 * vec
            st.success(f"U0 set to {u0_vec.tolist()}")

    # Initial Minkowski position X0 (t0, x0_1, ..., x0_d)
    st.markdown("**Initial Minkowski position $X_0=(t_0, x_0)$ (comma-separated):**")
    default_X0_text = ",".join(["0"] * (d + 1))
    X0_text = st.text_input("X0", value=default_X0_text)
    X0_vec = parse_vector_from_text(X0_text, d + 1)

    clamp_eps = st.number_input("Clamping ε for ||U||<1", min_value=1e-12, max_value=1e-3, value=1e-9, format="%.1e")
    show_speed = st.checkbox("Show 3-speed |x'| / t' plot", value=True)
    show_tables = st.checkbox("Show raw data tables (U, V, X)", value=False)
    # Sidebar toggles (put near other checkboxes)
    show_cone_spacetime = st.checkbox("Show spacetime plot with light cone (uses first two spatial dims)", value=True)
    show_cone_snapshots = st.checkbox("Overlay light-disc snapshots on spatial plot", value=False)
    n_cone_slices = st.slider("Number of light-disc snapshots", min_value=1, max_value=10, value=4)

# Run simulation
dt = float(total_time) / int(N)
U_paths = simulate_U_paths(d=d, S=total_time, N=int(N), n_paths=int(n_paths), rng=rng, u0_vec=u0_vec, clamp_eps=clamp_eps)
V_paths, mink_err = minkowski_velocity_from_U(U_paths)
X_paths = integrate_worldline(V_paths, dt=dt, X0=X0_vec)

# Diagnostics
max_abs_err = np.max(np.abs(mink_err))
mean_abs_err = float(np.mean(np.abs(mink_err)))
speed_vals = np.linalg.norm(V_paths[..., 1:], axis=-1) / np.maximum(V_paths[..., 0], 1e-15)  # shape (N+1, n_paths)

st.subheader("Minkowski Norm Check for 4-Velocity")
c1, c2, c3 = st.columns(3)
c1.metric("max |g(V,V)+1|", f"{max_abs_err:.3e}")
c2.metric("mean |g(V,V)+1|", f"{mean_abs_err:.3e}")
c3.metric("time step dt", f"{dt:.3e}")

# Time grid
tgrid = np.linspace(0.0, total_time, int(N) + 1)

# ----------------------------
# Plots: U in local coordinates
# ----------------------------

st.subheader("Velocity in Local Coordinates $U_s \\in \\mathbb{D}$")

with st.expander("Component-wise time series for U"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(int(n_paths)):
        for j in range(d):
            ax.plot(tgrid, U_paths[:, k, j], lw=1.0, label=f"path {k+1}, U[{j}]")
    ax.set_xlabel("proper time s")
    ax.set_ylabel("U components")
    ax.set_title("Local velocity components U(s)")
    if n_paths * d <= 12:
        ax.legend(loc="upper right", fontsize="small", ncol=2)
    st.pyplot(fig, clear_figure=True)

if d >= 2:
    st.markdown("Projection of U(s) in the first two coordinates (unit disk).")
    fig, ax = plt.subplots(figsize=(5, 5))
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), ls="--", lw=1.0)
    for k in range(int(n_paths)):
        ax.plot(U_paths[:, k, 0], U_paths[:, k, 1], lw=1.0, label=f"path {k+1}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("U[0]")
    ax.set_ylabel("U[1]")
    ax.set_title("U(s) trajectory in the unit disk (first two dims)")
    if n_paths <= 8:
        ax.legend(loc="best", fontsize="small")
    st.pyplot(fig, clear_figure=True)

if d >= 3:
    st.markdown("3D projection of U(s) in the first three coordinates.")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    for k in range(int(n_paths)):
        ax.plot(U_paths[:, k, 0], U_paths[:, k, 1], U_paths[:, k, 2], lw=1.0)
    ax.set_xlabel("U[0]"); ax.set_ylabel("U[1]"); ax.set_zlabel("U[2]")
    ax.set_title("U(s) in first three local coordinates")
    st.pyplot(fig, clear_figure=True)

# ----------------------------
# Plots: V in Minkowski coordinates
# ----------------------------

st.subheader("4-Velocity in Minkowski Coordinates $V_s = (t'_s, x'_s)$")

with st.expander("Component-wise time series for V = φ(U)"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(int(n_paths)):
        ax.plot(tgrid, V_paths[:, k, 0], lw=1.0, label=f"path {k+1}, t'")
    ax.set_xlabel("proper time s")
    ax.set_ylabel("t' (gamma)")
    ax.set_title("Temporal component t'(s) = γ(s)")
    if n_paths <= 10:
        ax.legend(loc="best", fontsize="small")
    st.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(int(n_paths)):
        for j in range(d):
            ax.plot(tgrid, V_paths[:, k, 1 + j], lw=1.0, label=f"path {k+1}, x'[{j}]")
    ax.set_xlabel("proper time s")
    ax.set_ylabel("spatial components of 4-velocity")
    ax.set_title("Spatial part of 4-velocity x'(s)")
    if n_paths * d <= 12:
        ax.legend(loc="upper right", fontsize="small", ncol=2)
    st.pyplot(fig, clear_figure=True)

if show_speed:
    st.markdown("Three-speed $|\\mathbf{v}| = \\|x'\\|/t'$ (must be < 1).")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for k in range(int(n_paths)):
        ax.plot(tgrid, speed_vals[:, k], lw=1.0, label=f"path {k+1}")
    ax.axhline(1.0, ls="--", lw=1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("proper time s")
    ax.set_ylabel("|v|")
    ax.set_title("Three-speed vs proper time")
    if n_paths <= 10:
        ax.legend(loc="best", fontsize="small")
    st.pyplot(fig, clear_figure=True)

if d >= 2:
    st.markdown("Projection of the spatial part of 4-velocity x'(s) onto the first two spatial dims.")
    fig, ax = plt.subplots(figsize=(5, 5))
    for k in range(int(n_paths)):
        ax.plot(V_paths[:, k, 1], V_paths[:, k, 2], lw=1.0, label=f"path {k+1}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x'[0]")
    ax.set_ylabel("x'[1]")
    ax.set_title("Spatial 4-velocity trajectory (first two dims)")
    if n_paths <= 8:
        ax.legend(loc="best", fontsize="small")
    st.pyplot(fig, clear_figure=True)

if d >= 3:
    st.markdown("3D projection of the spatial part of 4-velocity x'(s) onto first three spatial dims.")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    for k in range(int(n_paths)):
        ax.plot(V_paths[:, k, 1], V_paths[:, k, 2], V_paths[:, k, 3], lw=1.0)
    ax.set_xlabel("x'[0]"); ax.set_ylabel("x'[1]"); ax.set_zlabel("x'[2]")
    ax.set_title("Spatial 4-velocity (first three dims)")
    st.pyplot(fig, clear_figure=True)

# ----------------------------
# Plots: Position X in Minkowski
# ----------------------------

st.subheader("Worldline in Minkowski Space $X_s = (t_s, x_s)$ with $dX_s = V_s ds$")

with st.expander("Component-wise time series for X"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(int(n_paths)):
        ax.plot(tgrid, X_paths[:, k, 0], lw=1.0, label=f"path {k+1}, t(s)")
    ax.set_xlabel("proper time s")
    ax.set_ylabel("t(s)")
    ax.set_title("Temporal coordinate t(s)")
    if n_paths <= 10:
        ax.legend(loc="best", fontsize="small")
    st.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(int(n_paths)):
        for j in range(d):
            ax.plot(tgrid, X_paths[:, k, 1 + j], lw=1.0, label=f"path {k+1}, x[{j}](s)")
    ax.set_xlabel("proper time s")
    ax.set_ylabel("spatial coordinates")
    ax.set_title("Spatial position components x(s)")
    if n_paths * d <= 12:
        ax.legend(loc="upper right", fontsize="small", ncol=2)
    st.pyplot(fig, clear_figure=True)

if d >= 2:
    st.markdown("Spatial trajectory x(s) in first two dimensions.")
    fig, ax = plt.subplots(figsize=(5, 5))
    for k in range(int(n_paths)):
        ax.plot(X_paths[:, k, 1], X_paths[:, k, 2], lw=1.0, label=f"path {k+1}")
        ax.scatter(X_paths[0, k, 1], X_paths[0, k, 2], marker="o", s=30)
        ax.scatter(X_paths[-1, k, 1], X_paths[-1, k, 2], marker="s", s=30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_title("Spatial path (first two dims); circle=start, square=end")
    if n_paths <= 8:
        ax.legend(loc="best", fontsize="small")
    st.pyplot(fig, clear_figure=True)

if d >= 3:
    st.markdown("3D spatial trajectory x(s) in first three dimensions.")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    for k in range(int(n_paths)):
        ax.plot(X_paths[:, k, 1], X_paths[:, k, 2], X_paths[:, k, 3], lw=1.0)
        ax.scatter(X_paths[0, k, 1], X_paths[0, k, 2], X_paths[0, k, 3], marker="o", s=20)
        ax.scatter(X_paths[-1, k, 1], X_paths[-1, k, 2], X_paths[-1, k, 3], marker="s", s=20)
    ax.set_xlabel("x[0]"); ax.set_ylabel("x[1]"); ax.set_zlabel("x[2]")
    ax.set_title("Spatial path (first three dims); circle=start, square=end")
    st.pyplot(fig, clear_figure=True)

# --- Spacetime visualization with transparent light cone (uses first two spatial dims) ---
if show_cone_spacetime and d >= 2:
    st.markdown("Spacetime worldline with future light cone (projection to $(t, x^{(1)}, x^{(2)})$).")
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot worldlines (first two spatial dims)
    for k in range(int(n_paths)):
        t_series = X_paths[:, k, 0]
        x1 = X_paths[:, k, 1]
        x2 = X_paths[:, k, 2]
        ax.plot(t_series, x1, x2, lw=1.5)

    # Build cone from initial event of the first path (you can loop if you want per-path cones)
    k0 = 0
    t0 = X_paths[0, k0, 0]
    x0_1 = X_paths[0, k0, 1]
    x0_2 = X_paths[0, k0, 2]

    # Mesh for cone surface: r = (t - t0), theta in [0, 2pi]
    t_max = float(np.max(X_paths[:, :, 0]))
    T = np.linspace(t0, t_max, 80)
    TH = np.linspace(0.0, 2.0 * np.pi, 120)
    TT, THH = np.meshgrid(T, TH, indexing="ij")
    R = np.maximum(TT - t0, 0.0)  # future light cone only
    Xc = x0_1 + R * np.cos(THH)
    Yc = x0_2 + R * np.sin(THH)

    # Plot the cone as a semi-transparent surface
    ax.plot_surface(TT, Xc, Yc, alpha=0.15, linewidth=0, antialiased=False)

    # Aesthetic tweaks: label axes and set a reasonable view
    ax.set_xlabel("t")
    ax.set_ylabel("x[0]")
    ax.set_zlabel("x[1]")
    ax.set_title("Spacetime trajectory with future light cone from initial event")
    # Optional: align aspect roughly (matplotlib 3D is not metric-accurate)
    # ax.view_init(elev=20, azim=35)

    st.pyplot(fig, clear_figure=True)

elif show_cone_spacetime and d == 1:
    st.markdown("Spacetime worldline with light cone lines in $(t, x^{(1)})$ (since $d=1$).")
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in range(int(n_paths)):
        t_series = X_paths[:, k, 0]
        x1 = X_paths[:, k, 1]
        ax.plot(t_series, x1, lw=1.5, label=f"path {k+1}")
    # Light cone lines from initial event of first path: x - x0 = ±(t - t0)
    k0 = 0
    t0 = X_paths[0, k0, 0]
    x0_1 = X_paths[0, k0, 1]
    t_line = np.linspace(t0, float(np.max(X_paths[:, :, 0])), 200)
    ax.plot(t_line, x0_1 + (t_line - t0), ls="--", alpha=0.6)
    ax.plot(t_line, x0_1 - (t_line - t0), ls="--", alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("x[0]")
    ax.set_title("Spacetime trajectory with light cone lines")
    if n_paths <= 8:
        ax.legend(loc="best", fontsize="small")
    st.pyplot(fig, clear_figure=True)


# ----------------------------
# Optional: Raw tables
# ----------------------------

if show_tables:
    st.subheader("Raw Arrays")
    st.write("U paths shape:", U_paths.shape)
    st.write("V paths shape:", V_paths.shape)
    st.write("X paths shape:", X_paths.shape)
    st.write("U (first path):")
    st.dataframe(U_paths[:, 0, :])
    st.write("V (first path):")
    st.dataframe(V_paths[:, 0, :])
    st.write("X (first path):")
    st.dataframe(X_paths[:, 0, :])

st.caption(
    "Numerics: Euler–Maruyama for U with clamping to keep ||U||<1, mapping φ(U) to 4-velocity, "
    "and forward Euler integration for X. Units use c=1. "
    "Decrease dt (increase N) for higher fidelity, especially near the unit-ball boundary."
)
