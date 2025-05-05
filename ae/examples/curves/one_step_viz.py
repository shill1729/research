"""
Visualize one step or entire sample paths of Euler-Maruyama for

1. GT: local EM with sample path lift to ambient
2. GT: ambient EM with sample path reconstruction
2. GT: ambient EM with stepwise reconstruction

"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ae.toydata.curves import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data

n_train = 2
train_seed = None
path_seed = None
same_noise = False
use_xy = True # Use (x,y) as the input to the ambient SDE coefficients vs just (x,)

# Manifold parameters: dimensions and boundary
intrinsic_dim = 1
extrinsic_dim = 2
epsilon = 0.1
space_grid_size = 50

# Sample path input
tn = 0.5
ntime = 500
n_ensemble = 1000
h = tn / ntime
time_grid = np.linspace(0, tn, ntime+1)
# ============================================================================
# Generate data
# ============================================================================

# Pick the manifold and dynamics
curve = Parabola()
dynamics = RiemannianBrownianMotion()
manifold = RiemannianManifold(curve.local_coords(), curve.equation())
local_drift = dynamics.drift(manifold)
local_diffusion = dynamics.diffusion(manifold)

# Generate point cloud and process the data
point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
print("Extrinsic drift")
print(point_cloud.extrinsic_drift)
x, _, mu, cov, local_x = point_cloud.generate(n=n_train, seed=train_seed)
x, mu, cov, p, _, _ = process_data(x, mu, cov, d=intrinsic_dim)

# Manually do Euler-Maruyama for all the scenarios:
def extrinsic_drift(x):
    return point_cloud.np_extrinsic_drift(x[0]).reshape(extrinsic_dim)

def extrinsic_diffusion(x):
    return point_cloud.np_extrinsic_diffusion(x[0])

def local_drift(z):
    return point_cloud.np_local_drift(*z).reshape(intrinsic_dim)

def local_diffusion(z):
    return point_cloud.np_local_diffusion(*z)

def extrinsic_drift_xy(p):
    x = p[0]
    y = p[1]
    n1 = -2 * x
    n2 = 1
    return np.array([n1, n2]) / (4*y+1)**2

def extrinsic_diffusion_xy(p):
    x = p[0]
    y = p[1]
    p11 = 1 / (4*y+1)
    p12 = 2 * x / (4*y+1)
    p22 = 4 * y/ (4*y+1)
    return np.array([[p11, p12], [p12, p22]])


# Pick starting point
x0 = x[0, :].detach().numpy()
z0 = local_x[0, :]

local_gt = np.zeros((ntime+1, intrinsic_dim))
local_gt_lifted = np.zeros((ntime+1, extrinsic_dim))
ambient_gt = np.zeros((ntime+1, extrinsic_dim))
ambient_gt_path_recon = np.zeros((ntime+1, extrinsic_dim))
ambient_gt_step_recon = np.zeros((ntime+1, extrinsic_dim))

local_gt[0, :] = z0
local_gt_lifted[0, :] = x0
ambient_gt[0, :] = x0
ambient_gt_path_recon[0, :] = x0
ambient_gt_step_recon[0, :] = x0

rng = np.random.default_rng(path_seed)
em_step = None
for i in range(ntime):
    # Generate GT sample path in local coordinates
    db = rng.normal(scale=np.sqrt(h), size=intrinsic_dim).reshape(intrinsic_dim, 1)
    local_gt[i+1, :] = local_gt[i, :] + h * local_drift(local_gt[i, :]) + (local_diffusion(local_gt[i, :]) @ db).reshape(intrinsic_dim)

    # Generate GT sample path directly in ambient coordinates
    if same_noise and not use_xy:
        dw = db
    else:
        if use_xy:
            dw = rng.normal(scale=np.sqrt(h), size=extrinsic_dim).reshape(extrinsic_dim, 1)
            ambient_gt[i + 1, :] = ambient_gt[i, :] + h * extrinsic_drift_xy(ambient_gt[i, :]) + (
                        extrinsic_diffusion_xy(ambient_gt[i, :]) @ dw).reshape(extrinsic_dim)

            # Generate GT directly in ambient coordinates with step-wise reconstruction pass
            em_step = ambient_gt_step_recon[i, :] + h * extrinsic_drift_xy(ambient_gt_step_recon[i, :]) + (
                    extrinsic_diffusion_xy(ambient_gt_step_recon[i, :]) @ dw).reshape(extrinsic_dim)
        else:
            dw = rng.normal(scale=np.sqrt(h), size=intrinsic_dim).reshape(intrinsic_dim, 1)
            ambient_gt[i + 1, :] = ambient_gt[i, :] + h * extrinsic_drift(ambient_gt[i, :]) + (
                        extrinsic_diffusion(ambient_gt[i, :]) @ dw).reshape(extrinsic_dim)

            # Generate GT directly in ambient coordinates with step-wise reconstruction pass
            em_step = ambient_gt_step_recon[i, :] + h * extrinsic_drift(ambient_gt_step_recon[i, :]) + (
                    extrinsic_diffusion(ambient_gt_step_recon[i, :]) @ dw).reshape(extrinsic_dim)

    ambient_gt_step_recon[i + 1, :] = point_cloud.np_phi(em_step[0]).squeeze()


# Lifting and path-wise reconstruction
for i in range(ntime+1):
    local_gt_lifted[i, :] = point_cloud.np_phi(local_gt[i]).squeeze()
    ambient_gt_path_recon[i, :] = point_cloud.np_phi(ambient_gt[i, 0]).squeeze()

print("Error between local coordinate of step-wise and path recon")
print(np.mean(np.abs(ambient_gt_path_recon[:, 0]-ambient_gt_step_recon[:, 0])))

# Creating the manifold for reference:
curve_grid = np.zeros((space_grid_size, extrinsic_dim))
local_coord_grid = np.linspace(-1, 1, space_grid_size)
for i in range(space_grid_size):
    curve_grid[i] = point_cloud.np_phi(local_coord_grid[i]).squeeze()

print("Initial point x_0:")
print(local_gt_lifted[0, :])
print(ambient_gt[0, :])
print(ambient_gt_path_recon[0, :])
print(ambient_gt_step_recon[0, :])

fig = plt.figure()
# ax = plt.subplot(121)
# ax.plot(time_grid, local_gt, label="Local EM")
# ax.legend()
ax = plt.subplot(111)
ax.plot(local_coord_grid, curve_grid[:, 1], label="manifold")
ax.plot(local_gt_lifted[:, 0], local_gt_lifted[:, 1], label="Local EM, Lifted", alpha = 0.5)
ax.plot(ambient_gt[:, 0], ambient_gt[:, 1], label="Ambient EM", alpha=0.5)
ax.plot(ambient_gt_path_recon[:, 0], ambient_gt_path_recon[:, 1], label="Ambient EM, post-recon", alpha=0.5)
ax.plot(ambient_gt_step_recon[:, 0], ambient_gt_step_recon[:, 1], label="Ambient EM, step-recon", alpha=0.5)

# End points:
# Add scatter for initial and final points
ax.scatter(local_gt_lifted[0, 0], local_gt_lifted[0, 1],
           color='black', marker='o', label="$x_0$", alpha=0.9)
ax.scatter(local_gt_lifted[-1, 0], local_gt_lifted[-1, 1],
           color='red', marker='o', label="$x_h$", alpha=0.9)

# Ambient EM
ax.scatter(ambient_gt[0, 0], ambient_gt[0, 1],
           marker='x', color='green', label=r"$x_0$, Ambient", alpha=0.5)
ax.scatter(ambient_gt[-1, 0], ambient_gt[-1, 1],
           marker='x', color='green', label=r"$x_h$, Ambient", alpha=0.5)

# Post-reconstruction
ax.scatter(ambient_gt_path_recon[0, 0], ambient_gt_path_recon[0, 1],
           marker='v', color='purple', label=r"$x_0$, Post-recon", alpha=0.5)
ax.scatter(ambient_gt_path_recon[-1, 0], ambient_gt_path_recon[-1, 1],
           marker='v', color='purple', label=r"$x_h$, Post-recon", alpha=0.5)

# Stepwise reconstruction
ax.scatter(ambient_gt_step_recon[0, 0], ambient_gt_step_recon[0, 1],
           marker='^', color='orange', label=r"$x_0$, Step-recon", alpha=0.5)
ax.scatter(ambient_gt_step_recon[-1, 0], ambient_gt_step_recon[-1, 1],
           marker='^', color='orange', label=r"$x_h$, Step-recon", alpha=0.5)

ax.legend()
plt.show()

# Distribution analysis:
# ============================================================================
# Ensemble Simulation for Terminal Point Distributions
# ============================================================================
terminal_points = {
    "Lifted Local": [],
    "Ambient": [],
    "Post-recon": [],
    "Step-recon": [],
}

for j in range(n_ensemble):
    # Initialize paths
    local_path = np.zeros((ntime + 1, intrinsic_dim))
    local_path[0, :] = z0

    local_lifted_path = np.zeros((ntime + 1, extrinsic_dim))
    local_lifted_path[0, :] = x0

    ambient_path = np.zeros((ntime + 1, extrinsic_dim))
    ambient_path[0, :] = x0

    path_recon = np.zeros((ntime + 1, extrinsic_dim))
    path_recon[0, :] = x0

    step_recon = np.zeros((ntime + 1, extrinsic_dim))
    step_recon[0, :] = x0
    for i in range(ntime):
        # Simulate with new randomness
        db = np.random.normal(scale=np.sqrt(h), size=(intrinsic_dim, 1))
        if same_noise:
            dw = db
        else:
            if use_xy:
                dw = rng.normal(scale=np.sqrt(h), size=extrinsic_dim).reshape(extrinsic_dim, 1)
                ambient_path[i + 1, :] = ambient_path[i, :] + h * extrinsic_drift_xy(ambient_path[i, :]) + (
                            extrinsic_diffusion_xy(ambient_path[i, :]) @ dw).reshape(extrinsic_dim)
                em_step = step_recon[i, :] + h * extrinsic_drift_xy(step_recon[i, :]) + (
                            extrinsic_diffusion_xy(step_recon[i, :]) @ dw).reshape(extrinsic_dim)
            else:
                dw = rng.normal(scale=np.sqrt(h), size=intrinsic_dim).reshape(intrinsic_dim, 1)
                ambient_path[i + 1, :] = ambient_path[i, :] + h * extrinsic_drift(ambient_path[i, :]) + (
                            extrinsic_diffusion(ambient_path[i, :]) @ dw).reshape(extrinsic_dim)
                em_step = step_recon[i, :] + h * extrinsic_drift(step_recon[i, :]) + (
                            extrinsic_diffusion(step_recon[i, :]) @ dw).reshape(extrinsic_dim)

        local_path[i+1, :] = local_path[i, :] + h * local_drift(local_path[i, :]) + (local_diffusion(local_path[i, :]) @ db).reshape(intrinsic_dim)
        step_recon[i+1, :] = point_cloud.np_phi(em_step[0]).squeeze()

    for i in range(ntime+1):
        # Lift local path
        local_lifted_path[i, :] = point_cloud.np_phi(local_path[i, :]).squeeze()
        # Reconstruct ambient path
        path_recon[i, :] = point_cloud.np_phi(ambient_path[i, 0]).squeeze()

    # Store terminal points
    terminal_points["Lifted Local"].append(local_lifted_path[-1, :])
    terminal_points["Ambient"].append(ambient_path[-1, :])
    terminal_points["Post-recon"].append(path_recon[-1, :])
    terminal_points["Step-recon"].append(step_recon[-1, :])

# Stack results into arrays
for key in terminal_points:
    terminal_points[key] = np.stack(terminal_points[key], axis=0)  # shape: (n_ensemble, extrinsic_dim)



# ============================================================================
# Plotting terminal point distributions
# ============================================================================
fig, axes = plt.subplots(1, extrinsic_dim, figsize=(12, 4))
colors = {
    "Lifted Local": 'black',
    "Ambient": 'green',
    "Post-recon": 'purple',
    "Step-recon": 'orange',
}
linestyles = {
    "Lifted Local": '-',
    "Ambient": '--',
    "Post-recon": '-.',
    "Step-recon": ':',
}

for i in range(extrinsic_dim):
    ax = axes[i]
    for method in terminal_points:
        sns.kdeplot(
            terminal_points[method][:, i],
            ax=ax,
            label=fr"$x_h^{{({i+1})}}$, {method}",
            color=colors[method],
            linestyle=linestyles[method],
            fill=False
        )
    ax.set_title(f"Distribution of $x_h^{i+1}$")
    ax.legend()

plt.tight_layout()
plt.show()

#================================================================================================
# Feynman-Kac statistics
#================================================================================================
# Define named test functions
test_functions_named = {
    r"$x_1$": lambda x: x[:, 0],
    r"$x_2$": lambda x: x[:, 1],
    r"$x_1 + x_2$": lambda x: np.sum(x, axis=1),
    r"$x_1 * x_2$": lambda x: np.prod(x, axis=1),
    r"$sin(x_1 x_2)$": lambda x: np.sin(5 * np.prod(x, axis=1)),
    r"$|x|$": lambda x: np.linalg.vector_norm(x, axis=1),
    r"$|x_2-f(x_1)|$": lambda x: np.abs(x[:, 1]-point_cloud.np_phi(x[:, 0])[1])
}

# Compute FK statistics
fk_stats_named = {
    fname: {
        method: np.mean(f(terminal_points[method]))
        for method in terminal_points
    }
    for fname, f in test_functions_named.items()
}

# Display as DataFrame
df_fk = pd.DataFrame(fk_stats_named).T  # transpose to have functions as rows
df_fk.index.name = "Test Function"
df_fk.columns.name = "Method"
print(df_fk.round(13))