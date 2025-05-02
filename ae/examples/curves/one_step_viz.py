"""
Visualize one step or entire sample paths of Euler-Maruyama for

1. GT: local EM with sample path lift to ambient
2. GT: ambient EM with sample path reconstruction
2. GT: ambient EM with stepwise reconstruction

"""
import matplotlib.pyplot as plt

from ae.toydata.curves import *
from ae.toydata.local_dynamics import *
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data

n_train = 3
train_seed = 17
path_seed = 17

# Manifold parameters: dimensions and boundary
intrinsic_dim = 1
extrinsic_dim = 2
epsilon = 0.1
space_grid_size = 30

# Sample path input
tn = 0.02
ntime = 1
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
for i in range(ntime):
    # Generate GT sample path in local coordinates
    db = rng.normal(scale=np.sqrt(h), size=intrinsic_dim).reshape(intrinsic_dim, 1)
    local_gt[i+1, :] = local_gt[i, :] + h * local_drift(local_gt[i, :]) + (local_diffusion(local_gt[i, :]) @ db).reshape(intrinsic_dim)

    # Generate GT sample path directly in ambient coordinates
    # dw = rng.normal(scale=np.sqrt(h), size=intrinsic_dim).reshape(intrinsic_dim, 1)
    ambient_gt[i+1, :] = ambient_gt[i, :] + h * extrinsic_drift(ambient_gt[i, :]) + (extrinsic_diffusion(ambient_gt[i, :]) @ db).reshape(extrinsic_dim)

    # Generate GT directly in ambient coordinates with step-wise reconstruction pass
    em_step = ambient_gt_step_recon[i, :] + h * extrinsic_drift(ambient_gt_step_recon[i, :]) + (
                extrinsic_diffusion(ambient_gt_step_recon[i, :]) @ db).reshape(extrinsic_dim)
    ambient_gt_step_recon[i + 1, :] = point_cloud.np_phi(em_step[0]).squeeze()


# Lifting and path-wise reconstruction
for i in range(ntime):
    local_gt_lifted[i+1, :] = point_cloud.np_phi(local_gt[i+1]).squeeze()
    ambient_gt_path_recon[i + 1, :] = point_cloud.np_phi(ambient_gt[i+1, 0]).squeeze()


# Creating the manifold for reference:
curve_grid = np.zeros((space_grid_size, extrinsic_dim))
local_coord_grid = np.linspace(-1, 1, space_grid_size)
for i in range(space_grid_size):
    curve_grid[i] = point_cloud.np_phi(local_coord_grid[i]).squeeze()

print(local_gt_lifted[0, :])
print(ambient_gt[0, :])
print(ambient_gt_path_recon[0, :])
print(ambient_gt_step_recon[0, :])

fig = plt.figure()
ax = plt.subplot(121)
ax.plot(time_grid, local_gt, label="Local EM")
ax.legend()
ax = plt.subplot(122)
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



