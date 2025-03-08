from ae.experiments.datagen import ToyData
from ae.experiments.training import Trainer
from ae.experiments.manifold_errors import GeometryError
from ae.experiments.sde_errors import DynamicsError
from ae.toydata.surfaces import *
from ae.toydata.local_dynamics import *

import torch

# TODO: 1. add model loading from the correct path.
#  2. what sample path metrics do we care about and want to plot and save?
#  3. use part 2. to finish cleaning up, organizing and implementing path_plotting.py and sde_errors.py

device = torch.device("cpu")
train_seed = None
test_seed = None
norm = "fro"
# torch.manual_seed(train_seed)
# Point cloud parameters
num_points = 30
num_test = 20000
batch_size = 20
eps_max = 1.
eps_grid_size = 10
# The intrinsic and extrinsic dimensions.
extrinsic_dim, intrinsic_dim = 3, 2
hidden_dims = [16]
diffusion_layers = [16]
drift_layers = [16]
lr = 0.001
weight_decay = 0.
epochs_ae = 100
epochs_diffusion = 100
epochs_drift = 100
print_freq = 500
# Diffeo weight for accumulative orders
diffeo_weight_12 = 0.01  # this is the separate diffeo_weight for just the First order and second order
# First order weight: 0.08 was good
tangent_angle_weight = 0.01
# Second order weights accumulative
tangent_angle_weight2 = 0.01  # the first order weight for the second order model, if accumulating penalties
tangent_drift_weight = 0.001
# diffeo weight alone
# diffeo_weight = 0.2  # I think making this higher helps contract but worsens the second order
npaths = 30
ntime = 1000
tn = 0.5


# Main below
params = {
        "num_points": num_points,
        "num_test": num_test,
        "batch_size": batch_size,
        "eps_max": eps_max,
        "eps_grid_size": eps_grid_size,
        "extrinsic_dim": extrinsic_dim,
        "intrinsic_dim": intrinsic_dim,
        "hidden_dims": hidden_dims,
        "diffusion_layers": diffusion_layers,
        "drift_layers": drift_layers,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_ae": epochs_ae,
        "epochs_diffusion": epochs_diffusion,
        "epochs_drift": epochs_drift,
        "print_freq": print_freq,
        "tangent_angle_weight": tangent_angle_weight,
        "tangent_angle_weight2": tangent_angle_weight2,
        "tangent_drift_weight": tangent_drift_weight,
        "diffeo_weight": diffeo_weight_12,
        "norm": norm
    }

# anneal_weights = { "tangent_drift_weight": lambda epoch: tangent_drift_weight * (epoch / epochs_ae) if epoch >
# np.round(epochs_ae/5) else 0.}
anneal_weights = None

# anneal_weights = None
anneal_tag = "annealed_2nd" if anneal_weights is not None else "not_annealed"
surface = Paraboloid(5., 5.)
dynamics = RiemannianBrownianMotion()
toydata = ToyData(surface, dynamics)
trainer = Trainer(toydata, params, device, anneal_tag)
trainer.train(anneal_weights)
trainer.save_models()
# TODO : stop here and write loader

geometry = GeometryError(toydata, trainer, eps_max, device)
geometry.compute_and_plot_errors(eps_grid_size, num_test, None, device)
dynamics_error = DynamicsError(toydata, trainer)
# TODO put into analysis method of dynamics_error:
gt, at, aes, gt_local, aes_local = dynamics_error.sample_path_generator.generate_paths(tn, ntime, npaths, None)
# View the ambient sample paths
dynamics_error.sample_path_plotter.plot_ambient_sample_paths(gt, aes, at)
# View first step kernel density:
dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, False)
# View the terminal kernel densities:
dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, True)
dynamics_error.sample_path_plotter.plot_local_sample_paths(gt_local, aes_local)
dynamics_error.sample_path_plotter.plot_kernel_density(gt_local, aes_local, None, False)
dynamics_error.sample_path_plotter.plot_kernel_density(gt_local, aes_local, None, True)
# dynamics_error.run_all_analyses(tn, npaths, None, trainer.exp_dir)
