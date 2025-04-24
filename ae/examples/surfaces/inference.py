from ae.experiments import GeometryError, DynamicsError
from ae.experiments import Trainer
from ae.experiments.samplepaths.path_computations import compute_increments
from ae.experiments.training.helpers import get_time_horizon_name, print_dict
# Comment out this import on colab, provided you run a code-cell of the code from train.py
from train import large_dim, embed, embedding_seed, eps_max, device
import numpy as np
import os
print("Current working directory of inference.py")
print(os.getcwd())

# TODO: Refactor this to save the mean's of the ensembles of functionals
# Settings that remain constant
show_geo = True
show_stats = True
# Orthogonally project the increment for the ambient model?
project = False
eps_grid_size = 10
num_test = 20000
h = 0.001
n_paths = 1000
# Define a list of time horizons to test
time_horizons = [0.1]


# Load the pre-trained model: note working directory is currently ae/experiments
model_dir = "examples/surfaces/trained_models/Paraboloid/RiemannianBrownianMotion/trained_20250423-005325_h[16]_df[16]_dr[16]_lr0.001_epochs9000_not_annealed"
trainer = Trainer.load_from_pretrained(model_dir, device=device, large_dim=large_dim)
trainer.models = {name:model.to(device) for name, model in trainer.models.items()}
trainer.ambient_drift.to(device)
trainer.ambient_diffusion.to(device)
trainer.device = device

# Run geometry error once
print(trainer.toy_data.large_dim)
trainer.toy_data.embedding_seed = embedding_seed
geometry = GeometryError(trainer.toy_data, trainer, eps_max, device, show=show_geo, embed=embed)
# geometry.compute_and_plot_errors(eps_grid_size, num_test, None)
# geometry.plot_int_bd_surface(epsilon=eps_max)

# Loop over each time horizon and run dynamics error analysis
for tn in time_horizons:
    time_category = get_time_horizon_name(tn)
    n_time = int(np.ceil(tn / h))
    print("\n======================================")
    print("Time horizon:", tn, "Category:", time_category)
    print("Number of steps for h =", h, "is =", n_time)
    print("Number of paths =", n_paths)
    print("MC order =", 1 / np.sqrt(n_paths))
    print("Number of paths 1/h^2 =", 1 / h ** 2)

    # Dynamics errors
    dynamics_error = DynamicsError(trainer.toy_data, trainer, tn, show=show_stats, project=project)
    gt, at, aes, gt_local, aes_local = dynamics_error.sample_path_generator.generate_paths(tn, n_time, n_paths, None, embed, large_dim)

    # Plot ambient sample paths (only if the number of paths is small enough)
    # if n_paths < 5000:
    dynamics_error.sample_path_plotter.plot_sample_paths(gt, aes, at, True, "ambient")

    # Plot kernel density estimates for both first-step and terminal transition
    # kl_1step = dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, False)
    # kl_term = dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, True)

    if n_paths < 1000:
        # Plot local sample paths
        dynamics_error.sample_path_plotter.plot_sample_paths(gt_local, aes_local, None, False, "local")

    # Compute various statistics on the generated paths
    results = dynamics_error.analyze_statistical_properties(gt, at, aes, tn)
    dynamics_error.sample_path_plotter.run_all_analyses(results)

    # Compute increments and perform similar analyses
    gt_incr, at_incr, aes_incr = compute_increments(gt, at, aes)
    results2 = dynamics_error.analyze_statistical_properties(gt_incr, at_incr, aes_incr, tn)
    dynamics_error.sample_path_plotter.plot_deviation_of_means(results2, plot_name="increment")
    dynamics_error.sample_path_plotter.plot_covariance_errors(results2, plot_name="increment")

    # print("1st step PDF")
    # print_dict(kl_1step)
    # print("\nTerminal PDF")
    # print_dict(kl_term)

# from ae.experiments.manifold_errors import GeometryError
# from ae.experiments.sde_errors import DynamicsError
# from ae.experiments.training import Trainer
# from ae.experiments.path_computations import compute_increments
# import numpy as np
# eps_grid_size = 10
# num_test = 20000
# tn = 0.8
# h = 0.0005
# n_paths = 800
# n_time = int(np.ceil(tn/h))
# print("number of steps for h =" +str(h)+" is ="+str(n_time))
# print("number of paths ="+str(n_paths))
# print("mc order ="+str(1/np.sqrt(n_paths)))
# print("number of paths 1/h2 sq ="+str(1/h**2))
#
# model_dir = "trained_models/Paraboloid/RiemannianBrownianMotion/trained_20250308-152655_h[16]_df[2]_dr[2]_lr0.001_epochs5000_not_annealed"
# trainer = Trainer.load_from_pretrained(model_dir)
# device = "cpu"
# # TODO: put everything below in an its own performance analysis
# # Geometry errors
# geometry = GeometryError(trainer.toy_data, trainer, 1., device)
# geometry.compute_and_plot_errors(eps_grid_size, num_test, None, device)
#
# # TODO: put into analysis method of dynamics_error:
# # Dynamics errors
# dynamics_error = DynamicsError(trainer.toy_data, trainer, tn)
# gt, at, aes, gt_local, aes_local = dynamics_error.sample_path_generator.generate_paths(tn, n_time, n_paths, None)
# # View the ambient sample paths
# if n_paths < 1000:
#     dynamics_error.sample_path_plotter.plot_sample_paths(gt, aes, at, True, "ambient")
# # View first step kernel density and the terminal kernel densities:
# dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, False)
# dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, True)
# if n_paths < 1000:
#     # View the local sample paths: we expect these not to match in location
#     dynamics_error.sample_path_plotter.plot_sample_paths(gt_local, aes_local, None, False, "local")
# # First step kernel density of local paths and Terminal density of local paths: we expect these not to match
# # dynamics_error.sample_path_plotter.plot_kernel_density(gt_local, aes_local, None, False)
# # dynamics_error.sample_path_plotter.plot_kernel_density(gt_local, aes_local, None, True)
#
# # Compute various statistics
# results = dynamics_error.analyze_statistical_properties(gt, at, aes, tn)
# dynamics_error.sample_path_plotter.run_all_analyses(results)
# # TODO we don't want to run feynman kac analysis on increments, just the other stuff
# gt_incr, at_incr, aes_incr = compute_increments(gt, at, aes)
# results2 = dynamics_error.analyze_statistical_properties(gt_incr, at_incr, aes_incr, tn)
# # dynamics_error.sample_path_plotter.run_all_analyses(results2)
# # TODO these need different labels for increments, and saves
# dynamics_error.sample_path_plotter.plot_deviation_of_means(results2, plot_name="increment")
# dynamics_error.sample_path_plotter.plot_covariance_errors(results2, plot_name="increment")