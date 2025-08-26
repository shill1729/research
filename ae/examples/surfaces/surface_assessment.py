from ae.experiment_classes import GeometryError, DynamicsError
from ae.experiment_classes import Trainer
from ae.experiment_classes.samplepaths.path_computations import compute_increments
from ae.experiment_classes.training.helpers import get_time_horizon_name, print_dict
# Comment out this import on colab, provided you run a code-cell of the code from surface_training.py
from surface_training import large_dim, embed, embedding_seed, device
import numpy as np
import os
print("Current working directory of surface_assessment.py")
print(os.getcwd())

# Settings that remain constant
show_geo = True
show_stats = True
# Orthogonally project the increment for the ambient model?
project = False
eps_grid_size = 20
eps_max = 1
num_test = 20000

# Sample path properties
h = 0.001
n_paths = 250
# Define a list of time horizons to test
time_horizons = [0.5]


# Load the pre-trained model: note working directory is currently ae/experiment_classes
model_dir = "examples/surfaces/trained_models/ProductSurface/RiemannianBrownianMotion/trained_20250820-170956_h[32, 32]_df[16, 16]_dr[16, 16]_lr0.001_epochs9000_not_annealed"
trainer = Trainer.load_from_pretrained(model_dir, device=device, large_dim=large_dim)
trainer.models = {name:model.to(device) for name, model in trainer.models.items()}
trainer.ambient_drift.to(device)
trainer.ambient_diffusion.to(device)
trainer.device = device

# Run geometry error once
print(trainer.toy_data.large_dim)
trainer.toy_data.embedding_seed = embedding_seed
geometry = GeometryError(trainer.toy_data, trainer, eps_max, device, show=show_geo, embed=embed)
geometry.compute_and_plot_errors(eps_grid_size, num_test, None)
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
    if n_paths < 1000:
        dynamics_error.sample_path_plotter.plot_sample_paths(gt, aes, at, True, "ambient")

    # Plot kernel density estimates for both first-step and terminal transition
    kl_1step = dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at.detach(), False)
    kl_term = dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at.detach(), True)

    if n_paths < 1000:
        # Plot local sample paths
        dynamics_error.sample_path_plotter.plot_sample_paths(gt_local, aes_local, None, False, "local")

    # Compute various statistics on the generated paths
    results = dynamics_error.analyze_statistical_properties(gt, at, aes, tn)
    dynamics_error.sample_path_plotter.run_all_analyses(results)

    # Compute increments and perform similar analyses
    gt_incr, at_incr, aes_incr = compute_increments(gt, at, aes)
    results2 = dynamics_error.analyze_statistical_properties(gt_incr, at_incr, aes_incr, tn)
    # dynamics_error.sample_path_plotter.plot_deviation_of_means(results2, plot_name="increment")
    # dynamics_error.sample_path_plotter.plot_covariance_errors(results2, plot_name="increment")

    print("1st step PDF")
    print_dict(kl_1step)
    print("\nTerminal PDF")
    print_dict(kl_term)

