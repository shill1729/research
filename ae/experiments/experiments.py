from ae.experiments.manifold_errors import GeometryError
from ae.experiments.sde_errors import DynamicsError
from ae.experiments.training import Trainer
from ae.experiments.path_computations import compute_increments
import numpy as np
eps_grid_size = 10
num_test = 20000
tn = 0.8
h = 0.001
n_paths = 800
n_time = int(np.ceil(tn/h))
print("numper of steps for h =" +str(h)+" is ="+str(n_time))
print("number of paths ="+str(n_paths))
print("mc order ="+str(1/np.sqrt(n_paths)))
print("number of paths 1/h2 sq ="+str(1/h**2))

model_dir = "trained_models/Paraboloid/RiemannianBrownianMotion/"
trainer = Trainer.load_from_pretrained(model_dir)
device = "cpu"
# TODO: put everything below in an its own performance analysis
# Geometry errors
geometry = GeometryError(trainer.toy_data, trainer, 1., device)
geometry.compute_and_plot_errors(eps_grid_size, num_test, None, device)

# TODO: put into analysis method of dynamics_error:
# Dynamics errors
dynamics_error = DynamicsError(trainer.toy_data, trainer, tn)
gt, at, aes, gt_local, aes_local = dynamics_error.sample_path_generator.generate_paths(tn, n_time, n_paths, None)
# View the ambient sample paths
if n_paths < 1000:
    dynamics_error.sample_path_plotter.plot_sample_paths(gt, aes, at, True, "ambient")
# View first step kernel density and the terminal kernel densities:
dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, False)
dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, True)
if n_paths < 1000:
    # View the local sample paths: we expect these not to match in location
    dynamics_error.sample_path_plotter.plot_sample_paths(gt_local, aes_local, None, False, "local")
# First step kernel density of local paths and Terminal density of local paths: we expect these not to match
# dynamics_error.sample_path_plotter.plot_kernel_density(gt_local, aes_local, None, False)
# dynamics_error.sample_path_plotter.plot_kernel_density(gt_local, aes_local, None, True)

# Compute various statistics
results = dynamics_error.analyze_statistical_properties(gt, at, aes, tn)
dynamics_error.sample_path_plotter.run_all_analyses(results)
# TODO we don't want to run feynman kac analysis on increments, just the other stuff
gt_incr, at_incr, aes_incr = compute_increments(gt, at, aes)
results2 = dynamics_error.analyze_statistical_properties(gt_incr, at_incr, aes_incr, tn)
# dynamics_error.sample_path_plotter.run_all_analyses(results2)
# TODO these need different labels for increments, and saves
dynamics_error.sample_path_plotter.plot_deviation_of_means(results2, plot_name="increment")
dynamics_error.sample_path_plotter.plot_covariance_errors(results2, plot_name="increment")