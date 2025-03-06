import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from ae.experiments.latent_vs_ambient_dynamics import cloud_train
from ae.sdes import SDE
from ae.toydata.surfaces import *
from ae.experiments.helpers import save_plot
from load_models import (ae_diffusion_vanilla, ae_diffusion_first,
                         ae_diffusion_second, ae_diffusion_diffeo,
                         ambient_drift, ambient_diffusion, save_dir)

seed = None
npaths = 250  # Number of sample paths per model
# Run 0.05, 0.5, 1, and 5
T_max = 0.01
# short term vs long term
time_horizon = ""
if T_max <= 0.01:
    time_horizon = "/very_short_term/"
    ntime = 200  # Number of time steps per path
elif 0.01 < T_max <= 0.05:
    time_horizon = "/short_term/"
    ntime = 500  # Number of time steps per path
elif 0.05 < T_max <= 0.8:
    time_horizon = "/medium_term/"
    ntime = 1000  # Number of time steps per path
elif 0.8 < T_max <= 5.:
    time_horizon = "/long_term/"
    ntime = 2000  # Number of time steps per path
elif T_max >= 5.:
    ntime = 5500  # Number of time steps per path
    time_horizon = "/very_long_term/"
else:
    raise ValueError("T_max not within conditions")

time_grid = np.linspace(0, T_max, ntime + 1)
# Precompute test sample x0
x0 = cloud_train.generate(1, seed=seed)[0]  # numpy (D,)
x0_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)  # torch (1,D)

ae_models = {
    "Vanilla": ae_diffusion_vanilla,
    "First Order": ae_diffusion_first,
    "Second Order": ae_diffusion_second
    # "Diffeo": ae_diffusion_diffeo
}


def chart_error(x):
    p = np.abs(cloud_train.np_phi(*[x[0], x[1]])[2] - x[2])
    return p


# ("Norm", f_norm), ("First Coordinate", f_first_coordinate),
f_functions = [("Norm-Sq", lambda x: np.linalg.norm(x)**2),
               ("Sum of coordinates", lambda x: np.sum(x)),
               ("1st", lambda x: x[0]),
               ("2nd", lambda x: x[1]),
               ("3rd", lambda x: x[2]),
               ("Manifold constr", lambda x: chart_error(x))]


def compute_conditional_expectation(ensemble_at_t, f):
    return np.mean([f(x) for x in ensemble_at_t])


def compute_time_series(ensemble, f):
    ntime = ensemble.shape[1]
    time_series = np.zeros(ntime)
    for t in range(ntime):
        time_series[t] = compute_conditional_expectation(ensemble[:, t, :], f)
    return time_series


def compute_confidence_intervals(ensemble, f):
    """
    Compute mean and confidence intervals (standard error of the mean) over ensemble paths.
    """
    ntime = ensemble.shape[1]
    means = np.zeros(ntime)
    std_errors = np.zeros(ntime)

    for t in range(ntime):
        values = np.array([f(x) for x in ensemble[:, t, :]])
        means[t] = np.mean(values)
        std_errors[t] = np.std(values) / np.sqrt(len(values))  # Standard error of the mean

    return means, std_errors


# -------------------------------
# Precompute latent encoding for each AE model (since x0 is constant)
# -------------------------------
def get_z0(model, x0_torch, name):
    z0_tensor = model.autoencoder.encoder(x0_torch)
    x0_hat = model.autoencoder.decoder(z0_tensor).detach().numpy().squeeze(0)

    x0_numpy = x0_torch.squeeze(0).detach().numpy()
    z0_numpy = z0_tensor.detach().numpy().squeeze(0)
    print("\n " + str(name))
    print("l1 Recon Error for x0 = " + str(np.linalg.vector_norm(x0_hat - x0_numpy, ord=1)))
    print("l2 Recon Error for x0 = " + str(np.linalg.vector_norm(x0_hat - x0_numpy, ord=2)))
    return z0_numpy


# -------------------------------
# Define path generation functions
# -------------------------------
def generate_paths_ae(z0, model, T, npaths, ntime):
    latent_paths = model.latent_sde.sample_paths(z0, T, ntime, npaths)
    ambient_paths = np.zeros((npaths, ntime + 1, 3))
    for j in range(npaths):
        ambient_paths[j, :, :] = model.autoencoder.decoder(
            torch.tensor(latent_paths[j, :, :], dtype=torch.float32)).detach().numpy()
    return ambient_paths


# GROUND TRUTH
def generate_paths_ground_truth(x0, T, npaths, ntime):
    z0_true = x0[:2]
    ambient_paths = np.zeros((npaths, ntime + 1, 3))
    latent_paths = cloud_train.latent_sde.sample_ensemble(z0_true, T, ntime, npaths)
    # Assuming cloud_train.np_phi supports vectorized operations is ideal.
    # Here we use a list comprehension if not vectorized.
    for j in range(npaths):
        for i in range(ntime + 1):
            ambient_paths[j, i, :] = np.squeeze(cloud_train.np_phi(*latent_paths[j, i, :]))
    return ambient_paths


# AMBIENT MODEL
def generate_paths_ambient(x0, T, npaths, ntime, ambient_sde):
    return ambient_sde.sample_ensemble(x0, T, ntime, npaths, noise_dim=3)


# Compute conditional expectations for a given T by slicing the ensemble:
def get_state_at_time(full_paths, T, T_max, ntime):
    idx = int((T / T_max) * (ntime - 1))
    return full_paths[:, idx, :]


z0_dict = {name: get_z0(model, x0_torch, name) for name, model in ae_models.items()}

# Initialize a results dictionary for time series
results_time = {fname: {"Ground Truth": None, "Vanilla": None, "First Order": None,
                        "Second Order": None, "Diffeo": None, "Ambient": None}
                for fname, _ in f_functions}

# -------------------------------
# Precompute ambient SDE once
# -------------------------------
ambient_sde = SDE(ambient_drift.drift_numpy, ambient_diffusion.diffusion_numpy)
# Similarly for ground truth and ambient, if possible:
paths_ground_truth_full = generate_paths_ground_truth(x0, T_max, npaths, ntime)
paths_ambient_full = generate_paths_ambient(x0, T_max, npaths, ntime, ambient_sde)
paths_ae_full = {name: generate_paths_ae(z0_dict[name], model, T_max, npaths, ntime)
                 for name, model in ae_models.items()}
results_conf_intervals = {}

for fname, f in f_functions:
    results_time[fname] = {}
    results_conf_intervals[fname] = {}

    results_time[fname]["Ground Truth"], results_conf_intervals[fname]["Ground Truth"] = compute_confidence_intervals(
        paths_ground_truth_full, f)

    for name in ae_models.keys():
        results_time[fname][name], results_conf_intervals[fname][name] = compute_confidence_intervals(
            paths_ae_full[name], f)

    results_time[fname]["Ambient"], results_conf_intervals[fname]["Ambient"] = compute_confidence_intervals(
        paths_ambient_full, f)


# TODO: clean up this plotting
def plot_time_series(results_time, results_conf_intervals, time_grid):
    for fname, model_data in results_time.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        for model_name, values in model_data.items():
            conf_intervals = results_conf_intervals[fname][model_name]
            ax.plot(time_grid, values, label=model_name)
            ax.fill_between(time_grid, values - conf_intervals, values + conf_intervals, alpha=0.2)

        ax.set_xlabel("Time")
        ax.set_ylabel(f"E[{fname}(X_t) | X_0]")
        ax.set_title(f"{fname}")
        ax.legend()
        ax.grid(True)

        plt.show()
        fk_stats_folder = save_dir + time_horizon
        os.makedirs(fk_stats_folder, exist_ok=True)
        save_plot(fig, save_dir + time_horizon, f"mean_path_conf_{fname}")


def plot_time_series_mean_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, relative=False):
    """
    Plots the mean error \|m_t - \hat{m}_t\| for each model relative to the ground truth.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute mean over ensembles for each time step
    mean_gt_path = paths_ground_truth_full.mean(axis=0)  # Shape: (ntime, 3)
    mean_ambient_path = paths_ambient_full.mean(axis=0)  # Shape: (ntime, 3)
    mean_ae_paths = {name: paths.mean(axis=0) for name, paths in paths_ae_full.items()}
    model_means = {"Ground Truth": mean_gt_path, **mean_ae_paths, "Ambient": mean_ambient_path}
    mean_gt_norm = np.linalg.vector_norm(mean_gt_path, axis=1, ord=2, keepdims=False)
    for model_name, mean_path in model_means.items():
        abs_errors = np.linalg.vector_norm(mean_gt_path - mean_path, axis=1, ord=2, keepdims=False)
        if relative:
            errors = abs_errors / mean_gt_norm
        else:
            errors = abs_errors
        ax.plot(time_grid, errors, label=model_name)
    error_type = "Relative error of " if relative else "Absolute error of"
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Error: ")
    ax.set_title(error_type+" $\\|E(X_t)-E(\hat{X}_t)\\|_2$")
    ax.legend()
    ax.grid(True)
    plt.show()

    fk_stats_folder = save_dir + time_horizon
    os.makedirs(fk_stats_folder, exist_ok=True)
    save_plot(fig, save_dir + time_horizon, "mean_vector_errors")




def plot_time_series_mean_deviation_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full):
    """
    Plots the mean Euclidean deviation \|X_t - \hat{X}_t\| for each model relative to the ground truth.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute mean Euclidean deviation over ensembles for each time step
    # mean_deviation_gt = np.linalg.norm(paths_ground_truth_full - paths_ground_truth_full.mean(axis=0), axis=2)
    mean_deviation_ambient = np.linalg.vector_norm(paths_ground_truth_full - paths_ambient_full, ord=2, axis=2)
    mean_deviation_ae = {name: np.linalg.vector_norm(paths_ground_truth_full - paths, ord=2, axis=2) for name, paths in
                         paths_ae_full.items()}

    model_deviation_means = {
                             # "Ground Truth": mean_deviation_gt.mean(axis=0),
                             "Ambient": mean_deviation_ambient.mean(axis=0),
                             **{name: dev.mean(axis=0) for name, dev in mean_deviation_ae.items()}}

    # Plot mean Euclidean deviations
    for model_name, deviations in model_deviation_means.items():
        ax.plot(time_grid, deviations, label=model_name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Mean of Euclidean deviation: ")
    ax.set_title(" $E(\\|X_t-\hat{X}_t\\|_2)$")
    ax.legend()
    ax.grid(True)
    plt.show()

    # Save the plot
    fk_stats_folder = save_dir + time_horizon
    os.makedirs(fk_stats_folder, exist_ok=True)
    save_plot(fig, save_dir + time_horizon, "mean_vector_deviation_errors")


def plot_time_series_variance_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full):
    """
    Plots the variance of the Euclidean deviation \|X_t - \hat{X}_t\| for each model relative to the ground truth.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute mean Euclidean deviation over ensembles for each time step
    # mean_deviation_gt = np.linalg.norm(paths_ground_truth_full - paths_ground_truth_full.mean(axis=0), axis=2)
    mean_deviation_ambient = np.linalg.vector_norm(paths_ground_truth_full - paths_ambient_full, ord=2, axis=2)
    mean_deviation_ae = {name: np.linalg.vector_norm(paths_ground_truth_full - paths, ord=2, axis=2) for name, paths in
                         paths_ae_full.items()}

    model_deviation_means = {
                             # "Ground Truth": mean_deviation_gt.mean(axis=0),
                             "Ambient": mean_deviation_ambient.var(axis=0),
                             **{name: dev.var(axis=0) for name, dev in mean_deviation_ae.items()}}

    # Plot mean Euclidean deviations
    for model_name, deviations in model_deviation_means.items():
        ax.plot(time_grid, deviations, label=model_name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Variance of Euclidean deviation: ")
    ax.set_title(" $Var(\\|X_t-\hat{X}_t\\|_2)$")
    ax.legend()
    ax.grid(True)
    plt.show()

    # Save the plot
    fk_stats_folder = save_dir + time_horizon
    os.makedirs(fk_stats_folder, exist_ok=True)
    save_plot(fig, save_dir + time_horizon, "variance_deviation_errors")


def compute_covariance(paths):
    """
    Compute the covariance matrix for each time step across the ensemble.

    paths: np.array of shape (n_ensemble, n_time, n_dim)

    Returns:
        covariances: np.array of shape (n_time, n_dim, n_dim)
    """
    n_time, n_dim = paths.shape[1], paths.shape[2]
    covariances = np.zeros((n_time, n_dim, n_dim))

    for t in range(n_time):
        covariances[t] = np.cov(paths[:, t, :].T, bias=True)  # bias=True for MLE-like normalization

    return covariances


def plot_time_series_cov_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, relative=False,
                                norm=None):
    """
    Plots the mean error \|Cov(X_t) - Cov(\hat{X}_t)\|_F^2 for each model relative to the ground truth.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute covariance matrices over ensembles for each time step
    cov_gt = compute_covariance(paths_ground_truth_full)  # Shape: (ntime, n_dim, n_dim)
    cov_ambient = compute_covariance(paths_ambient_full)  # Shape: (ntime, n_dim, n_dim)
    cov_ae_models = {name: compute_covariance(paths) for name, paths in paths_ae_full.items()}
    cov_gt_norm = np.linalg.matrix_norm(cov_gt, ord="fro")
    model_covariances = {"Ground Truth": cov_gt, **cov_ae_models, "Ambient": cov_ambient}

    for model_name, cov_path in model_covariances.items():
        absolute_errors = np.linalg.matrix_norm(cov_gt - cov_path, ord=norm)
        if relative:
            errors = absolute_errors / cov_gt_norm
        else:
            errors = absolute_errors
        ax.plot(time_grid, errors, label=model_name)
    error_type = "Relative error" if relative else "Absolute error"
    ax.set_xlabel("Time")
    ax.set_ylabel("Covariance Error")
    ax.set_title(error_type+" $\\|Cov(X_t)-Cov(\hat{X}_t)\\|_{"+str(norm)+"}$")
    ax.legend()
    ax.grid(True)
    plt.show()

    # Save the plot
    fk_stats_folder = save_dir + time_horizon
    os.makedirs(fk_stats_folder, exist_ok=True)
    save_plot(fig, save_dir + time_horizon, "cov_errors")


def plot_time_series_no_conf(results_time, time_grid):
    for fname, model_data in results_time.items():
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 2, 1)
        gt = model_data["Ground Truth"]
        for model_name, values in model_data.items():
            ax.plot(time_grid, values, label=model_name)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"E[{fname}(X_t) | X_0]")
        ax.set_title(f"{fname}")
        ax.legend()
        ax.grid(True)
        ax = plt.subplot(1, 2, 2)
        for model_name, values in model_data.items():
            ax.plot(time_grid, (np.array(values) - np.array(gt)) ** 2, label=model_name)
        plt.xlabel("Time Horizon T")
        plt.ylabel(f"mse")
        plt.title(f"Error of {fname}")
        plt.legend()
        plt.grid(True)
        plt.show()
        fk_stats_folder = save_dir + time_horizon
        os.makedirs(fk_stats_folder, exist_ok=True)
        save_plot(fig, fk_stats_folder, plot_name=f"mean_path_{fname}")


plot_time_series_no_conf(results_time, time_grid)
# plot_time_series(results_time, results_conf_intervals, time_grid)
plot_time_series_mean_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, False)
# plot_time_series_mean_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, True)
plot_time_series_mean_deviation_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full)
plot_time_series_variance_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full)
plot_time_series_cov_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, False, "fro")
# plot_time_series_cov_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, True, "fro")
plot_time_series_cov_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, False, "nuc")
# plot_time_series_cov_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, True, "nuc")
plot_time_series_cov_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, False, -2)
# plot_time_series_cov_errors(time_grid, paths_ground_truth_full, paths_ambient_full, paths_ae_full, True, -2)

