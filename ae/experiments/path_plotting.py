from ae.experiments.datagen import ToyData
from ae.experiments.training import Trainer
from ae.experiments.pathgen import SamplePathGenerator
from ae.experiments.helpers import save_plot

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# TODO: decide what and how we want to plot it.


class SamplePathPlotter:
    def __init__(self, toydata: ToyData, trainer: Trainer):
        self.toydata = toydata
        self.trainer = trainer
        self.sample_path_generator = SamplePathGenerator(toydata, trainer)

    def plot_ambient_sample_paths(self, gt, ae_paths_dict, vanilla_ambient_paths):
        """

        :param gt:
        :param vanilla_ambient_paths:
        :param ae_paths_dict:
        :return:
        """
        # TODO: eventually we should add 2d cases.
        # gt, vanilla_ambient_paths, ae_paths, _ = self.sample_path_generator.generate_paths(tn, ntime, npaths, seed)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Define colors
        colors = {"Ground Truth": "black", "Ambient Model": "blue"}
        for i, name in enumerate(self.trainer.models.keys()):
            if i == 0:
                colors[name] = "orange"
            elif i == 1:
                colors[name] = "green"
            elif i == 2:
                colors[name] = "red"
            else:
                colors[name] = f"C{i + 3}"  # Additional colors from matplotlib cycle

        # Plot ground truth
        for path in gt:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colors["Ground Truth"], alpha=0.3)

        # Plot ambient paths
        for path in vanilla_ambient_paths:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colors["Ambient Model"], alpha=0.3)

        # Plot AE paths
        for name, paths in ae_paths_dict.items():
            for path in paths:
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colors[name], alpha=0.3)

        # Set legend with representative colors
        handles = [plt.Line2D([0], [0], color=color, lw=2, label=label) for label, color in colors.items()]
        ax.legend(handles=handles)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Sample Paths in 3D Space")
        plt.show()
        # TODO: add save plot
        plt.close(fig)

    def plot_local_sample_paths(self, gt, ae_paths_dict):
        """

        :param gt:
        :param vanilla_ambient_paths:
        :param ae_paths_dict:
        :return:
        """
        # TODO: eventually we should add 2d cases.
        # gt, vanilla_ambient_paths, ae_paths, _ = self.sample_path_generator.generate_paths(tn, ntime, npaths, seed)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Define colors
        colors = {"Ground Truth": "black", "Ambient Model": "blue"}
        for i, name in enumerate(self.trainer.models.keys()):
            if i == 0:
                colors[name] = "orange"
            elif i == 1:
                colors[name] = "green"
            elif i == 2:
                colors[name] = "red"
            else:
                colors[name] = f"C{i + 3}"  # Additional colors from matplotlib cycle

        # Plot ground truth
        for path in gt:
            ax.plot(path[:, 0], path[:, 1], color=colors["Ground Truth"], alpha=0.3)

        # Plot AE paths
        for name, paths in ae_paths_dict.items():
            for path in paths:
                ax.plot(path[:, 0], path[:, 1], color=colors[name], alpha=0.3)

        # Set legend with representative colors
        handles = [plt.Line2D([0], [0], color=color, lw=2, label=label) for label, color in colors.items()]
        ax.legend(handles=handles)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Sample Paths in 2D Space")
        plt.show()
        # TODO: add save plot.
        plt.close(fig)

    @staticmethod
    def plot_kernel_density(gt_ensemble, model_ensembles, ambient_ensemble=None, terminal=True):
        """
        Plots kernel density estimates for the terminal time step across d=3 coordinates.
        Compares the ground truth (GT) to the ambient model and other models.

        Parameters:
        - gt_ensemble: numpy array of shape (npaths, ntime+1, d), ground truth ensemble.
        - ambient_ensemble: numpy array of shape (npaths, ntime+1, d), ambient model ensemble.
        - model_ensembles: dict with model names as keys and numpy arrays of shape (npaths, ntime+1, d).
        - terminal: True for terminal distribution, false for first-step distribution
        """
        npaths, ntime_plus_1, d = gt_ensemble.shape
        # TODO: label the plot based on terminal density or first step density
        # Extract only the terminal time step
        # terminal_idx = ntime_plus_1 - 1
        if terminal:
            time_type = "terminal"
            terminal_idx = -1
        else:
            time_type = "1st-step"
            terminal_idx = 1
        gt_values = gt_ensemble[:, terminal_idx, :]
        ambient_values = None
        if ambient_ensemble is not None:
            ambient_values = ambient_ensemble[:, terminal_idx, :]

        model_values = {name: ens[:, terminal_idx, :] for name, ens in model_ensembles.items()}

        fig, axes = plt.subplots(1, d, figsize=(15, 5))

        for i in range(d):  # Iterate over coordinates
            ax = axes[i]
            sns.kdeplot(gt_values[:, i], label="GT", linewidth=2, ax=ax)
            if ambient_values is not None:
                sns.kdeplot(ambient_values[:, i], label="Ambient Model", linewidth=2, linestyle="dashed", ax=ax)

            for model_name, model in model_values.items():
                sns.kdeplot(model[:, i], label=model_name, linewidth=2, linestyle="dashed", ax=ax)

            ax.set_xlabel(f"Coordinate {i + 1}")
            ax.set_ylabel("Density")
            ax.set_title(time_type+f" Density for coordinate {i + 1}")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()
        # TODO: add save plot
        plt.close(fig)

    def plot_time_series_with_errors(self, results, save_dir=None):
        """
        Plot time series with errors for each test function

        Args:
            results (dict): Results from analyze_statistical_properties
            save_dir (str, optional): Directory to save plots
        """
        time_grid = results["time_grid"]
        results_time = results["results_time"]
        time_horizon = results["time_horizon"]

        for fname, model_data in results_time.items():
            fig = plt.figure(figsize=(12, 5))
            # Plot means
            ax1 = fig.add_subplot(1, 2, 1)
            gt = model_data["Ground Truth"]
            for model_name, values in model_data.items():
                ax1.plot(time_grid, values, label=model_name)
            ax1.set_xlabel("Time")
            ax1.set_ylabel(f"E[{fname}(X_t) | X_0]")
            ax1.set_title(f"{fname}")
            ax1.legend()
            ax1.grid(True)

            # Plot squared errors
            ax2 = fig.add_subplot(1, 2, 2)
            for model_name, values in model_data.items():
                if model_name != "Ground Truth":  # Skip ground truth in error plot
                    ax2.plot(time_grid, (np.array(values) - np.array(gt)) ** 2, label=model_name)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("MSE")
            ax2.set_title(f"Error of {fname}")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()
            fk_stats_folder = self.trainer.exp_dir + time_horizon
            os.makedirs(fk_stats_folder, exist_ok=True)
            save_plot(fig, fk_stats_folder, "feynman_kac_error_" + fname)
            plt.close(fig)

    def plot_time_series_with_confidence(self, results, save_dir=None):
        """
        Plot time series with confidence intervals for each test function

        Args:
            results (dict): Results from analyze_statistical_properties
            save_dir (str, optional): Directory to save plots
        """
        time_grid = results["time_grid"]
        results_time = results["results_time"]
        results_conf_intervals = results["results_conf_intervals"]
        time_horizon = results["time_horizon"]

        for fname, model_data in results_time.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            for model_name, values in model_data.items():
                conf_intervals = results_conf_intervals[fname][model_name]
                ax.plot(time_grid, values, label=model_name)
                ax.fill_between(time_grid, values - conf_intervals, values + conf_intervals, alpha=0.2)

            ax.set_xlabel("Time")
            ax.set_ylabel(f"E[{fname}(X_t) | X_0]")
            ax.set_title(f"{fname} with Confidence Intervals")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.show()

            # Save the plot if directory is provided
            fk_stats_folder = self.trainer.exp_dir + time_horizon
            os.makedirs(fk_stats_folder, exist_ok=True)
            save_plot(fig, fk_stats_folder, "feynman_kac_conf_" + fname)
            plt.close(fig)

    def plot_mean_errors(self, results, relative=False, save_dir=None):
        """
        Plot mean errors between ground truth and model predictions

        Args:
            results (dict): Results from analyze_statistical_properties
            relative (bool, optional): Whether to compute relative errors
            save_dir (str, optional): Directory to save plots
        """
        time_grid = results["time_grid"]
        paths_ground_truth = results["paths_ground_truth"]
        paths_ambient = results["paths_ambient"]
        paths_ae = results["paths_ae"]
        time_horizon = results["time_horizon"]

        # Compute mean over ensembles for each time step
        mean_gt_path = paths_ground_truth.mean(axis=0)  # Shape: (ntime, 3)
        mean_ambient_path = paths_ambient.mean(axis=0)  # Shape: (ntime, 3)
        mean_ae_paths = {name: paths.mean(axis=0) for name, paths in paths_ae.items()}

        # Combine all means
        model_means = {"Ground Truth": mean_gt_path, **mean_ae_paths, "Ambient": mean_ambient_path}

        # Compute ground truth norm for relative errors
        mean_gt_norm = np.linalg.vector_norm(mean_gt_path, axis=1, ord=2, keepdims=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot errors for each model
        for model_name, mean_path in model_means.items():
            if model_name != "Ground Truth":  # Skip ground truth
                abs_errors = np.linalg.vector_norm(mean_gt_path - mean_path, axis=1, ord=2, keepdims=False)
                if relative:
                    errors = abs_errors / mean_gt_norm
                    label = f"{model_name} (Relative)"
                else:
                    errors = abs_errors
                    label = model_name
                ax.plot(time_grid, errors, label=label)

        error_type = "Relative error of " if relative else "Absolute error of"
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Error")
        ax.set_title(error_type + " $\\|E(X_t)-E(\hat{X}_t)\\|_2$")
        ax.legend()
        ax.grid(True)
        plt.show()

        # Save the plot if directory is provided
        fk_stats_folder = self.trainer.exp_dir + time_horizon
        os.makedirs(fk_stats_folder, exist_ok=True)
        save_plot(fig, fk_stats_folder, plot_name="error of means")
        plt.close(fig)

    def plot_mean_deviation_errors(self, results, save_dir=None):
        """
        Plot mean Euclidean deviation errors

        Args:
            results (dict): Results from analyze_statistical_properties
            save_dir (str, optional): Directory to save plots
        """
        time_grid = results["time_grid"]
        paths_ground_truth = results["paths_ground_truth"]
        paths_ambient = results["paths_ambient"]
        paths_ae = results["paths_ae"]
        time_horizon = results["time_horizon"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute mean Euclidean deviation over ensembles for each time step
        mean_deviation_ambient = np.linalg.vector_norm(paths_ground_truth - paths_ambient, ord=2, axis=2)
        mean_deviation_ae = {name: np.linalg.vector_norm(paths_ground_truth - paths, ord=2, axis=2)
                             for name, paths in paths_ae.items()}

        model_deviation_means = {
            "Ambient": mean_deviation_ambient.mean(axis=0),
            **{name: dev.mean(axis=0) for name, dev in mean_deviation_ae.items()}}

        # Plot mean Euclidean deviations
        for model_name, deviations in model_deviation_means.items():
            ax.plot(time_grid, deviations, label=model_name)

        ax.set_xlabel("Time")
        ax.set_ylabel("Mean of Euclidean deviation")
        ax.set_title("$E(\\|X_t-\hat{X}_t\\|_2)$")
        ax.legend()
        ax.grid(True)
        plt.show()

        # Save the plot if directory is provided
        fk_stats_folder = self.trainer.exp_dir + time_horizon
        os.makedirs(fk_stats_folder, exist_ok=True)
        save_plot(fig, fk_stats_folder, plot_name="mean of euclidean distance to gt")
        plt.close(fig)

    def plot_variance_deviation_errors(self, results, save_dir=None):
        """
        Plot variance of Euclidean deviation errors

        Args:
            results (dict): Results from analyze_statistical_properties
            save_dir (str, optional): Directory to save plots
        """
        time_grid = results["time_grid"]
        paths_ground_truth = results["paths_ground_truth"]
        paths_ambient = results["paths_ambient"]
        paths_ae = results["paths_ae"]
        time_horizon = results["time_horizon"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute variance of Euclidean deviation over ensembles for each time step
        mean_deviation_ambient = np.linalg.vector_norm(paths_ground_truth - paths_ambient, ord=2, axis=2)
        mean_deviation_ae = {name: np.linalg.vector_norm(paths_ground_truth - paths, ord=2, axis=2)
                             for name, paths in paths_ae.items()}

        model_deviation_variances = {
            "Ambient": mean_deviation_ambient.var(axis=0),
            **{name: dev.var(axis=0) for name, dev in mean_deviation_ae.items()}}

        # Plot variance of Euclidean deviations
        for model_name, variances in model_deviation_variances.items():
            ax.plot(time_grid, variances, label=model_name)

        ax.set_xlabel("Time")
        ax.set_ylabel("Variance of Euclidean deviation")
        ax.set_title("$Var(\\|X_t-\hat{X}_t\\|_2)$")
        ax.legend()
        ax.grid(True)
        plt.show()

        # Save the plot if directory is provided
        fk_stats_folder = self.trainer.exp_dir + time_horizon
        os.makedirs(fk_stats_folder, exist_ok=True)
        save_plot(fig, fk_stats_folder, plot_name="variance of euclidean distance")
        plt.close(fig)

    def plot_covariance_errors(self, results, relative=False, norm="fro", save_dir=None):
        """
        Plot covariance matrix errors

        Args:
            results (dict): Results from analyze_statistical_properties
            relative (bool, optional): Whether to compute relative errors
            norm (str, optional): Matrix norm to use ("fro", "nuc", or -2)
            save_dir (str, optional): Directory to save plots
        """
        time_grid = results["time_grid"]
        paths_ground_truth = results["paths_ground_truth"]
        paths_ambient = results["paths_ambient"]
        paths_ae = results["paths_ae"]
        time_horizon = results["time_horizon"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute covariance matrices over ensembles for each time step
        cov_gt = self.compute_covariance(paths_ground_truth)  # Shape: (ntime, n_dim, n_dim)
        cov_ambient = self.compute_covariance(paths_ambient)  # Shape: (ntime, n_dim, n_dim)
        cov_ae_models = {name: self.compute_covariance(paths) for name, paths in paths_ae.items()}

        # Compute ground truth norm for relative errors
        cov_gt_norm = np.linalg.matrix_norm(cov_gt, ord=norm if norm != -2 else None)

        # Combine all covariance matrices
        model_covariances = {"Ground Truth": cov_gt, **cov_ae_models, "Ambient": cov_ambient}

        # Plot errors for each model
        for model_name, cov_path in model_covariances.items():
            if model_name != "Ground Truth":  # Skip ground truth
                if norm == -2:
                    # For spectral norm (-2)
                    absolute_errors = np.array([np.linalg.norm(cov_gt[t] - cov_path[t], ord=None)
                                                for t in range(len(cov_gt))])
                else:
                    absolute_errors = np.linalg.matrix_norm(cov_gt - cov_path, ord=norm)

                if relative:
                    errors = absolute_errors / cov_gt_norm
                else:
                    errors = absolute_errors
                ax.plot(time_grid, errors, label=model_name)

        error_type = "Relative error" if relative else "Absolute error"
        norm_name = str(norm) if norm != -2 else "spectral"
        ax.set_xlabel("Time")
        ax.set_ylabel("Covariance Error")
        ax.set_title(f"{error_type} $\\|Cov(X_t)-Cov(\\hat{{X}}_t)\\|_{{{norm_name}}}$")
        ax.legend()
        ax.grid(True)
        plt.show()

        # Save the plot if directory is provided
        fk_stats_folder = self.trainer.exp_dir + time_horizon
        os.makedirs(fk_stats_folder, exist_ok=True)
        save_plot(fig, fk_stats_folder, plot_name="Frobenius error of covariances")
        plt.close(fig)

    def run_all_analyses(self, results, save_dir):
        """
        Run all analyses and generate all plots

        Args:
            tn (float): Maximum time horizon
            npaths (int, optional): Number of sample paths per model
            seed (int, optional): Random seed for reproducibility
            save_dir (str, optional): Directory to save plots
        """
        # print(f"Running analyses for time horizon T_max = {tn}")
        # results = self.analyze_statistical_properties(tn, npaths, seed)

        # Generate all plots
        self.plot_time_series_with_errors(results, save_dir)
        # self.plot_time_series_with_confidence(results, save_dir)
        self.plot_mean_errors(results, False, save_dir)
        # self.plot_mean_errors(results, True, save_dir)
        self.plot_mean_deviation_errors(results, save_dir)
        self.plot_variance_deviation_errors(results, save_dir)
        self.plot_covariance_errors(results, False, "fro", save_dir)
        # self.plot_covariance_errors(results, True, "fro", save_dir)
        self.plot_covariance_errors(results, False, "nuc", save_dir)
        # self.plot_covariance_errors(results, True, "nuc", save_dir)
        self.plot_covariance_errors(results, False, -2, save_dir)
        # self.plot_covariance_errors(results, True, -2, save_dir)
        return results
