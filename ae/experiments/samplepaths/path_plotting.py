from ae.toydata.datagen import ToyData
from ae.experiments.training.training import Trainer
from ae.experiments.samplepaths.pathgen import SamplePathGenerator
from ae.experiments.samplepaths.path_computations import *
from ae.experiments.training.helpers import save_plot, get_time_horizon_name

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class SamplePathPlotter:
    def __init__(self, toydata: ToyData, trainer: Trainer, tn: float, show=False):
        self.toydata = toydata
        self.trainer = trainer
        self.sample_path_generator = SamplePathGenerator(toydata, trainer)
        self.tn = tn
        self.time_horizon = get_time_horizon_name(tn)
        self.save_folder = self.trainer.exp_dir + self.time_horizon
        self.show = show

    def _save_plot(self, fig, fk_stats_folder, name):
        os.makedirs(fk_stats_folder, exist_ok=True)
        save_plot(fig, fk_stats_folder, name)

    def _get_model_colors(self):
        """Assigns consistent colors to models."""
        colors = {"Ground Truth": "black", "Ambient Model": "blue"}
        for i, name in enumerate(self.trainer.models.keys()):
            colors[name] = ["orange", "green", "red"][i] if i < 3 else f"C{i + 3}"
        return colors

    def _setup_plot(self, title, xlabel, ylabel, is_3d=False):
        """Creates a figure and axes for plotting."""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return fig, ax

    def _plot_paths(self, ax, paths, color, alpha=0.3, is_3d=False):
        """Helper function to plot paths in 2D or 3D."""
        for path in paths:
            if is_3d:
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, alpha=alpha)
            else:
                ax.plot(path[:, 0], path[:, 1], color=color, alpha=alpha)

    def _finalize_plot(self, fig, ax, colors, plot_name=""):
        """Finalizes plot with legend and optional saving."""
        handles = [plt.Line2D([0], [0], color=color, lw=2, label=label) for label, color in colors.items()]
        ax.legend(handles=handles)
        if self.show:
            plt.show()
        self._save_plot(fig, self.save_folder, name="sample_path_"+plot_name)
        plt.close(fig)

    def plot_sample_paths(self, gt, ae_paths_dict, vanilla_ambient_paths, is_3d=True, plot_name=""):
        """Unified function for both 2D and 3D path plotting."""
        colors = self._get_model_colors()
        fig, ax = self._setup_plot(
            title="Sample Paths in 3D Space" if is_3d else "Sample Paths in 2D Space",
            xlabel="X", ylabel="Y", is_3d=is_3d
        )

        self._plot_paths(ax, gt, colors["Ground Truth"], is_3d=is_3d)
        if is_3d:
            self._plot_paths(ax, vanilla_ambient_paths.cpu(), colors["Ambient Model"], is_3d=is_3d)

        for name, paths in ae_paths_dict.items():
            self._plot_paths(ax, paths.cpu().detach(), colors[name], is_3d=is_3d)

        self._finalize_plot(fig, ax, colors, plot_name)

    def plot_kernel_density(self, gt_ensemble, model_ensembles, ambient_ensemble=None, terminal=True):
        """Generalized KDE plot for terminal or initial distributions."""
        kl_dict = compute_kl_divergences(gt_ensemble, model_ensembles, ambient_ensemble.cpu(), terminal, self.save_folder)
        npaths, ntime_plus_1, d = gt_ensemble.shape
        time_type = "terminal" if terminal else "1st-step"
        terminal_idx = -1 if terminal else 1
        colors = self._get_model_colors()
        max_plots = min(d, 4)
        fig, axes = plt.subplots(1, max_plots, figsize=(15, 5))

        for i in range(max_plots):
            ax = axes[i]
            sns.kdeplot(gt_ensemble[:, terminal_idx, i], label="GT", color=colors["Ground Truth"], linewidth=2, ax=ax)
            if ambient_ensemble is not None:
                sns.kdeplot(ambient_ensemble[:, terminal_idx, i], label="Ambient Model", color=colors["Ambient Model"],linewidth=2,
                            linestyle="dashed", ax=ax)

            for model_name, model in model_ensembles.items():
                sns.kdeplot(model[:, terminal_idx, i], label=model_name, color=colors[model_name],linewidth=2, linestyle="dashed", ax=ax)

            ax.set_xlabel(f"Coordinate {i + 1}")
            ax.set_title(f"{time_type} Density for coordinate {i + 1}")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        if self.show:
            plt.show()
        # Only save ambient coordinate densities, since we know locals aren't going to match.
        if ambient_ensemble is not None:
            save_plot(fig, self.save_folder, plot_name="kde_" + time_type)
        plt.close(fig)
        return kl_dict

    def plot_time_series_with_errors(self, results):
        """
        Plot time series with errors for each test function

        Args:
            results (dict): Results from analyze_statistical_properties
            save_dir (str, optional): Directory to save plots
        """
        time_grid = results["time_grid"]
        results_time = results["results_time"]
        time_horizon = results["time_horizon"]
        colors = self._get_model_colors()
        for fname, model_data in results_time.items():
            fig = plt.figure(figsize=(12, 5))
            # Plot means
            ax1 = fig.add_subplot(1, 2, 1)
            gt = model_data["Ground Truth"]
            for model_name, values in model_data.items():
                if model_name != "Ambient Model":
                    ax1.plot(time_grid, values.cpu().detach(), label=model_name, color=colors[model_name])
            ax1.set_xlabel("Time")
            ax1.set_ylabel(f"E[{fname}(X_t) | X_0]")
            ax1.set_title(f"{fname}")
            ax1.legend()
            ax1.grid(True)

            # Plot squared errors
            ax2 = fig.add_subplot(1, 2, 2)
            for model_name, values in model_data.items():
                if model_name != "Ground Truth" and model_name != "Ambient Model":  # Skip ground truth in error plot
                    ax2.plot(time_grid, (np.array(values.cpu().detach()) - np.array(gt.cpu().detach())) ** 2, label=model_name, color=colors[model_name])
            ax2.set_xlabel("Time")
            ax2.set_ylabel("MSE")
            ax2.set_title(f"Error of {fname}")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            if self.show:
                plt.show()
            self._save_plot(fig, self.save_folder, "fk_"+fname)
            plt.close(fig)

    def plot_time_series_with_confidence(self, results):
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
        colors = self._get_model_colors()
        for fname, model_data in results_time.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            for model_name, values in model_data.items():
                conf_intervals = results_conf_intervals[fname][model_name]
                ax.plot(time_grid, values, label=model_name, color=colors[model_name])
                ax.fill_between(time_grid, values - conf_intervals, values + conf_intervals, alpha=0.2, color=colors[model_name])

            ax.set_xlabel("Time")
            ax.set_ylabel(f"E[{fname}(X_t) | X_0]")
            ax.set_title(f"{fname} with Confidence Intervals")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            if self.show:
                plt.show()

            # Save the plot if directory is provided
            self._save_plot(fig, self.save_folder, "fk_conf" + fname)
            plt.close(fig)

    def plot_deviation_of_means(self, results, relative=False, plot_name=""):
        """
        Plot mean errors between ground truth and model predictions

        Args:
            results (dict): Results from analyze_statistical_properties
            relative (bool, optional): Whether to compute relative errors
            save_dir (str, optional): Directory to save plots
        """
        time_grid = results["time_grid"]
        gt = results["paths_ground_truth"]
        amb = results["paths_ambient"]
        ae_dict = results["paths_ae"]
        time_horizon = results["time_horizon"]
        # Compute mean over ensembles for each time step
        mean_gt_path, mean_ambient_path, mean_ae_paths = compute_mean_sample_paths(gt, amb, ae_dict)
        # Combine all means
        model_means = {"Ground Truth": mean_gt_path, **mean_ae_paths, "Ambient Model": mean_ambient_path}
        # Compute ground truth norm for relative errors
        mean_gt_norm = np.linalg.vector_norm(mean_gt_path.cpu().detach().numpy(), axis=1, ord=2, keepdims=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = self._get_model_colors()
        # Plot errors for each model
        for model_name, mean_path in model_means.items():
            if model_name != "Ground Truth":  # Skip ground truth
                abs_errors = np.linalg.vector_norm(mean_gt_path.cpu().detach().numpy() - mean_path.cpu().detach().numpy(), axis=1, ord=2, keepdims=False)
                if relative:
                    errors = abs_errors / mean_gt_norm
                    label = f"{model_name} (Relative)"
                else:
                    errors = abs_errors
                    label = model_name
                ax.plot(time_grid, errors, label=label, color=colors[model_name])
        error_type = "Relative error of " if relative else "Absolute error of"
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Error")
        ax.set_title(error_type + " $\\|E(X_t)-E(\hat{X}_t)\\|_2$")
        ax.legend()
        ax.grid(True)
        if self.show:
            plt.show()
        # Save the plot if directory is provided
        self._save_plot(fig, self.save_folder, "deviation_of_means_"+plot_name)
        plt.close(fig)

    def plot_mean_deviation(self, results, plot_name=""):
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
        colors = self._get_model_colors()
        # Compute mean Euclidean deviation over ensembles for each time step
        deviation_ambient = np.linalg.vector_norm(paths_ground_truth - paths_ambient.cpu().detach().numpy(), ord=2, axis=2)
        deviation_ae = {name: np.linalg.vector_norm(paths_ground_truth - paths.cpu().detach().numpy(), ord=2, axis=2)
                             for name, paths in paths_ae.items()}

        model_deviation_means = {
            "Ambient Model": deviation_ambient.mean(axis=0),
            **{name: dev.mean(axis=0) for name, dev in deviation_ae.items()}}

        # Plot mean Euclidean deviations
        for model_name, deviations in model_deviation_means.items():
            ax.plot(time_grid, deviations, label=model_name, color=colors[model_name])

        ax.set_xlabel("Time")
        ax.set_ylabel("Mean of Euclidean deviation")
        ax.set_title("$E(\\|X_t-\hat{X}_t\\|_2)$")
        ax.legend()
        ax.grid(True)
        if self.show:
            plt.show()

        # Save the plot if directory is provide
        self._save_plot(fig, self.save_folder, name="mean_deviation_"+plot_name)
        plt.close(fig)

    def plot_variance_deviation_errors(self, results, plot_name=""):
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
        colors = self._get_model_colors()
        # Compute variance of Euclidean deviation over ensembles for each time step
        deviation_ambient = np.linalg.vector_norm(paths_ground_truth - paths_ambient.cpu().detach().numpy(), ord=2, axis=2)
        deviation_ae = {name: np.linalg.vector_norm(paths_ground_truth - paths.cpu().detach().numpy(), ord=2, axis=2)
                        for name, paths in paths_ae.items()}

        model_deviation_variances = {
            "Ambient Model": deviation_ambient.var(axis=0),
            **{name: dev.var(axis=0) for name, dev in deviation_ae.items()}}

        # Plot variance of Euclidean deviations
        for model_name, variances in model_deviation_variances.items():
            ax.plot(time_grid, variances, label=model_name, color=colors[model_name])

        ax.set_xlabel("Time")
        ax.set_ylabel("Variance of Euclidean deviation")
        ax.set_title("$Var(\\|X_t-\\hat{X}_t\\|_2)$")
        ax.legend()
        ax.grid(True)
        if self.show:
            plt.show()

        # Save the plot if directory is provided
        self._save_plot(fig, self.save_folder, name="variance_deviation_"+plot_name)
        plt.close(fig)

    def plot_covariance_errors(self, results, relative=False, norm="fro", plot_name=""):
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
        colors = self._get_model_colors()
        # Compute covariance matrices over ensembles for each time step
        cov_gt, cov_ambient, cov_ae_models = compute_covariance_sample_paths(paths_ground_truth, paths_ambient, paths_ae)

        # Compute ground truth norm for relative errors
        cov_gt_norm = np.linalg.matrix_norm(cov_gt, ord=norm)

        # Combine all covariance matrices
        model_covariances = {"Ground Truth": cov_gt, **cov_ae_models, "Ambient Model": cov_ambient}

        # Plot errors for each model
        for model_name, cov_path in model_covariances.items():
            if model_name != "Ground Truth":  # Skip ground truth
                absolute_errors = np.linalg.matrix_norm(cov_gt - cov_path.detach(), ord=norm)

                if relative:
                    errors = absolute_errors / cov_gt_norm
                else:
                    errors = absolute_errors
                ax.plot(time_grid, errors, label=model_name, color=colors[model_name])

        error_type = "Relative error" if relative else "Absolute error"
        norm_name = str(norm) if norm != -2 else "spectral"
        ax.set_xlabel("Time")
        ax.set_ylabel("Covariance Error")
        ax.set_title(f"{error_type} $\\|Cov(X_t)-Cov(\\hat{{X}}_t)\\|_{{{norm_name}}}$")
        ax.legend()
        ax.grid(True)
        if self.show:
            plt.show()

        # Save the plot if directory is providee
        self._save_plot(fig, self.save_folder, name=norm_name+"_covariance_error_"+plot_name)
        plt.close(fig)

    def run_all_analyses(self, results):
        """

        :param results: dictionary returned from DynamicsErrors.analyze_statistical_properties
        :return:
        """
        # print(f"Running analyses for time horizon T_max = {tn}")
        # results = self.analyze_statistical_properties(tn, npaths, seed)

        # Generate all plots
        self.plot_time_series_with_errors(results)
        # self.plot_time_series_with_confidence(results)
        # TODO we need to lift the ground truth sample paths when we embed.
        self.plot_deviation_of_means(results, False, plot_name="state")
        # self.plot_deviation_of_means(results, True, plot_name="state")
        self.plot_mean_deviation(results, plot_name="state")
        self.plot_variance_deviation_errors(results, plot_name="state")
        self.plot_covariance_errors(results, False, "fro", plot_name="state")
        # self.plot_covariance_errors(results, True, "fro", plot_name="state")
        self.plot_covariance_errors(results, False, "nuc", "state")
        # self.plot_covariance_errors(results, True, "nuc", "state")
        self.plot_covariance_errors(results, False, -2, "state")
        # self.plot_covariance_errors(results, True, -2, save_dir)
        return results
