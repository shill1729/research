from ae.experiments.datagen import ToyData
from ae.experiments.training import Trainer
from ae.experiments.pathgen import SamplePathGenerator
from ae.experiments.path_plotting import SamplePathPlotter
from ae.experiments.path_computations import *
from ae.experiments.helpers import get_time_horizon_name
import numpy as np


class DynamicsError:
    def __init__(self, toydata: ToyData, trainer: Trainer, tn: float, show=False):
        self.toydata = toydata
        self.trainer = trainer
        self.sample_path_generator = SamplePathGenerator(self.toydata, self.trainer)
        self.sample_path_plotter = SamplePathPlotter(self.toydata, self.trainer, tn, show=show)

    @staticmethod
    def get_optimal_ntime(tn):
        """
        Determine the optimal number of time steps based on time horizon

        Args:
            tn (float): Maximum time horizon

        Returns:
            int: Recommended number of time steps
        """
        if tn <= 0.01:
            return 200
        elif 0.01 < tn <= 0.05:
            return 500
        elif 0.05 < tn <= 0.8:
            return 1000
        elif 0.8 < tn <= 5.0:
            return 5900
        else:
            return 10000

    def chart_error(self, x):
        """
        Calculate the manifold constraint error for a point

        Args:
            x (numpy.ndarray): Point in ambient space

        Returns:
            float: Error value
        """
        # Assuming point_cloud has a np_phi method similar to cloud_train
        p = np.abs(self.toydata.point_cloud.np_phi(*[x[0], x[1]])[2] - x[2])
        return p

    def chart_error_vectorized(self, paths):
        """
        Vectorized version of chart_error for ensemble paths

        Args:
            paths (numpy.ndarray): Ensemble paths of shape (n_ensemble, n_time, n_dim)

        Returns:
            numpy.ndarray: Error values of shape (n_ensemble, n_time)
        """
        # Extract individual coordinates for vectorized computation
        x_coords = paths[:, :, 0]  # Shape: (n_ensemble, n_time)
        y_coords = paths[:, :, 1]  # Shape: (n_ensemble, n_time)
        z_coords = paths[:, :, 2]  # Shape: (n_ensemble, n_time)

        n_ensemble, n_time = x_coords.shape
        errors = np.zeros((n_ensemble, n_time))

        # Compute errors for each point in the ensemble
        for i in range(n_ensemble):
            for t in range(n_time):
                expected_z = self.toydata.point_cloud.np_phi(x_coords[i, t], y_coords[i, t])[2]
                errors[i, t] = np.abs(expected_z - z_coords[i, t])

        return errors

    def get_standard_test_functions(self):
        """
        Return a list of standard test functions for evaluating dynamics

        Returns:
            list: List of (function name, vectorized function) tuples
        """
        return [
            ("l2 norm", lambda paths: np.linalg.vector_norm(paths, axis=2, ord=2)),
            ("polynomial", lambda paths: np.tanh(paths[:, :, 2]**2-paths[:, :, 1]*paths[:, :, 0])),
            ("cosine-poly", lambda paths: np.cos(paths[:, :, 2]**2-paths[:, :, 1]**3)*paths[:, :, 0]),
            ("sin(x1)x3", lambda paths: np.sin(4*paths[:, :, 1])*paths[:, :, 2]),
            ("rational function", lambda paths: paths[:, :, 2]/(1+paths[:, :, 1]**2+paths[:, :, 0]**2)),
            ("Manifold constraint", self.chart_error_vectorized),
            ("x1", lambda paths: paths[:, :, 0]),
            ("x2", lambda paths: paths[:, :, 1]),
            ("x3", lambda paths: paths[:, :, 2])
        ]

    def analyze_statistical_properties(self, paths_ground_truth, paths_ambient, paths_ae, tn):
        """
        Analyze statistical properties of paths for different models using vectorized operations

        Args:
            paths_ground_truth (numpy.ndarray): Ground truth paths of shape (n_ensemble, n_time, n_dim)
            paths_ambient (numpy.ndarray): Ambient model paths of shape (n_ensemble, n_time, n_dim)
            paths_ae (dict): Dictionary of autoencoder paths, each of shape (n_ensemble, n_time, n_dim)
            tn (float): Maximum time horizon

        Returns:
            dict: Dictionary containing time series results
        """
        n = paths_ground_truth.shape[0]
        # Get time horizon name
        time_horizon = get_time_horizon_name(tn)
        ntime = paths_ambient.shape[1]
        time_grid = np.linspace(0, tn, ntime)

        # Initialize results dictionaries for standard statistical measures
        means_gt, means_amb, means_ae = compute_mean_sample_paths(
            paths_ground_truth, paths_ambient, paths_ae)

        vars_gt, vars_amb, vars_ae = compute_variance_sample_paths(
            paths_ground_truth, paths_ambient, paths_ae)

        covs_gt, covs_amb, covs_ae = compute_covariance_sample_paths(
            paths_ground_truth, paths_ambient, paths_ae)

        increments_gt, increments_amb, increments_ae = compute_increments(
            paths_ground_truth, paths_ambient, paths_ae)

        norms_gt, norms_amb, norms_ae = compute_norms(
            paths_ground_truth, paths_ambient, paths_ae)

        # Process standard test functions using vectorized operations
        standard_test_results = {}
        conf_intervals_results = {}

        for fname, f_vec in self.get_standard_test_functions():
            # Apply the vectorized function to get values for all paths at all times
            gt_values, amb_values, ae_values = apply_function(
                paths_ground_truth, paths_ambient, paths_ae, f_vec)

            # Calculate means over ensemble dimension (axis=0)
            f_gt_means, f_amb_means, f_ae_means = compute_mean_sample_paths(gt_values, amb_values, ae_values)

            # Calculate standard errors of the mean (SEM)
            gt_sem, amb_sem, ae_sem = apply_function(gt_values, amb_values, ae_values,
                                                     lambda x: np.std(x, axis=0)/np.sqrt(n))

            # Store results
            standard_test_results[fname] = {
                "Ground Truth": f_gt_means,
                "Ambient Model": f_amb_means,
                **f_ae_means
            }

            conf_intervals_results[fname] = {
                "Ground Truth": gt_sem,
                "Ambient Model": amb_sem,
                **ae_sem
            }

        # Prepare additional advanced statistics
        # Feynman-Kac formulas for specific examples
        fk_l2 = feynman_kac_formula(
            paths_ground_truth, paths_ambient, paths_ae,
            lambda x: x[:, :, 0]**2-x[:, :, 1])

        fk_coord1 = feynman_kac_formula(
            paths_ground_truth, paths_ambient, paths_ae,
            lambda x: x[:, :, 0] ** 2)

        return {
            "time_grid": time_grid,
            "time_horizon": time_horizon,
            "paths_ground_truth": paths_ground_truth,
            "paths_ambient": paths_ambient,
            "paths_ae": paths_ae,

            # Standard statistics
            "means": {
                "Ground Truth": means_gt,
                "Ambient": means_amb,
                **means_ae
            },
            "variances": {
                "Ground Truth": vars_gt,
                "Ambient": vars_amb,
                **vars_ae
            },
            "covariances": {
                "Ground Truth": covs_gt,
                "Ambient": covs_amb,
                **covs_ae
            },
            "increments": {
                "Ground Truth": increments_gt,
                "Ambient": increments_amb,
                **increments_ae
            },
            "norms": {
                "Ground Truth": norms_gt,
                "Ambient": norms_amb,
                **norms_ae
            },

            # Results from standard test functions
            "results_time": standard_test_results,
            "results_conf_intervals": conf_intervals_results,

            # Advanced statistics (Feynman-Kac examples)
            "feynman_kac": {
                "l2_squared": fk_l2,
                "first_coord_squared": fk_coord1
            }
        }

