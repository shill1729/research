from ae.experiments.datagen import ToyData
from ae.experiments.training import Trainer
from ae.experiments.pathgen import SamplePathGenerator
from ae.experiments.path_plotting import SamplePathPlotter

import numpy as np


class DynamicsError:
    def __init__(self, toydata: ToyData, trainer: Trainer):
        self.toydata = toydata
        self.trainer = trainer
        self.sample_path_generator = SamplePathGenerator(self.toydata, self.trainer)
        self.sample_path_plotter = SamplePathPlotter(self.toydata, self.trainer)

    def get_time_horizon_name(self, tn):
        """
        Determine the time horizon category based on tn value

        Args:
            tn (float): Maximum time horizon

        Returns:
            str: Time horizon category name
        """
        if tn <= 0.01:
            return "/very_short_term/"
        elif 0.01 < tn <= 0.05:
            return "/short_term/"
        elif 0.05 < tn <= 0.8:
            return "/medium_term/"
        elif 0.8 < tn <= 5.0:
            return "/long_term/"
        else:
            return "/very_long_term/"

    def get_optimal_ntime(self, tn):
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

    def compute_conditional_expectation(self, ensemble_at_t, f):
        """
        Compute the conditional expectation of a function over an ensemble at a specific time

        Args:
            ensemble_at_t (numpy.ndarray): Ensemble states at time t
            f (function): Function to evaluate

        Returns:
            float: Mean function value
        """
        return np.mean([f(x) for x in ensemble_at_t])

    def compute_confidence_intervals(self, ensemble, f):
        """
        Compute mean and confidence intervals (standard error of the mean) over ensemble paths

        Args:
            ensemble (numpy.ndarray): Ensemble paths
            f (function): Function to evaluate

        Returns:
            tuple: (means, standard_errors)
        """
        ntime = ensemble.shape[1]
        means = np.zeros(ntime)
        std_errors = np.zeros(ntime)

        for t in range(ntime):
            values = np.array([f(x) for x in ensemble[:, t, :]])
            means[t] = np.mean(values)
            std_errors[t] = np.std(values) / np.sqrt(len(values))  # Standard error of the mean
        return means, std_errors

    def get_standard_test_functions(self):
        """
        Return a list of standard test functions for evaluating dynamics

        Returns:
            list: List of (name, function) tuples
        """
        return [
            ("l2 norm", lambda x: np.linalg.norm(x)),
            ("l1 norm", lambda x: np.sum(np.abs(x))),
            ("1st", lambda x: x[0]),
            ("2nd", lambda x: x[1]),
            ("3rd", lambda x: x[2]),
            ("Manifold constr", lambda x: self.chart_error(x))
        ]

    def analyze_statistical_properties(self, tn, npaths=1000, seed=None):
        """
        Analyze statistical properties of paths for different models

        Args:
            tn (float): Maximum time horizon
            npaths (int, optional): Number of sample paths per model
            seed (int, optional): Random seed for reproducibility

        Returns:
            dict: Dictionary containing time series results
        """
        # Determine appropriate number of time steps
        ntime = self.get_optimal_ntime(tn)
        time_horizon = self.get_time_horizon_name(tn)
        time_grid = np.linspace(0, tn, ntime + 1)

        # Generate paths
        paths_ground_truth, paths_ambient, paths_ae, local_gt, local_aes = (
            self.sample_path_generator.generate_paths(tn, ntime, npaths, seed))

        # Initialize results dictionaries
        f_functions = self.get_standard_test_functions()
        results_time = {fname: {} for fname, _ in f_functions}
        results_conf_intervals = {fname: {} for fname, _ in f_functions}

        # Compute statistics for each test function
        for fname, f in f_functions:
            # Ground truth
            results_time[fname]["Ground Truth"], results_conf_intervals[fname]["Ground Truth"] = \
                self.compute_confidence_intervals(paths_ground_truth, f)

            # Ambient model
            results_time[fname]["Ambient"], results_conf_intervals[fname]["Ambient"] = \
                self.compute_confidence_intervals(paths_ambient, f)

            # Autoencoder models
            for name, paths in paths_ae.items():
                results_time[fname][name], results_conf_intervals[fname][name] = \
                    self.compute_confidence_intervals(paths, f)

        return {
            "time_grid": time_grid,
            "results_time": results_time,
            "results_conf_intervals": results_conf_intervals,
            "paths_ground_truth": paths_ground_truth,
            "paths_ambient": paths_ambient,
            "paths_ae": paths_ae,
            "time_horizon": time_horizon
        }
