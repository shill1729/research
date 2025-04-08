"""
    Collection of functions for computing statistics of sample paths over time.
"""
import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import trapz

from ae.experiments.training.helpers import save_table


def apply_function(gt, amb, aes: dict, func):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :param func:
    :return:
    """
    gt_func = func(torch.tensor(gt, dtype=torch.float32, device=amb.device))
    amb_func = func(amb)
    aes_func = {name: func(ensemble) for name, ensemble in aes.items()}
    return gt_func, amb_func, aes_func


def compute_covariance_sample_path(paths: torch.Tensor):
    """
    Compute the covariance matrix for each time step across the ensemble

    Args:
        paths (numpy.ndarray): Ensemble paths of shape (n_ensemble, n_time, n_dim)

    Returns:
        numpy.ndarray: Covariance matrices of shape (n_time, n_dim, n_dim)
    """
    n_time, n_dim = paths.size()[1], paths.size()[2]
    covariances = torch.zeros((n_time, n_dim, n_dim))
    for t in range(n_time):
        covariances[t] = torch.cov(paths[:, t, :].mT)  # bias=True for MLE-like normalization
    return covariances


def compute_increments(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, lambda x: torch.diff(x, dim=1))


def compute_norms(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, lambda x: torch.linalg.vector_norm(x, dim=2, ord=2))


def compute_mean_sample_paths(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, lambda x: torch.mean(x, dim=0))


def compute_variance_sample_paths(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, lambda x: torch.var(x, dim=0))


def compute_covariance_sample_paths(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, compute_covariance_sample_path)


def feynman_kac_formula(gt, amb, aes: dict, func):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :param func: function of x
    :return:
    """
    fgt, famb, faes = apply_function(gt, amb, aes, func)
    return compute_mean_sample_paths(fgt, famb, faes)


def kl_divergence(p_kde, q_kde, x_range):
    """Compute KL divergence D_KL(P || Q) using numerical integration."""
    p_vals = p_kde(x_range)
    q_vals = q_kde(x_range)

    # Ensure numerical stability
    p_vals = np.clip(p_vals, 1e-10, None)
    q_vals = np.clip(q_vals, 1e-10, None)

    kl_div = trapz(p_vals * (np.log(p_vals) - np.log(q_vals)), x_range)
    return kl_div


def compute_kl_divergences(gt_ensemble, model_ensembles, ambient_ensemble=None, terminal=True, save_folder=None):
    """Computes KL divergences for all models against GT and organizes results into separate tables per coordinate."""
    npaths, ntime_plus_1, d = gt_ensemble.shape
    terminal_idx = -1 if terminal else 1
    time_type = "terminal" if terminal else "1st-step"

    all_dfs = {}

    for i in range(d):
        # Compute GT KDE
        gt_kde = gaussian_kde(gt_ensemble[:, terminal_idx, i])
        x_range = np.linspace(np.min(gt_ensemble[:, terminal_idx, i]),
                              np.max(gt_ensemble[:, terminal_idx, i]), 1000)

        model_kl_values = {}  # Store KL divergences

        if ambient_ensemble is not None:
            ambient_kde = gaussian_kde(ambient_ensemble[:, terminal_idx, i])
            kl_ambient = kl_divergence(gt_kde, ambient_kde, x_range)
            model_kl_values["Ambient Model"] = kl_ambient

        for model_name, model in model_ensembles.items():
            model_kde = gaussian_kde(model[:, terminal_idx, i].cpu().detach())
            kl_model = kl_divergence(gt_kde, model_kde, x_range)
            model_kl_values[model_name] = kl_model

        # Find the "winning" model (lowest KL divergence)
        best_model = min(model_kl_values, key=model_kl_values.get)

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(model_kl_values, orient="index", columns=["KL Divergence"])
        df.index.name = "Model"

        # Apply LaTeX bold formatting to the best model
        df["KL Divergence"] = df["KL Divergence"].apply(lambda x: f"\\textbf{{{x:.4f}}}" if x == model_kl_values[best_model] else f"{x:.4f}")

        # Store the DataFrame
        all_dfs[f"Coordinate_{i+1}"] = df

        # Save LaTeX table
        if save_folder is not None:
            latex_table = df.to_latex(escape=False)  # Allow LaTeX bold formatting
            save_table(latex_table, save_folder, f"kl_divergence_{time_type}_coord_{i+1}")

    return all_dfs
