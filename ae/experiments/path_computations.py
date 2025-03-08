"""
    Collection of functions for computing statistics of sample paths over time.
"""
import numpy as np


def apply_function(gt, amb, aes: dict, func):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :param func:
    :return:
    """
    gt_func = func(gt)
    amb_func = func(amb)
    aes_func = {name: func(ensemble) for name, ensemble in aes.items()}
    return gt_func, amb_func, aes_func


def compute_covariance_sample_path(paths):
    """
    Compute the covariance matrix for each time step across the ensemble

    Args:
        paths (numpy.ndarray): Ensemble paths of shape (n_ensemble, n_time, n_dim)

    Returns:
        numpy.ndarray: Covariance matrices of shape (n_time, n_dim, n_dim)
    """
    n_time, n_dim = paths.shape[1], paths.shape[2]
    covariances = np.zeros((n_time, n_dim, n_dim))
    for t in range(n_time):
        covariances[t] = np.cov(paths[:, t, :], rowvar=False, bias=True)  # bias=True for MLE-like normalization
    return covariances


def compute_increments(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, lambda x: np.diff(x, axis=1))


def compute_norms(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, lambda x: np.linalg.vector_norm(x, axis=2, ord=2))


def compute_mean_sample_paths(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, lambda x: np.mean(x, axis=0))


def compute_variance_sample_paths(gt, amb, aes: dict):
    """

    :param gt: (npaths, ntime+1, D)
    :param amb: (npaths, ntime+1, D)
    :param aes: dict of {names: (npaths, ntime+1, D), ...}
    :return:
    """
    return apply_function(gt, amb, aes, lambda x: np.var(x, axis=0))


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
