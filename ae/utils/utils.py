"""
    This module contains utility functions for PyTorch.

    Specifically, we have

    set_grad_tracking: turn on/off the parameters of a nn.Module
    select_device: choose the computational device: cpu or gpu (cuda, or mps)
    process_data: take the point cloud/dynamics data and estimate the orthogonal projection
    compute_orthogonal_projection_from_cov:

"""

import torch
import numpy as np
from torch import nn

class FrobeniusClipParametrization(nn.Module):
    def __init__(self, max_norm: float):
        super().__init__()
        self.max_norm = max_norm

    def forward(self, X):
        frob_norm = torch.norm(X, p='fro')
        if frob_norm > self.max_norm:
            X = X * (self.max_norm / frob_norm)
        return X


def random_rotation_matrix(D, seed=None):
    A = np.random.default_rng(seed).standard_normal((D, D))
    Q, R = np.linalg.qr(A)
    diag_sign = np.sign(np.diag(R))
    Q *= diag_sign
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    print("Seed = "+str(seed))
    print(Q)
    return Q

# Padding operator: pad with (D-d) zeros
def pad(v, target_dim):
    pad_width = [(0, 0)] * (v.ndim - 1) + [(0, target_dim - v.shape[-1])]
    return np.pad(v, pad_width, mode='constant')

def embed_data(x, mu, Sigma, R):
    """
    Rotate an SDE process in batch.

    Parameters:
    - x:      (n, d) array of points
    - mu:     (n, d) array of drifts
    - Sigma:  (n, d, d) array of covariances
    - R:      (D, D) rotation matrix (orthogonal)

    Returns:
    - x_rot:      (n, D) rotated positions
    - mu_rot:     (n, D) rotated drifts
    - Sigma_rot:  (n, D, D) rotated covariances
    """
    n, d = x.shape
    D = R.shape[0]
    assert R.shape == (D, D)
    assert mu.shape == (n, d)
    assert Sigma.shape == (n, d, d)



    # Pad x and mu to shape (n, D)
    x_padded = pad(x, D)
    mu_padded = pad(mu, D)

    # Pad Sigma to (n, D, D)
    Sigma_padded = np.zeros((n, D, D))
    Sigma_padded[:, :d, :d] = Sigma

    # Apply rotation
    x_rot = x_padded @ R.T
    mu_rot = mu_padded @ R.T
    Sigma_rot = R @ Sigma_padded @ R.T  # shape (n, D, D)

    return x_rot, mu_rot, Sigma_rot


def compute_orthogonal_projection_from_cov(cov, d=2):
    """

    :param cov: the ambient covariance
    :param d: the intrinsic dimension
    :return: a DxD matrix representing the orthogonal projection onto the d-dimensional tangent space
    """
    left_singular_vectors = np.linalg.svd(cov)[0]
    orthonormal_frame = left_singular_vectors[:, :, 0:d]
    observed_projection = np.matmul(orthonormal_frame, orthonormal_frame.transpose(0, 2, 1))
    return observed_projection


def process_data(x, mu, cov, d, return_frame=True, device="cpu"):
    """     Use SVD to compute the orthogonal projection P onto the tangent space at x given
    the infinitesimal covariance at x. Also convert everything to torch with 32 bit floating point numbers with the
    correct device. Optionally return the orthonormal frame H in P=HH^T
    """
    x = torch.tensor(x, dtype=torch.float32, device=device)
    mu = torch.tensor(mu, dtype=torch.float32, device=device)
    cov = torch.tensor(cov, dtype=torch.float32, device=device)
    left_singular_vectors = torch.linalg.svd(cov)[0]
    orthonormal_frame = left_singular_vectors[:, :, 0:d]
    observed_projection = torch.bmm(orthonormal_frame, orthonormal_frame.mT)
    n, D, _ = observed_projection.size()
    observed_normal_projection = torch.eye(D, device=device).expand(n, D, D) - observed_projection
    if return_frame:
        return x, mu, cov, observed_projection, observed_normal_projection, orthonormal_frame
    else:
        return x, mu, cov, observed_projection, observed_normal_projection


def set_grad_tracking(model: nn.Module, enable: bool = False) -> None:
    """
    Enable or disable gradient tracking for a nn.Module's parameters.

    Parameters:
    model (nn.Module): The model for which to toggle gradient tracking.
    enable (bool, optional): True to enable gradient tracking, False to disable. Default is False.

    Returns:
    None
    """
    for parameter in model.parameters():
        parameter.requires_grad = enable
    return None


# define function to set device
def select_device(preferred_device=None):
    """
        Selects the appropriate device for PyTorch computations.

        Parameters:
        preferred_device (str, optional): The preferred device to use ('cuda', 'mps', or 'cpu').
                                          If not specified, the function will select the best available device.

        Returns:
        torch.device: The selected device.

        If the preferred device is not available, it falls back to the first available device in the order of
        'cuda', 'mps', 'cpu'.
    """
    available_devices = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True  # cpu is always available
    }

    if preferred_device:
        if preferred_device in available_devices and available_devices[preferred_device]:
            device = torch.device(preferred_device)
            print(f"Using {preferred_device.upper()}.")
        else:
            print(f"{preferred_device.upper()} is not available. Falling back to available devices.")
            device = next((torch.device(dev) for dev, available in available_devices.items() if available),
                          torch.device("cpu"))
            print(f"Using {device.type.upper()}.")
    else:
        device = next((torch.device(dev) for dev, available in available_devices.items() if available),
                      torch.device("cpu"))
        print(f"Using {device.type.upper()}.")

    return device


if __name__ == "__main__":
    # usage examples
    d = select_device("cuda")  # tries to use cuda
    print(d)

    d = select_device("mps")  # tries to use mps
    print(d)

    d = select_device("tpu")  # invalid device, falls back to available options
    print(d)

    d = select_device("cpu")  # invalid device, falls back to available options
    print(d)

    d = select_device()  # no preference, uses available devices in the predefined order
    print(d)
