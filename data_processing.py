import torch


def process_data(x, mu, cov, d):
    x = torch.tensor(x, dtype=torch.float32)
    mu = torch.tensor(mu, dtype=torch.float32)
    cov = torch.tensor(cov, dtype=torch.float32)
    left_singular_vectors = torch.linalg.svd(cov)[0]
    orthonormal_frame = left_singular_vectors[:, :, 0:d]
    observed_projection = torch.bmm(orthonormal_frame, orthonormal_frame.mT)
    return x, mu, cov, observed_projection


