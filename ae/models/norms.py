from torch import Tensor

import torch


# The Second order term coming from Ito's applied to vector valued functions (by applying Ito's component wise)
def frobenius_inner_product_single(m1: Tensor, m2: Tensor) -> Tensor:
    """
        Batched version of the Frobenius inner product for matrix m x d and m x d. The inner product of A and B is
        the trace of A^T B.

    :param m1: a tensor of shape (n, m, d), for n batches of m x d matrices
    :param m2: a tensor of shape (n, m, d), for n batches of m x d matrices
    :return: a tensor of shape (n, )
    """
    qv = torch.einsum("njk,nkj->n", m1.mT, m2)
    return qv

# The Second order term coming from Ito's applied to vector valued functions (by applying Ito's component wise)
def frobenius_inner_product_vec(latent_covariance: Tensor, decoder_hessian: Tensor) -> Tensor:
    """
        Batched version of the Frobenius inner product for matrix m x d and m x d. The inner product of A and B is
        the trace of A^T B.

    :param latent_covariance: a tensor of shape (n, m, d), for n batches of m x d matrices
    :param decoder_hessian: a tensor of shape (n, D, m, d), for n batches of m x d matrices
    :return: a tensor of shape (n, D)
    """
    qv = torch.einsum("njk,nrkj->nr", latent_covariance, decoder_hessian)
    return qv