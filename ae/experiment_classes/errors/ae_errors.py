import torch

from ae.models.sdes_latent import ambient_quadratic_variation_drift

def compute_all_losses_for_model(model, x, mu, cov, p, intrinsic_dim):
    """
    Compute reconstruction and geometric losses for a single AE-SDE model.
    """
    losses = {}

    # 1. Reconstruction
    z = model.autoencoder.encoder(x)
    x_recon = model.autoencoder.decoder(z)
    losses["Reconstruction"] = torch.mean(torch.linalg.vector_norm(x_recon - x, ord=2, dim=1) ** 2).item()

    # 2. Tangent projection
    p_model = model.autoencoder.neural_orthogonal_projection(z)
    losses["Tangent penalty"] = torch.mean(torch.linalg.matrix_norm(p_model - p, ord="fro") ** 2).item()

    # 3. Curvature drift (Ito penalty)
    dpi = model.autoencoder.encoder_jacobian(x)
    decoder_hessian = model.autoencoder.decoder_hessian(z)
    latent_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
    q_hat = ambient_quadratic_variation_drift(latent_cov, decoder_hessian)
    model_tangent_drift = mu - 0.5 * q_hat
    normal_component = model_tangent_drift - torch.bmm(p, model_tangent_drift.unsqueeze(-1)).squeeze(-1)
    losses["Ito penalty"] = torch.mean(torch.linalg.vector_norm(normal_component, ord=2, dim=1) ** 2).item()

    # 4. Decoder SVs
    dphi = model.autoencoder.decoder_jacobian(z)
    losses["Min Smallest Decoder SV"] = torch.min(torch.linalg.matrix_norm(dphi, ord=-2)).item()
    losses["Max Largest Decoder SV"] = torch.max(torch.linalg.matrix_norm(dphi, ord=2)).item()

    # 5. Encoder SVs
    losses["Min Smallest Encoder SV"] = torch.min(torch.linalg.matrix_norm(dpi, ord=-2)).item()
    losses["Max Largest Encoder SV"] = torch.max(torch.linalg.matrix_norm(dpi, ord=2)).item()

    # 6. Moore–Penrose error
    g = model.autoencoder.neural_metric_tensor(z)
    g_inv = torch.linalg.inv(g)
    dphi_inv = torch.bmm(g_inv, dphi.mT)
    losses["Moore-Penrose error"] = torch.mean(torch.linalg.matrix_norm(dpi - dphi_inv, ord="fro") ** 2).item()

    # 7. Diffeo penalty
    jacob_prod = torch.bmm(dpi, dphi)
    eye = torch.eye(dpi.shape[1], device=dpi.device).expand(dpi.shape[0], -1, -1)
    losses["Diffeomorphism Error"] = torch.mean(torch.linalg.matrix_norm(jacob_prod - eye, ord="fro") ** 2).item()

    # 8. Ambient covariance error
    model_latent_cov = model.compute_local_covariance(x)
    model_ambient_cov = torch.bmm(dphi, torch.bmm(model_latent_cov, dphi.mT))
    ambient_cov_error = torch.linalg.matrix_norm(model_ambient_cov - cov, ord="fro") ** 2
    losses["Ambient Cov Errors"] = torch.mean(ambient_cov_error).item()

    # 9. Ambient drift error (extrinsic form from latent)
    model_latent_drift = model.compute_local_drift(x)
    model_ambient_drift = torch.bmm(dphi, model_latent_drift.unsqueeze(-1)).squeeze(-1) + 0.5 * q_hat
    drift_err = torch.linalg.vector_norm(model_ambient_drift - mu, ord=2, dim=1) ** 2
    losses["Ambient Drift Errors"] = torch.mean(drift_err).item()

    # 10. Ambient drift error (direct computation)
    model_ambient_drift = model.compute_ambient_drift(x, cov)
    drift_err2 = torch.linalg.vector_norm(model_ambient_drift - mu, ord=2, dim=1) ** 2
    losses["Ambient Drift 2"] = torch.mean(drift_err2).item()

    return losses
