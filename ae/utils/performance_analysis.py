import torch
from torch import Tensor
import matplotlib.pyplot as plt

from ae.models.sdes_latent import AutoEncoderDiffusion
from ae.models.losses.losses_autoencoder import TotalLoss


def compute_test_losses(ae_diffusion: AutoEncoderDiffusion,
                        ae_loss: TotalLoss,
                        x_test: Tensor,
                        p_test: Tensor,
                        frame_test: Tensor,
                        cov_test: Tensor,
                        mu_test: Tensor,
                        device="cpu"):
    n, D, _ = p_test.size()
    normal_proj_test = torch.eye(D, device=device).expand(n, D, D) - p_test
    # Get reconstructed test data
    ae = ae_diffusion.autoencoder
    x_test_recon = ae.decoder(ae.encoder(x_test))

    # Compute test reconstruction error
    reconstruction_loss_test = torch.sqrt(ae_loss.reconstruction_loss(x_test_recon, x_test)).item()

    # Compute Jacobians and Hessians
    decoder_jacobian_test = ae.decoder_jacobian(ae.encoder(x_test))
    encoder_jacobian_test = ae.encoder_jacobian(x_test)
    decoder_hessian_test = ae.decoder_hessian(ae.encoder(x_test))

    # Contractive regularization
    contractive_loss_test = ae_loss.contractive_reg(encoder_jacobian_test).item()

    # Neural metric tensor
    metric_tensor_test = torch.bmm(decoder_jacobian_test.mT, decoder_jacobian_test)
    # Tangent error
    tangent_bundle_loss_test = ae_loss.tangent_bundle_reg(decoder_jacobian_test,
                                                          metric_tensor_test,
                                                          p_test).item()
    # Drift alignment regularization: old syntax
    # drift_alignment_loss_test = ae_loss.drift_alignment_reg.forward(encoder_jacobian_test,
    #                                                                 decoder_hessian_test,
    #                                                                 cov_test,
    #                                                                 mu_test,
    #                                                                 normal_proj_test,
    #                                                                 decoder_jacobian_test).item()
    # Drift alignment regularization: new syntax
    drift_alignment_loss_test = ae_loss.drift_alignment_reg(encoder_jacobian_test,
                                                                    decoder_hessian_test,
                                                                    cov_test,
                                                                    mu_test,
                                                                    frame_test).item()

    # Diffeomorphism regularization 1
    diffeomorphism_loss_test = ae_loss.diffeomorphism_reg(decoder_jacobian_test, encoder_jacobian_test).item()
    decoder_contraction = ae_loss.contractive_reg(decoder_jacobian_test).item()
    tangent_angle_loss = ae_loss.tangent_angles_reg.forward(frame_test, decoder_jacobian_test, metric_tensor_test).item()

    # Return all the losses in a dictionary
    return {
        "reconstruction loss": reconstruction_loss_test,
        "encoder contractive loss": contractive_loss_test,
        "decoder contractive loss": decoder_contraction,
        "tangent bundle loss": tangent_bundle_loss_test,  # Note this is redundant--it is equal to tangent angle loss
        "tangent angle loss": tangent_angle_loss,
        "tangent drift alignment loss": drift_alignment_loss_test,
        "diffeomorphism loss": diffeomorphism_loss_test
    }


def plot_tangent_planes(x, xhat, H, Hmodel, delta=0.05, resolution=10):
    """
    Plot the tangent planes at each point in the point cloud.

    Parameters:
        x: Tensor of shape (n, D), the point cloud.
        P: Tensor of shape (n, D, D), the bundle of orthogonal projections.
        H: Tensor of shape (n, D, d), the tangent frames.
        delta: Scalar, half-width of the tangent plane patches.
        resolution: Integer, resolution of each tangent plane patch.
    """
    n, D = x.shape
    grid = torch.linspace(-delta, delta, resolution)
    s, t = torch.meshgrid(grid, grid, indexing='ij')  # Shape: (num_points, num_points)

    # Flatten grid to compute plane points
    st = torch.stack([s.flatten(), t.flatten()], dim=-1).T  # Shape: (2, num_points**2)

    # Initialize 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through each point in the point cloud
    for i in range(n):
        # Point x[i]
        xi = x[i]  # Shape: (D,)
        zi = xhat[i]
        # Tangent frame H[i]
        Hi = H[i]  # Shape: (D, d)

        Hmodeli = Hmodel[i]

        # True Tangent plane patch: x[i] + Hi @ (s, t)
        patch = xi.unsqueeze(1) + Hi @ st  # Shape: (D, num_points**2)
        patch = patch.T  # Shape: (num_points**2, D)

        # Model Tangent plane patch: x[i] + Hmodeli @ (s, t)
        patch1 = zi.unsqueeze(1) + Hmodeli @ st  # Shape: (D, resolution**2)
        patch1 = patch1.T  # Shape: (resolution**2, D)

        # Reshape for plotting
        patch = patch.view(resolution, resolution, D)  # Shape: (resolution, resolution, D)
        patch1 = patch1.view(resolution, resolution, D)  # Shape: (resolution, resolution, D)

        # Plot the patch
        ax.plot_surface(
            patch[..., 0].cpu().numpy(),
            patch[..., 1].cpu().numpy(),
            patch[..., 2].cpu().numpy(),
            color='b', alpha=0.5
        )
        # Plot the patch
        ax.plot_surface(
            patch1[..., 0].cpu().numpy(),
            patch1[..., 1].cpu().numpy(),
            patch1[..., 2].cpu().numpy(),
            color='r', alpha=0.5
        )

    # Scatter plot of original points
    ax.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), x[:, 2].cpu().numpy(), color='b', s=5)
    ax.scatter(xhat[:, 0].cpu().numpy(), xhat[:, 1].cpu().numpy(), xhat[:, 2].cpu().numpy(), color='r', s=5)
    # Set axis labels and aspect ratio
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.legend(["True", "Model"])
    plt.show()