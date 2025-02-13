# We visualize the true sample paths against the model on the neural surface
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt

    # --- IMPORTS FROM YOUR CODE BASE ---
    from ae.symbolic.diffgeo import RiemannianManifold
    from ae.toydata.pointclouds import PointCloud
    from ae.toydata.local_dynamics import *
    from ae.toydata.surfaces import *
    from ae.models.autoencoder import AutoEncoder
    from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
    from ae.models.losses import LossWeights, TotalLoss, LocalDiffusionLoss, LocalDriftLoss
    from ae.utils import process_data
    from ae.models.fitting import ThreeStageFit
    from ae.utils.performance_analysis import plot_tangent_planes, compute_test_losses

    # ---------------------------
    # Provided compute_test_losses
    # ---------------------------
    def compute_test_losses(ae_diffusion: AutoEncoderDiffusion,
                            ae_loss: TotalLoss,
                            x_test: torch.Tensor,
                            p_test: torch.Tensor,
                            frame_test: torch.Tensor,
                            cov_test: torch.Tensor,
                            mu_test: torch.Tensor,
                            device="cpu"):
        n, D, _ = p_test.size()
        normal_proj_test = torch.eye(D, device=device).expand(n, D, D) - p_test
        # Get reconstructed test data
        ae = ae_diffusion.autoencoder
        x_test_recon = ae.decoder(ae.encoder(x_test))

        # Compute test reconstruction error
        reconstruction_loss_test = ae_loss.reconstruction_loss.forward(x_test_recon, x_test).item()

        # Compute Jacobians and Hessians
        decoder_jacobian_test = ae.decoder_jacobian(ae.encoder(x_test))
        encoder_jacobian_test = ae.encoder_jacobian(x_test)
        decoder_hessian_test = ae.decoder_hessian(ae.encoder(x_test))

        # Contractive regularization
        contractive_loss_test = ae_loss.contractive_reg(encoder_jacobian_test).item()

        # Neural metric tensor
        metric_tensor_test = torch.bmm(decoder_jacobian_test.mT, decoder_jacobian_test)
        # Tangent error
        tangent_bundle_loss_test = ae_loss.tangent_bundle_reg.forward(decoder_jacobian_test,
                                                                      metric_tensor_test,
                                                                      p_test).item()
        # Drift alignment regularization
        drift_alignment_loss_test = ae_loss.drift_alignment_reg.forward(encoder_jacobian_test,
                                                                        decoder_hessian_test,
                                                                        cov_test,
                                                                        mu_test,
                                                                        normal_proj_test).item()

        # Diffeomorphism regularization 1
        diffeomorphism_loss_test = ae_loss.diffeomorphism_reg(decoder_jacobian_test, encoder_jacobian_test).item()
        decoder_contraction = ae_loss.contractive_reg(decoder_jacobian_test).item()
        tangent_angle_loss = ae_loss.tangent_angles_reg.forward(frame_test, decoder_jacobian_test, metric_tensor_test).item()

        # Return all the losses in a dictionary
        return {
            "reconstruction loss": reconstruction_loss_test,
            "encoder contractive loss": contractive_loss_test,
            "decoder contractive loss": decoder_contraction,
            "tangent bundle loss": tangent_bundle_loss_test,  # Note: redundant with tangent angle loss
            "tangent angle loss": tangent_angle_loss,
            "tangent drift alignment loss": drift_alignment_loss_test,
            "diffeomorphism loss": diffeomorphism_loss_test
        }

    # ---------------------------
    # SET HYPERPARAMETERS & SEEDS
    # ---------------------------
    device = torch.device("cpu")
    train_seed = None
    test_seed = None
    norm = "fro"
    # torch.manual_seed(train_seed)

    # Point cloud parameters
    num_points = 30
    num_test = 100

    # The intrinsic and extrinsic dimensions.
    extrinsic_dim, intrinsic_dim = 3, 2
    hidden_dims = [16]
    diffusion_layers = [16]
    drift_layers = [16]
    lr = 0.001
    weight_decay = 0.
    epochs_ae = 4000
    epochs_diffusion = 4000
    epochs_drift = 4000
    batch_size = 15
    print_freq = 100
    tangent_angle_weight = 0.01
    tangent_drift_weight = 0.01
    diffeo_weight = 0.1
    # -------------------------------------
    # CHOOSE THE SURFACE AND GET THE BOUNDS
    # -------------------------------------
    surface = Paraboloid(2, 2)
    bounds = surface.bounds()  # native bounds in the local coordinates
    # For testing, we will enlarge these bounds by ε.
    # (Note: bounds is a list of (low, high) tuples, one for each local coordinate.)

    # Initialize the manifold and dynamics.
    manifold = RiemannianManifold(surface.local_coords(), surface.equation())
    dynamics = RiemannianBrownianMotion()
    local_drift = dynamics.drift(manifold)
    local_diffusion = dynamics.diffusion(manifold)

    # -------------------------------------
    # GENERATE TRAINING DATA (POINT CLOUD)
    # -------------------------------------
    cloud_train = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
    # Generate: points, weights, drifts, covariances, and local coordinates.
    x, _, mu, cov, local_x = cloud_train.generate(num_points, seed=train_seed)
    # Process data: note that process_data returns the projection matrices (p) and orthonormal frame.
    x, mu, cov, p, _, orthonormal_frame = process_data(x, mu, cov, d=intrinsic_dim, return_frame=True, device=device)
    # (Keep local_x separately for interior vs. boundary checks)

    # -------------------------------------
    # SET UP THREE MODELS WITH DIFFERENT LOSSES
    # -------------------------------------
    # Vanilla: no extra penalty.
    weights_vanilla = LossWeights()
    # First order: activate only tangent angle loss.
    weights_first = LossWeights()
    weights_first.tangent_angle_weight = tangent_angle_weight  # adjust as needed
    # Second order: activate only tangent drift alignment loss.
    weights_second = LossWeights()
    weights_second.tangent_angle_weight = tangent_angle_weight
    weights_second.tangent_drift_weight = tangent_drift_weight  # adjust as needed
    weights_second.diffeomorphism_reg = diffeo_weight
    # Instantiate the AutoEncoder + Neural SDE models.
    # Vanilla model:
    ae_vanilla = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, nn.Tanh(), nn.Tanh())
    latent_sde_vanilla = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, nn.Tanh(), nn.Tanh(), nn.Tanh())
    ae_diffusion_vanilla = AutoEncoderDiffusion(latent_sde_vanilla, ae_vanilla)

    # First order model:
    ae_first = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, nn.Tanh(), nn.Tanh())
    latent_sde_first = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, nn.Tanh(), nn.Tanh(), nn.Tanh())
    ae_diffusion_first = AutoEncoderDiffusion(latent_sde_first, ae_first)

    # Second order model:
    ae_second = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, nn.Tanh(), nn.Tanh())
    latent_sde_second = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, nn.Tanh(), nn.Tanh(), nn.Tanh())
    ae_diffusion_second = AutoEncoderDiffusion(latent_sde_second, ae_second)

    # -------------------------------------
    # TRAIN EACH MODEL
    # -------------------------------------
    fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)

    print("Training vanilla model (no extra penalties)...")
    ae_diffusion_vanilla = fit3.three_stage_fit(ae_diffusion_vanilla,
                                                weights_vanilla,
                                                x, mu, cov, p, orthonormal_frame,
                                                norm, device)

    print("\nTraining first order model (tangent angle loss)...")
    ae_diffusion_first = fit3.three_stage_fit(ae_diffusion_first,
                                              weights_first,
                                              x, mu, cov, p, orthonormal_frame,
                                              norm, device)

    print("\nTraining second order model (tangent drift alignment loss)...")
    ae_diffusion_second = fit3.three_stage_fit(ae_diffusion_second,
                                               weights_second,
                                               x, mu, cov, p, orthonormal_frame,
                                               norm, device)

    # -------------------------------------
    # Prepare a TotalLoss instance for testing
    # -------------------------------------
    ae_loss = TotalLoss(weights_vanilla, norm)  # (the loss modules are inside TotalLoss)

    # -------------------------------------
    # Helper: determine which test samples are in the interior (training domain)
    # -------------------------------------
    def is_interior_local(local_coords: torch.Tensor, bounds_list):
        """
        local_coords: tensor of shape (n, intrinsic_dim) representing the local coordinates.
        bounds_list: list of (low, high) tuples for each intrinsic coordinate.
        Returns a boolean numpy array mask.
        """
        interior_mask = torch.ones(local_coords.shape[0], dtype=torch.bool, device=local_coords.device)
        for i, (low, high) in enumerate(bounds_list):
            interior_mask = interior_mask & (local_coords[:, i] >= low) & (local_coords[:, i] <= high)
        return interior_mask.cpu().numpy()

    # -------------------------------------
    # VARY EPSILON TO GENERATE TEST DATA AND COMPUTE ERRORS
    # -------------------------------------
    epsilons = np.linspace(0.0, 1, 12)  # for example, 0.0, 0.1, ... 0.5
    errors_vanilla_interior, errors_vanilla_boundary = [], []
    errors_first_interior, errors_first_boundary = [], []
    errors_second_interior, errors_second_boundary = [], []
    k = 0
    for eps in epsilons:
        k += 1
        # Enlarge the bounds by ε for testing.
        current_bounds = [(b[0] - eps, b[1] + eps) for b in bounds]
        cloud_test = PointCloud(manifold, current_bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
        # Generate test data; note we also obtain the local coordinates (local_x_test)
        x_test, _, mu_test, cov_test, local_x_test = cloud_test.generate(num_test*k, seed=test_seed)
        # Process test data. process_data returns: x_test, mu_test, cov_test, p_test (projection matrices),
        # _ (unused) and orthonormal_frame_test.
        x_test, mu_test, cov_test, p_test, _, orthonormal_frame_test = process_data(
            x_test, mu_test, cov_test, d=intrinsic_dim, return_frame=True, device=device
        )

        # Determine interior vs. boundary using the local coordinates (from cloud_test.generate)
        interior_mask = is_interior_local(local_x_test, bounds)
        boundary_mask = ~interior_mask

        # For each model, compute the losses on the interior and on the boundary separately.
        # We use compute_test_losses and extract the "reconstruction loss".
        def subset_losses(model):
            # Subset each tensor if there is at least one sample.
            if np.any(interior_mask):
                # Interior losses:
                x_int = x_test[interior_mask]
                p_int = p_test[interior_mask]
                frame_int = orthonormal_frame_test[interior_mask]
                cov_int = cov_test[interior_mask]
                mu_int = mu_test[interior_mask]
                losses_int = compute_test_losses(model, ae_loss, x_int, p_int, frame_int, cov_int, mu_int, device=device)
                int_loss = losses_int["reconstruction loss"]
            else:
                int_loss = np.nan

            if np.any(boundary_mask):
                # Boundary losses:
                x_bnd = x_test[boundary_mask]
                p_bnd = p_test[boundary_mask]
                frame_bnd = orthonormal_frame_test[boundary_mask]
                cov_bnd = cov_test[boundary_mask]
                mu_bnd = mu_test[boundary_mask]
                losses_bnd = compute_test_losses(model, ae_loss, x_bnd, p_bnd, frame_bnd, cov_bnd, mu_bnd, device=device)
                bnd_loss = losses_bnd["reconstruction loss"]
            else:
                bnd_loss = np.nan

            return int_loss, bnd_loss

        int_loss_vanilla, bnd_loss_vanilla = subset_losses(ae_diffusion_vanilla)
        int_loss_first,   bnd_loss_first   = subset_losses(ae_diffusion_first)
        int_loss_second,  bnd_loss_second  = subset_losses(ae_diffusion_second)

        errors_vanilla_interior.append(int_loss_vanilla)
        errors_vanilla_boundary.append(bnd_loss_vanilla)
        errors_first_interior.append(int_loss_first)
        errors_first_boundary.append(bnd_loss_first)
        errors_second_interior.append(int_loss_second)
        errors_second_boundary.append(bnd_loss_second)

        print(f"Epsilon {eps:.2f}:")
        print(f"  Vanilla - Interior Rec. Loss: {int_loss_vanilla:.4f}, Boundary Rec. Loss: {bnd_loss_vanilla:.4f}")
        print(f"  First   - Interior Rec. Loss: {int_loss_first:.4f}, Boundary Rec. Loss: {bnd_loss_first:.4f}")
        print(f"  Second  - Interior Rec. Loss: {int_loss_second:.4f}, Boundary Rec. Loss: {bnd_loss_second:.4f}\n")

    # -------------------------------------
    # PLOT THE ERROR CURVES
    # -------------------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epsilons, errors_vanilla_interior, marker='o', label='Vanilla')
    plt.plot(epsilons, errors_first_interior, marker='o', label='First Order')
    plt.plot(epsilons, errors_second_interior, marker='o', label='Second Order')
    plt.xlabel('Epsilon')
    plt.ylabel('Interior Reconstruction Loss')
    plt.title('Interior Error vs Epsilon')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epsilons, errors_vanilla_boundary, marker='o', label='Vanilla')
    plt.plot(epsilons, errors_first_boundary, marker='o', label='First Order')
    plt.plot(epsilons, errors_second_boundary, marker='o', label='Second Order')
    plt.xlabel('Epsilon')
    plt.ylabel('Boundary Reconstruction Loss')
    plt.title('Boundary Error vs Epsilon')
    plt.legend()

    plt.tight_layout()
    plt.show()
