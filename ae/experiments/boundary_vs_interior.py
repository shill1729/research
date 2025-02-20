if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt

    # --- IMPORTS FROM CODE BASE ---
    from ae.symbolic.diffgeo import RiemannianManifold
    from ae.toydata.pointclouds import PointCloud
    from ae.toydata.local_dynamics import *
    from ae.toydata.surfaces import *
    from ae.models.autoencoder import AutoEncoder
    from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
    from ae.models.losses import LossWeights, TotalLoss, LocalDiffusionLoss, LocalDriftLoss
    from ae.utils import process_data
    from ae.models.fitting import ThreeStageFit, fit_model
    from ae.utils.performance_analysis import compute_test_losses


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
    batch_size = 15
    eps_max = 1.
    eps_grid_size = 10

    # The intrinsic and extrinsic dimensions.
    extrinsic_dim, intrinsic_dim = 3, 2
    hidden_dims = [16, 16]
    diffusion_layers = [16]
    drift_layers = [32]
    lr = 0.001
    weight_decay = 0.
    epochs_ae = 3000
    epochs_diffusion = 3000
    epochs_drift = 3000
    print_freq = 500
    tangent_angle_weight = 0.01
    tangent_drift_weight = 0.01
    diffeo_weight = 0.1
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()
    drift_act = nn.Tanh()
    diffusion_act = nn.Tanh()

    # -------------------------------------
    # CHOOSE THE SURFACE AND GET THE BOUNDS
    # -------------------------------------
    surface = WaveSurface()
    bounds = surface.bounds()  # native bounds in the local coordinates
    # For testing, we will enlarge these bounds by ε.

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
    # Process data: process_data returns projection matrices (p) and an orthonormal frame.
    x, mu, cov, p, _, orthonormal_frame = process_data(x, mu, cov, d=intrinsic_dim, return_frame=True, device=device)
    # (Keep local_x separately for interior vs. boundary checks)

    # -------------------------------------
    # SET UP MODELS WITH DIFFERENT LOSSES
    # -------------------------------------
    # Vanilla: no extra penalty.
    weights_vanilla = LossWeights()
    # First order: activate only tangent angle loss.
    weights_first = LossWeights()
    weights_first.diffeomorphism_reg = diffeo_weight
    weights_first.tangent_angle_weight = tangent_angle_weight
    # Second order: activate only tangent drift alignment loss.
    weights_second = LossWeights()
    weights_second.tangent_angle_weight = tangent_angle_weight
    weights_second.tangent_drift_weight = tangent_drift_weight
    weights_second.diffeomorphism_reg = diffeo_weight
    # Diffeomorphic model:
    weights_diffeo = LossWeights()
    weights_diffeo.diffeomorphism_reg = diffeo_weight

    # Instantiate the AutoEncoder + Neural SDE models.
    # Vanilla model:
    # TODO: pass the device call
    ae_vanilla = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde_vanilla = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers,
                                         drift_act, diffusion_act, encoder_act)
    ae_diffusion_vanilla = AutoEncoderDiffusion(latent_sde_vanilla, ae_vanilla)

    # First order model:
    ae_first = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde_first = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers,
                                       drift_act, diffusion_act, encoder_act)
    ae_diffusion_first = AutoEncoderDiffusion(latent_sde_first, ae_first)

    # Second order model:
    ae_second = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde_second = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers,
                                        drift_act, diffusion_act, encoder_act)
    ae_diffusion_second = AutoEncoderDiffusion(latent_sde_second, ae_second)

    # Diffeomorphic model:
    ae_diffeo = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde_diffeo = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers,
                                        drift_act, diffusion_act, encoder_act)
    ae_diffusion_diffeo = AutoEncoderDiffusion(latent_sde_diffeo, ae_diffeo)

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
    print("\nTraining diffeo model...")
    ae_diffusion_diffeo = fit3.three_stage_fit(ae_diffusion_diffeo,
                                               weights_diffeo,
                                               x, mu, cov, p, orthonormal_frame,
                                               norm, device)

    # -------------------------------------
    # Prepare a TotalLoss instance for testing (used for reconstruction loss)
    # -------------------------------------
    ae_loss = TotalLoss(weights_vanilla, norm)  # (loss modules inside TotalLoss)

    # -------------------------------------
    # Instantiate diffusion and drift loss modules (for testing)
    # -------------------------------------
    diffusion_loss_obj = LocalDiffusionLoss(norm)
    drift_loss_obj = LocalDriftLoss()

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
    # Helpers to compute diffusion and drift losses on a subset
    # -------------------------------------
    def compute_diff_and_drift(model: AutoEncoderDiffusion, x_subset, cov_subset, mu_subset):
        if x_subset.shape[0] == 0:
            return np.nan, np.nan
        dpi = model.autoencoder.encoder.jacobian_network(x_subset).detach()
        # Compute the encoded covariance: dpi * cov * dpi^T
        encoded_cov = torch.bmm(torch.bmm(dpi, cov_subset), dpi.transpose(1, 2))
        diff_loss = diffusion_loss_obj.forward(ae_diffusion=model, x=x_subset, encoded_cov=encoded_cov)
        drift_loss_val = drift_loss_obj.forward(ae_diffusion=model, x=x_subset, observed_ambient_drift=mu_subset)
        return diff_loss.item(), drift_loss_val.item()

    # Helper to compute reconstruction loss on a subset (using compute_test_losses)
    def subset_reconstruction_losses(model):
        if np.any(interior_mask):
            x_int = x_test[interior_mask]
            p_int = p_test[interior_mask]
            frame_int = orthonormal_frame_test[interior_mask]
            cov_int = cov_test[interior_mask]
            mu_int = mu_test[interior_mask]
            losses_int = compute_test_losses(model, ae_loss, x_int, p_int, frame_int, cov_int, mu_int, device=device)
            rec_int = losses_int["reconstruction loss"]
        else:
            rec_int = np.nan

        if np.any(boundary_mask):
            x_bnd = x_test[boundary_mask]
            p_bnd = p_test[boundary_mask]
            frame_bnd = orthonormal_frame_test[boundary_mask]
            cov_bnd = cov_test[boundary_mask]
            mu_bnd = mu_test[boundary_mask]
            losses_bnd = compute_test_losses(model, ae_loss, x_bnd, p_bnd, frame_bnd, cov_bnd, mu_bnd, device=device)
            rec_bnd = losses_bnd["reconstruction loss"]
        else:
            rec_bnd = np.nan

        return rec_int, rec_bnd

    # Helper to compute diffusion and drift losses on a subset
    def subset_diffusion_and_drift_losses(model):
        if np.any(interior_mask):
            x_int = x_test[interior_mask]
            cov_int = cov_test[interior_mask]
            mu_int = mu_test[interior_mask]
            diff_int, drift_int = compute_diff_and_drift(model, x_int, cov_int, mu_int)
        else:
            diff_int, drift_int = np.nan, np.nan

        if np.any(boundary_mask):
            x_bnd = x_test[boundary_mask]
            cov_bnd = cov_test[boundary_mask]
            mu_bnd = mu_test[boundary_mask]
            diff_bnd, drift_bnd = compute_diff_and_drift(model, x_bnd, cov_bnd, mu_bnd)
        else:
            diff_bnd, drift_bnd = np.nan, np.nan

        return diff_int, diff_bnd, drift_int, drift_bnd

    # -------------------------------------
    # VARY EPSILON TO GENERATE TEST DATA AND COMPUTE LOSSES
    # -------------------------------------
    epsilons = np.linspace(0.01, eps_max, eps_grid_size)  # e.g., 0.0, 0.1, ... 1.0
    # Lists for Reconstruction losses:
    errors_vanilla_interior, errors_vanilla_boundary = [], []
    errors_first_interior, errors_first_boundary = [], []
    errors_second_interior, errors_second_boundary = [], []
    errors_diffeo_interior, errors_diffeo_boundary = [], []
    # Lists for Diffusion losses:
    errors_vanilla_diff_interior, errors_vanilla_diff_boundary = [], []
    errors_first_diff_interior, errors_first_diff_boundary = [], []
    errors_second_diff_interior, errors_second_diff_boundary = [], []
    errors_diffeo_diff_interior, errors_diffeo_diff_boundary = [], []
    # Lists for Drift losses:
    errors_vanilla_drift_interior, errors_vanilla_drift_boundary = [], []
    errors_first_drift_interior, errors_first_drift_boundary = [], []
    errors_second_drift_interior, errors_second_drift_boundary = [], []
    errors_diffeo_drift_interior, errors_diffeo_drift_boundary = [], []

    k = 0
    for eps in epsilons:
        k += 1
        # Enlarge the bounds by ε for testing.
        current_bounds = [(b[0] - eps, b[1] + eps) for b in bounds]
        cloud_test = PointCloud(manifold, current_bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
        # Generate test data; note we also obtain the local coordinates (local_x_test)
        x_test, _, mu_test, cov_test, local_x_test = cloud_test.generate(num_test * k, seed=test_seed)
        # Process test data. process_data returns: x_test, mu_test, cov_test, p_test, _ and orthonormal_frame_test.
        x_test, mu_test, cov_test, p_test, _, orthonormal_frame_test = process_data(
            x_test, mu_test, cov_test, d=intrinsic_dim, return_frame=True, device=device
        )
        local_x_test = torch.tensor(local_x_test, dtype=torch.float32, device=device)
        # Determine interior vs. boundary using the local coordinates.
        interior_mask = is_interior_local(local_x_test, bounds)
        boundary_mask = ~interior_mask
        print("Interior set is = " + str(np.sum(interior_mask) / (num_test * k)) + " % of test data")
        print("Boundary set is = "+str(np.sum(boundary_mask)/(num_test*k))+" % of test data")
        # For each model, compute the losses on the interior and on the boundary.
        rec_vanilla_int, rec_vanilla_bnd = subset_reconstruction_losses(ae_diffusion_vanilla)
        rec_first_int, rec_first_bnd = subset_reconstruction_losses(ae_diffusion_first)
        rec_second_int, rec_second_bnd = subset_reconstruction_losses(ae_diffusion_second)
        rec_diffeo_int, rec_diffeo_bnd = subset_reconstruction_losses(ae_diffusion_diffeo)

        diff_vanilla_int, diff_vanilla_bnd, drift_vanilla_int, drift_vanilla_bnd = subset_diffusion_and_drift_losses(
            ae_diffusion_vanilla)
        diff_first_int, diff_first_bnd, drift_first_int, drift_first_bnd = subset_diffusion_and_drift_losses(
            ae_diffusion_first)
        diff_second_int, diff_second_bnd, drift_second_int, drift_second_bnd = subset_diffusion_and_drift_losses(
            ae_diffusion_second)
        diff_diffeo_int, diff_diffeo_bnd, drift_diffeo_int, drift_diffeo_bnd = subset_diffusion_and_drift_losses(
            ae_diffusion_diffeo)

        errors_vanilla_interior.append(rec_vanilla_int)
        errors_vanilla_boundary.append(rec_vanilla_bnd)
        errors_first_interior.append(rec_first_int)
        errors_first_boundary.append(rec_first_bnd)
        errors_second_interior.append(rec_second_int)
        errors_second_boundary.append(rec_second_bnd)
        errors_diffeo_interior.append(rec_diffeo_int)
        errors_diffeo_boundary.append(rec_diffeo_bnd)

        errors_vanilla_diff_interior.append(diff_vanilla_int)
        errors_vanilla_diff_boundary.append(diff_vanilla_bnd)
        errors_first_diff_interior.append(diff_first_int)
        errors_first_diff_boundary.append(diff_first_bnd)
        errors_second_diff_interior.append(diff_second_int)
        errors_second_diff_boundary.append(diff_second_bnd)
        errors_diffeo_diff_interior.append(diff_diffeo_int)
        errors_diffeo_diff_boundary.append(diff_diffeo_bnd)

        errors_vanilla_drift_interior.append(drift_vanilla_int)
        errors_vanilla_drift_boundary.append(drift_vanilla_bnd)
        errors_first_drift_interior.append(drift_first_int)
        errors_first_drift_boundary.append(drift_first_bnd)
        errors_second_drift_interior.append(drift_second_int)
        errors_second_drift_boundary.append(drift_second_bnd)
        errors_diffeo_drift_interior.append(drift_diffeo_int)
        errors_diffeo_drift_boundary.append(drift_diffeo_bnd)

        print(f"Epsilon {eps:.2f}:")
        print(f"  Vanilla - Interior Rec. Loss: {rec_vanilla_int:.4f}, Boundary Rec. Loss: {rec_vanilla_bnd:.4f}")
        print(f"  First   - Interior Rec. Loss: {rec_first_int:.4f}, Boundary Rec. Loss: {rec_first_bnd:.4f}")
        print(f"  Second  - Interior Rec. Loss: {rec_second_int:.4f}, Boundary Rec. Loss: {rec_second_bnd:.4f}")
        print(f"  Diffeo  - Interior Rec. Loss: {rec_diffeo_int:.4f}, Boundary Rec. Loss: {rec_diffeo_bnd:.4f}\n")
    # -------------------------------------
    # PLOT THE ERROR CURVES: 3 rows (Reconstruction, Diffusion, Drift) x 2 columns (Interior, Boundary)
    # -------------------------------------
    # errors_vanilla_interior = np.log(errors_vanilla_boundary)
    # errors_first_interior = np.log(errors_first_interior)
    # errors_second_interior = np.log(errors_second_interior)
    # errors_diffeo_interior = np.log(errors_diffeo_interior)
    #
    # errors_vanilla_boundary = np.log(errors_vanilla_boundary)
    # errors_first_boundary = np.log(errors_first_boundary)
    # errors_second_boundary = np.log(errors_second_boundary)
    # errors_diffeo_boundary = np.log(errors_diffeo_boundary)
    #
    # errors_vanilla_diff_interior = np.log(errors_vanilla_diff_interior)
    # errors_first_diff_interior = np.log(errors_first_diff_interior)
    # errors_second_diff_interior = np.log(errors_second_diff_interior)
    # errors_diffeo_diff_interior = np.log(errors_diffeo_diff_interior)
    #
    # errors_vanilla_diff_boundary = np.log(errors_vanilla_diff_boundary)
    # errors_first_diff_boundary = np.log(errors_first_diff_boundary)
    # errors_second_diff_boundary = np.log(errors_second_diff_boundary)
    # errors_diffeo_diff_boundary = np.log(errors_diffeo_diff_boundary)
    #
    # errors_vanilla_drift_interior = np.log(errors_vanilla_drift_interior)
    # errors_first_drift_interior = np.log(errors_first_drift_interior)
    # errors_second_drift_interior = np.log(errors_second_drift_interior)
    # errors_diffeo_drift_interior = np.log(errors_diffeo_drift_interior)
    #
    # errors_vanilla_drift_boundary = np.log(errors_vanilla_drift_boundary)
    # errors_first_drift_boundary = np.log(errors_first_drift_boundary)
    # errors_second_drift_boundary = np.log(errors_second_drift_boundary)
    # errors_diffeo_drift_boundary = np.log(errors_diffeo_drift_boundary)

    test_set_multiple_range = np.arange(1, eps_grid_size+1)*num_test
    plt.figure(figsize=(8, 8))
    # Row 1: Reconstruction Loss
    plt.subplot(3, 2, 1)
    plt.plot(test_set_multiple_range, errors_vanilla_interior, marker='o', label='Vanilla')
    plt.plot(test_set_multiple_range, errors_first_interior, marker='o', label='First Order')
    plt.plot(test_set_multiple_range, errors_second_interior, marker='o', label='Second Order')
    plt.plot(test_set_multiple_range, errors_diffeo_interior, marker='o', label='Diffeo')
    plt.xlabel('Size of total test set')
    plt.ylabel('Interior Rec. Loss')
    plt.title('Reconstruction Loss - Interior')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(epsilons, errors_vanilla_boundary, marker='o', label='Vanilla')
    plt.plot(epsilons, errors_first_boundary, marker='o', label='First Order')
    plt.plot(epsilons, errors_second_boundary, marker='o', label='Second Order')
    plt.plot(epsilons, errors_diffeo_boundary, marker='o', label='Diffeo')
    plt.xlabel('Epsilon')
    plt.ylabel('Boundary Rec. Loss')
    plt.title('Reconstruction Loss - Boundary')
    plt.legend()

    # Row 2: Diffusion Loss
    plt.subplot(3, 2, 3)
    plt.plot(test_set_multiple_range, errors_vanilla_diff_interior, marker='o', label='Vanilla')
    plt.plot(test_set_multiple_range, errors_first_diff_interior, marker='o', label='First Order')
    plt.plot(test_set_multiple_range, errors_second_diff_interior, marker='o', label='Second Order')
    plt.plot(test_set_multiple_range, errors_diffeo_diff_interior, marker='o', label='Diffeo')
    plt.xlabel('Size of total test set')
    plt.ylabel('Interior Diff. Loss')
    plt.title('Diffusion Loss - Interior')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(epsilons, errors_vanilla_diff_boundary, marker='o', label='Vanilla')
    plt.plot(epsilons, errors_first_diff_boundary, marker='o', label='First Order')
    plt.plot(epsilons, errors_second_diff_boundary, marker='o', label='Second Order')
    plt.plot(epsilons, errors_diffeo_diff_boundary, marker='o', label='Diffeo')
    plt.xlabel('Epsilon')
    plt.ylabel('Boundary Diff. Loss')
    plt.title('Diffusion Loss - Boundary')
    plt.legend()

    # Row 3: Drift Loss
    plt.subplot(3, 2, 5)
    plt.plot(test_set_multiple_range, errors_vanilla_drift_interior, marker='o', label='Vanilla')
    plt.plot(test_set_multiple_range, errors_first_drift_interior, marker='o', label='First Order')
    plt.plot(test_set_multiple_range, errors_second_drift_interior, marker='o', label='Second Order')
    plt.plot(test_set_multiple_range, errors_diffeo_drift_interior, marker='o', label='Diffeo')
    plt.xlabel('Size of total test set')
    plt.ylabel('Interior Drift Loss')
    plt.title('Drift Loss - Interior')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(epsilons, errors_vanilla_drift_boundary, marker='o', label='Vanilla')
    plt.plot(epsilons, errors_first_drift_boundary, marker='o', label='First Order')
    plt.plot(epsilons, errors_second_drift_boundary, marker='o', label='Second Order')
    plt.plot(epsilons, errors_diffeo_drift_boundary, marker='o', label='Diffeo')
    plt.xlabel('Epsilon')
    plt.ylabel('Boundary Drift Loss')
    plt.title('Drift Loss - Boundary')
    plt.legend()

    plt.tight_layout()
    plt.show()
