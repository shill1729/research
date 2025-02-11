# We visualize the true sample paths against the model on the neural surface
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt
    # Toy data imports
    from ae.symbolic.diffgeo import RiemannianManifold
    from ae.toydata.pointclouds import PointCloud
    from ae.toydata.local_dynamics import *
    from ae.toydata.surfaces import *
    # Model imports
    from ae.models.autoencoder import AutoEncoder
    from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
    # Loss imports
    from ae.models.losses import LossWeights, TotalLoss, LocalDiffusionLoss, LocalDriftLoss
    # Pre-training/traing and performance imports
    from ae.utils import process_data
    from ae.models.fitting import ThreeStageFit
    from ae.utils.performance_analysis import plot_tangent_planes, compute_test_losses

    # Inputs
    device = torch.device("cpu")
    train_seed = None
    test_seed = None
    norm = "fro"
    embed = False
    R_seed = 123  # Set your seed for the random projection
    # torch.manual_seed(train_seed)
    # Point cloud parameters
    num_points = 30
    num_test = 100
    # Offset the boundary for point cloud
    epsilon = 0.01
    extrinsic_dim, intrinsic_dim = 3, 2
    hidden_dims = [16]
    diffusion_layers = [16]
    drift_layers = [16]
    lr = 0.001
    weight_decay = 0.
    # EPOCHS FOR TRAINING
    epochs_ae = 2000
    epochs_diffusion = 2000
    epochs_drift = 2000
    batch_size = 15
    print_freq = 100
    # PATHS PARAMETERS
    ntime = 1000
    npaths = 5
    tn = 2.5
    # weights for different penalties
    weights = LossWeights()
    weights.encoder_contraction_weight = 0.
    weights.decoder_contraction_weight = 0.
    weights.tangent_angle_weight = 0.
    weights.tangent_drift_weight = 0.
    weights.diffeomorphism_reg = 0.

    # Generate a fixed random matrix R with a seed
    torch.manual_seed(R_seed)

    D = 100  # Desired embedding dimension
    R = torch.randn(D, 3)  # Random matrix R of size (D, 3)

    # Activation functions
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()
    drift_act = nn.Tanh()
    diffusion_act = nn.Tanh()

    # Pick a surface and modify large bounds as needed
    surface = Paraboloid(2, 2)
    bounds = surface.bounds()
    large_bounds = [(b[0] - epsilon, b[1] + epsilon) for b in surface.bounds()]

    # Initialize the manifold
    manifold = RiemannianManifold(surface.local_coords(), surface.equation())
    dynamics = RiemannianBrownianMotion()

    # Another arbitrary diffusion
    local_drift = dynamics.drift(manifold)
    local_diffusion = dynamics.diffusion(manifold)

    # Generate the point cloud plus dynamics observations
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
    # returns points, weights, drifts, cov, local coord
    x, _, mu, cov, local_x = cloud.generate(num_points, seed=train_seed)
    x, mu, cov, p, _, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True, device=device)

    # TODO: refactor this into a function in utils.
    if embed:
        print("Embedding into " + str(D) + " dimensions")
        x = torch.matmul(x, R.T)
        mu = torch.matmul(mu, R.T)
        # Transform covariance for each sample in the batch
        cov = torch.einsum('ij,bjk,kl->bil', R, cov, R.T)  # Result: (n, D, D)
        x, mu, cov, p, _, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)
        extrinsic_dim = D

    ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
    latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, drift_act, diffusion_act, encoder_act)
    ae_diffusion = AutoEncoderDiffusion(latent_sde, ae)

    fit3 = ThreeStageFit(lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq)
    ae_diffusion = fit3.three_stage_fit(ae_diffusion, weights, x, mu, cov, p, orthonormal_frame, norm, device)

    # Generate test data:
    # Now all three components have been trained, we assess the performance on the test set.
    # returns points, weights, drifts, cov, local coord
    # Generate the point cloud plus dynamics observations
    # TODO this will fail on embed=True until the above is taken care of regarding embedding
    cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
    x_test, _, mu_test, cov_test, local_x_test = cloud.generate(num_test, seed=test_seed)
    x_test, mu_test, cov_test, p_test, _, orthonormal_frame_test = process_data(x_test,
                                                                                mu_test,
                                                                                cov_test,
                                                                                d=intrinsic_dim,
                                                                                return_frame=True,
                                                                                device=device)

    # Print post-training losses on the testing set for the AE
    print("\nTesting-set losses for the Autoencoder:")
    ae_loss = TotalLoss(weights, norm)
    diffusion_loss = LocalDiffusionLoss(norm)
    drift_loss = LocalDriftLoss()
    test_ae_losses = compute_test_losses(ae_diffusion,
                                         ae_loss,
                                         x_test,
                                         p_test,
                                         orthonormal_frame_test,
                                         cov_test,
                                         mu_test,
                                         device=device)
    for key, value in test_ae_losses.items():
        print(f"{key} = {value:.4f}")

    # Compute diffusion losses for testing set:
    dpi_test = ae.encoder.jacobian_network(x_test).detach()
    encoded_cov_test = torch.bmm(torch.bmm(dpi_test, cov_test), dpi_test.mT)
    diffusion_loss_test = diffusion_loss.forward(ae_diffusion=ae_diffusion,
                                                 x=x_test,
                                                 encoded_cov=encoded_cov_test
                                                 )
    drift_loss_test = drift_loss.forward(ae_diffusion=ae_diffusion,
                                         x=x_test,
                                         observed_ambient_drift=mu_test)
    print("\nSDE losses")
    print("Diffusion extrapolation loss = " + str(diffusion_loss_test.detach().numpy()))
    print("Drift extrapolation loss = " + str(drift_loss_test.detach().numpy()))

    # Plot tangent spaces
    x_test_hat = ae_diffusion.autoencoder.forward(x_test).detach()
    Hmodel = ae_diffusion.autoencoder.compute_orthonormal_frame(x_test).detach()
    plot_tangent_planes(x_test.detach(), x_test_hat, orthonormal_frame_test, Hmodel, resolution=8)

    # Plot SDEs
    z0_true = x[0, :2].detach()
    x0 = x[0, :]
    z0 = ae.encoder.forward(x0).detach()
    true_latent_paths = cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)
    model_latent_paths = latent_sde.sample_paths(z0, tn, ntime, npaths)
    true_ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
    model_ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
    for j in range(npaths):
        model_ambient_paths[j, :, :] = ae.decoder(
            torch.tensor(model_latent_paths[j, :, :], dtype=torch.float32)).detach().numpy()
        for i in range(ntime + 1):
            if embed:
                path3 = np.squeeze(cloud.np_phi(*true_latent_paths[j, i, :]))
                pathD = np.matmul(path3, R.T)
                true_ambient_paths[j, i, :] = pathD
            else:
                true_ambient_paths[j, i, :] = np.squeeze(cloud.np_phi(*true_latent_paths[j, i, :]))

    x_test = x_test.detach()
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    for i in range(npaths):
        ax.plot3D(true_ambient_paths[i, :, 0], true_ambient_paths[i, :, 1], true_ambient_paths[i, :, 2], c="black",
                  alpha=0.8)
        ax.plot3D(model_ambient_paths[i, :, 0], model_ambient_paths[i, :, 1], model_ambient_paths[i, :, 2], c="blue",
                  alpha=0.8)
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="Reconstruction")
    ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2])
    plt.show()

