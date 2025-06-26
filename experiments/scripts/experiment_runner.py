# experiment_runner.py
import os
import torch
from experiment_config import ExperimentConfig
from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion, AmbientDriftNetwork, AmbientDiffusionNetwork, ThreeStageFit, fit_model
from ae.toydata import PointCloud
from ae.toydata.local_dynamics import *
from ae.utils import process_data
from ae.models.losses.losses_autoencoder import LossWeights
from ae.models.losses.losses_ambient import AmbientDriftLoss, AmbientCovarianceLoss

ACTIVATION_MAP = {
    # Basic activations
    'Tanh': torch.nn.Tanh,
    'ReLU': torch.nn.ReLU,
    'Sigmoid': torch.nn.Sigmoid,
    'LeakyReLU': torch.nn.LeakyReLU,

    # Additional activations
    'ELU': torch.nn.ELU,
    'CELU': torch.nn.CELU,
    'SELU': torch.nn.SELU,
    'GELU': torch.nn.GELU,
    'Hardshrink': torch.nn.Hardshrink,
    'Hardtanh': torch.nn.Hardtanh,
    'Hardswish': torch.nn.Hardswish,
    'Hardsigmoid': torch.nn.Hardsigmoid,
    'LogSigmoid': torch.nn.LogSigmoid,
    'MultiheadAttention': torch.nn.MultiheadAttention,
    'PReLU': torch.nn.PReLU,
    'ReLU6': torch.nn.ReLU6,
    'RReLU': torch.nn.RReLU,
    'SiLU': torch.nn.SiLU,  # Also known as Swish
    'Mish': torch.nn.Mish,
    'Softplus': torch.nn.Softplus,
    'Softshrink': torch.nn.Softshrink,
    'Softsign': torch.nn.Softsign,
    'Tanhshrink': torch.nn.Tanhshrink,
    'Threshold': torch.nn.Threshold,
    'GLU': torch.nn.GLU,

    # Normalization modules that are sometimes used in activation contexts
    'LayerNorm': torch.nn.LayerNorm,
    'LocalResponseNorm': torch.nn.LocalResponseNorm,
    'CrossMapLRN2d': torch.nn.CrossMapLRN2d,

    # Dropouts (included as they're often used alongside activations)
    'Dropout': torch.nn.Dropout,
    'Dropout2d': torch.nn.Dropout2d,
    'Dropout3d': torch.nn.Dropout3d,
    'AlphaDropout': torch.nn.AlphaDropout,
    'FeatureAlphaDropout': torch.nn.FeatureAlphaDropout
}

def build_activation(name):
    return ACTIVATION_MAP[name]()

def run_full_experiment(config: ExperimentConfig, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    config.save(output_dir)

    # === Setup toy data ===
    curve = eval(config.manifold_class)()
    dynamics = eval(config.dynamics_class)()
    manifold = RiemannianManifold(curve.local_coords(), curve.equation())
    local_drift = dynamics.drift(manifold)
    local_diffusion = dynamics.diffusion(manifold)

    point_cloud = PointCloud(manifold, curve.bounds(), local_drift, local_diffusion, True)
    x, _, mu, cov, local_x = point_cloud.generate(n=config.n_train, seed=config.seed)
    x, mu, cov, p, n, h = process_data(x, mu, cov, d=config.intrinsic_dim)

    # === Fit AE-SDE with 0th, 1st, and 2nd order penalties ===
    for order, (fo_w, so_w, diffeo_w) in enumerate([
        (0, 0, 0),
        (config.loss_weights['tangent_angle_weight'], 0, config.loss_weights['diffeomorphism_reg']),
        (config.loss_weights['tangent_angle_weight'], config.loss_weights['tangent_drift_weight'], config.loss_weights['diffeomorphism_reg'])
    ]):
        ae = AutoEncoder(
            extrinsic_dim=config.extrinsic_dim,
            intrinsic_dim=config.intrinsic_dim,
            hidden_dims=config.hidden_dims,
            encoder_act=build_activation(config.encoder_activation),
            decoder_act=build_activation(config.decoder_activation)
        )

        latent_sde = LatentNeuralSDE(
            config.intrinsic_dim,
            config.drift_layers,
            config.diff_layers,
            build_activation(config.drift_activation),
            build_activation(config.diffusion_activation)
        )

        model = AutoEncoderDiffusion(latent_sde, ae)
        weights = LossWeights(
            tangent_angle_weight=fo_w,
            tangent_drift_weight=so_w,
            diffeomorphism_reg=diffeo_w
        )

        fit = ThreeStageFit(
            config.lr, config.epochs['ae'], config.epochs['diffusion'], config.epochs['drift'],
            config.weight_decay, config.batch_size, 1000
        )
        fit.three_stage_fit(model, weights, x, mu, cov, p, h)
        torch.save({
            'ae_state_dict': model.autoencoder.state_dict(),
            'sde_state_dict': model.latent_sde.state_dict()
        }, os.path.join(output_dir, f"ae_sde_order{order}.pth"))

    # === Fit Euclidean SDE ===
    ambient_drift_model = AmbientDriftNetwork(
        input_dim=config.extrinsic_dim,
        output_dim=config.extrinsic_dim,
        hidden_dims=config.drift_layers,
        drift_act=build_activation(config.drift_activation)
    )
    ambient_diff_model = AmbientDiffusionNetwork(
        input_dim=config.extrinsic_dim,
        output_dim=config.extrinsic_dim,
        hidden_dims=config.diff_layers,
        diff_act=build_activation(config.diffusion_activation)
    )
    ambient_drift_loss = AmbientDriftLoss()
    ambient_cov_loss = AmbientCovarianceLoss()

    print("\nTraining ambient drift model")
    fit_model(ambient_drift_model, ambient_drift_loss, x, mu,
              lr=config.lr,
              epochs=config.epochs['drift'],
              print_freq=1000,
              weight_decay=config.weight_decay,
              batch_size=config.batch_size)

    print("\nTraining ambient diffusion model")
    fit_model(ambient_diff_model, ambient_cov_loss, x, cov,
              lr=config.lr,
              epochs=config.epochs['diffusion'],
              print_freq=1000,
              weight_decay=config.weight_decay,
              batch_size=config.batch_size)



    torch.save(ambient_drift_model.state_dict(), os.path.join(output_dir, "euclidean_drift.pth"))
    torch.save(ambient_diff_model.state_dict(), os.path.join(output_dir, "euclidean_diffusion.pth"))
