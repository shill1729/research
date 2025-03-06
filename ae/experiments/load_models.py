
import torch
import os
import numpy as np
from ae.models.autoencoder import AutoEncoder
from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
from ae.models.ambient_sdes import AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.experiments.latent_vs_ambient_dynamics import hidden_dims, drift_layers, diffusion_layers, extrinsic_dim, intrinsic_dim
from ae.experiments.latent_vs_ambient_dynamics import encoder_act, decoder_act, drift_act, diffusion_act, surface, dynamics
# Define model structure (must match the original architecture)


# Initialize models with the same architecture
ae_vanilla = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
latent_sde_vanilla = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, drift_act, diffusion_act, encoder_act)
ae_diffusion_vanilla = AutoEncoderDiffusion(latent_sde_vanilla, ae_vanilla)

ae_first = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
latent_sde_first = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, drift_act, diffusion_act, encoder_act)
ae_diffusion_first = AutoEncoderDiffusion(latent_sde_first, ae_first)

ae_second = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
latent_sde_second = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, drift_act, diffusion_act, encoder_act)
ae_diffusion_second = AutoEncoderDiffusion(latent_sde_second, ae_second)

ae_diffeo = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act)
latent_sde_diffeo = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, drift_act, diffusion_act, encoder_act)
ae_diffusion_diffeo = AutoEncoderDiffusion(latent_sde_diffeo, ae_diffeo)

# Ambient networks
ambient_drift = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, drift_act)
ambient_diffusion = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diffusion_layers, diffusion_act)

# Load model weights
surface_name = surface.__class__.__name__
dynamics_name = dynamics.__class__.__name__
print("Current setting in latent_vs_ambient_dynamics.py = ")
print(surface_name)
print(dynamics_name)
save_dir = "trained_models/Paraboloid/LangevinHarmonicOscillator/trained_20250306-150817_h[32, 32]_df[8]_dr[8]_lr0.001_epochs9000"
ae_diffusion_vanilla.load_state_dict(torch.load(os.path.join(save_dir, "ae_diffusion_vanilla.pth")))
ae_diffusion_first.load_state_dict(torch.load(os.path.join(save_dir, "ae_diffusion_first.pth")))
ae_diffusion_second.load_state_dict(torch.load(os.path.join(save_dir, "ae_diffusion_second.pth")))
# ae_diffusion_diffeo.load_state_dict(torch.load(os.path.join(save_dir, "ae_diffusion_diffeo.pth")))

ambient_drift.load_state_dict(torch.load(os.path.join(save_dir, "ambient_drift.pth")))
ambient_diffusion.load_state_dict(torch.load(os.path.join(save_dir, "ambient_diffusion.pth")))

print("Models successfully loaded and ready for analysis.")

