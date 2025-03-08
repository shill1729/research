import os
import json
import torch
from ae.experiments.datagen import ToyData
from ae.experiments.manifold_errors import GeometryError
from ae.experiments.sde_errors import DynamicsError
from ae.models.autoencoder import AutoEncoder
from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
from ae.models.ambient_sdes import AmbientDriftNetwork, AmbientDiffusionNetwork
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self, model_dir, device="cpu"):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.surface, self.dynamics = self._extract_surface_and_dynamics()
        self.config = self._load_config()
        self.toy_data = self._setup_toy_data()
        self.models = self._load_models()
        print(f"Loaded models for Surface: {self.surface}, Dynamics: {self.dynamics}")

    def _extract_surface_and_dynamics(self):
        """Extract surface and dynamics names from the directory path"""
        parts = self.model_dir.split(os.sep)  # Split path into components
        if len(parts) < 3:
            raise ValueError(f"Unexpected model directory structure: {self.model_dir}")
        return parts[-3], parts[-2]  # Extract surface and dynamics names

    def _load_config(self):
        """ Load experiment configuration from JSON """
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            return json.load(f)

    def _setup_toy_data(self):
        """ Dynamically initialize toy data object based on extracted names """
        from ae.toydata.surfaces import __dict__ as surface_classes
        from ae.toydata.local_dynamics import __dict__ as dynamics_classes

        # Get class dynamically from extracted names
        if self.surface not in surface_classes or self.dynamics not in dynamics_classes:
            raise ValueError(f"Unknown surface or dynamics: {self.surface}, {self.dynamics}")

        surface = surface_classes[self.surface]()
        dynamics = dynamics_classes[self.dynamics]()
        return ToyData(surface, dynamics)

    def _load_models(self):
        """ Load trained models from disk """
        extrinsic_dim = self.config["extrinsic_dim"]
        intrinsic_dim = self.config["intrinsic_dim"]
        hidden_dims = self.config["hidden_dims"]
        drift_layers = self.config["drift_layers"]
        diffusion_layers = self.config["diffusion_layers"]
        activation = torch.nn.Tanh()

        models = {}
        for model_type in ["vanilla", "first", "second"]:
            model_path = os.path.join(self.model_dir, f"ae_diffusion_{model_type}.pth")
            if not os.path.exists(model_path):
                print(f"Warning: Missing model file {model_path}")
                continue

            ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, activation, activation)
            latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, activation, activation,
                                         activation)
            model = AutoEncoderDiffusion(latent_sde, ae)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            models[model_type] = model

        # Load ambient networks
        for net_type in ["ambient_drift", "ambient_diffusion"]:
            net_class = AmbientDriftNetwork if net_type == "ambient_drift" else AmbientDiffusionNetwork
            net_path = os.path.join(self.model_dir, f"{net_type}.pth")
            if not os.path.exists(net_path):
                print(f"Warning: Missing model file {net_path}")
                continue

            network = net_class(extrinsic_dim, extrinsic_dim, drift_layers, activation)
            network.load_state_dict(torch.load(net_path, map_location=self.device))
            network.to(self.device)
            network.eval()
            models[net_type] = network

        return models

    def evaluate_models(self, eps_max=1., eps_grid_size=10, num_test=20000):
        """ Compute geometry and dynamics errors """
        geometry = GeometryError(self.toy_data, self, eps_max, self.device)
        geometry.compute_and_plot_errors(eps_grid_size, num_test, None, self.device)

        dynamics_error = DynamicsError(self.toy_data, self)
        gt, at, aes, _ = dynamics_error.sample_path_generator.generate_paths(0.02, 100, 50, None)
        dynamics_error.sample_path_plotter.plot_terminal_kernel_density(gt, at, aes)

        print("Evaluation completed successfully.")


if __name__ == "__main__":
    model_dir = "trained_models/Paraboloid/RiemannianBrownianMotion/trained_20250307-215128_h[16]_df[16]_dr[16]_lr0.001_epochs100_not_annealed"
    evaluator = ModelEvaluator(model_dir)
    evaluator.evaluate_models()
