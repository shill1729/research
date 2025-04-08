import os
import json
import inspect
import torch
import torch.nn as nn
from pathlib import Path

from ae.toydata.datagen import ToyData
from ae.experiments.training.helpers import setup_experiment_dir
from ae.models.ambient_sdes import AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.autoencoder import AutoEncoder
from ae.models.fitting import ThreeStageFit, fit_model
from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
from ae.models.losses import AmbientDriftLoss, AmbientDiffusionLoss
from ae.models.losses import LossWeights, LocalDiffusionLoss, LocalDriftLoss
from ae.toydata.surfaces import SurfaceBase
from ae.toydata.local_dynamics import DynamicsBase


class Trainer:
    def __init__(self, toy_data: ToyData, params: dict, device="cpu", anneal_tag="not_annealed", embed=False):
        self.toy_data = toy_data
        self.device = torch.device(device)
        self.params = params
        self.anneal_tag = anneal_tag
        self.exp_dir = None
        self.embed = embed
        self._initialize_models()


    def _setup_experiment(self):
        base_dir = f"examples/surfaces/trained_models/{self.toy_data.surface.__class__.__name__}/" \
                   f"{self.toy_data.dynamics.__class__.__name__}"
        return setup_experiment_dir(self.params, base_dir, self.anneal_tag)
    # def _setup_experiment(self):
    #     # Absolute path to the project root (2 levels up from this file)
    #     root_dir = Path(__file__).resolve().parents[1]
    #
    #     base_dir = root_dir / "examples" / "surfaces" / "trained_models" / \
    #                self.toy_data.surface.__class__.__name__ / \
    #                self.toy_data.dynamics.__class__.__name__
    #
    #     return setup_experiment_dir(self.params, base_dir, self.anneal_tag)

    def _initialize_models(self):
        extrinsic_dim, intrinsic_dim = self.params["extrinsic_dim"], self.params["intrinsic_dim"]
        hidden_dims, diffusion_layers, drift_layers = self.params["hidden_dims"], self.params["diffusion_layers"], \
        self.params["drift_layers"]
        activation = nn.Tanh()

        self.models = {}
        for model_type in ["vanilla", "first", "second"]:
            ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, activation, activation)
            latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diffusion_layers, activation, activation,
                                         activation)
            self.models[model_type] = AutoEncoderDiffusion(latent_sde, ae)

        self.ambient_drift = AmbientDriftNetwork(extrinsic_dim, extrinsic_dim, drift_layers, activation)
        self.ambient_diffusion = AmbientDiffusionNetwork(extrinsic_dim, extrinsic_dim, diffusion_layers, activation)
        weights_vanilla = LossWeights()
        weights_first_order = LossWeights(diffeomorphism_reg=self.params["diffeo_weight"],
                                          tangent_angle_weight=self.params["tangent_angle_weight"])
        weights_second_order = LossWeights(diffeomorphism_reg=self.params["diffeo_weight"],
                                           tangent_angle_weight=self.params["tangent_angle_weight2"],
                                           tangent_drift_weight=self.params["tangent_drift_weight"])
        print("Second order model")
        print(weights_second_order)
        self.losses = {
            "vanilla": weights_vanilla,
            "first": weights_first_order,
            "second": weights_second_order,
        }
        self.diffusion_loss = LocalDiffusionLoss("fro")
        self.drift_loss = LocalDriftLoss()

    def train(self, anneal_weights=None):
        self.exp_dir = self._setup_experiment()
        fit3 = ThreeStageFit(self.params["lr"],
                             self.params["epochs_ae"],
                             self.params["epochs_diffusion"],
                             self.params["epochs_drift"],
                             self.params["weight_decay"],
                             self.params["batch_size"],
                             self.params["print_freq"])
        data = self.toy_data.generate_data(self.params["num_points"], self.params["intrinsic_dim"], device=self.device,
                                           embed=self.embed)

        for model_type, model in self.models.items():
            print(f"Training {model_type} model...")
            if model_type != "second":
                anneal_weights_to_use = None
            else:
                anneal_weights_to_use = anneal_weights
            self.models[model_type] = fit3.three_stage_fit(model, self.losses[model_type],
                                                           data["x"], data["mu"], data["cov"], data["p"],
                                                           data["orthonormal_frame"],
                                                           anneal_weights_to_use, self.params["norm"], self.device)
        print("Training ambient networks...")
        x = data["x"]
        # z = self.models["vanilla"].autoencoder.encoder(x).detach()
        print("Training Ambient drift:")
        fit_model(self.ambient_drift, AmbientDriftLoss(), x, data["mu"], self.params["lr"],
                  self.params["epochs_drift"],
                  self.params["print_freq"], self.params["weight_decay"], self.params["batch_size"])
        print("Training Ambient diffusion/covariance:")
        fit_model(self.ambient_diffusion, AmbientDiffusionLoss(), x, data["cov"], self.params["lr"],
                  self.params["epochs_drift"],
                  self.params["print_freq"], self.params["weight_decay"], self.params["batch_size"])

    def save_models(self):
        for model_type, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(self.exp_dir, f"ae_diffusion_{model_type}.pth"))
        torch.save(self.ambient_drift.state_dict(), os.path.join(self.exp_dir, "ambient_drift.pth"))
        torch.save(self.ambient_diffusion.state_dict(), os.path.join(self.exp_dir, "ambient_diffusion.pth"))
        print("Models successfully saved.")

    @classmethod
    def load_from_pretrained(cls, pretrained_dir, device="cpu", large_dim=None):
        """
        Load a Trainer from a pretrained directory.

        The directory must follow the structure:
        `trained_models/{surface}/{dynamics}/trained_...`

        Args:
            pretrained_dir (str): Path to the directory containing the pretrained model.
            device (str or torch.device): Device to load the model onto.

        Returns:
            Trainer: A Trainer instance initialized with the pretrained weights.
        """
        # Extract surface and dynamics from the directory structure
        path_parts = pretrained_dir.split(os.sep)
        if len(path_parts) < 3:
            raise ValueError("Pretrained directory structure should be 'trained_models/{surface}/{dynamics}/...'")

        surface_name = path_parts[-3]
        dynamics_name = path_parts[-2]

        # Load the configuration file
        config_path = os.path.join(pretrained_dir, "config.json")
        with open(config_path, "r") as f:
            params = json.load(f)

        # Dynamically instantiate the correct ToyData object
        toy_data = cls._instantiate_toy_data(surface_name, dynamics_name, params, large_dim)

        # Initialize the trainer
        trainer = cls(toy_data, params, device)
        trainer.exp_dir = pretrained_dir  # Override experiment directory

        # Load model weights
        trainer._load_model_weights(pretrained_dir)

        print(f"Successfully loaded model from {pretrained_dir}")
        return trainer


    @staticmethod
    def _instantiate_toy_data(surface_name, dynamics_name, params, large_dim=None):
        """
        Dynamically creates a ToyData instance based on surface and dynamics class names.

        Args:
            surface_name (str): The name of the surface class.
            dynamics_name (str): The name of the dynamics class.
            params (dict): Model parameters (which may be useful for initialization).

        Returns:
            ToyData: An instance of the appropriate ToyData class.
        """
        from ae.toydata.surfaces import __dict__ as surfaces_dict
        from ae.toydata.local_dynamics import __dict__ as dynamics_dict

        # Find the correct surface class
        surface_class = next(
            (cls for name, cls in surfaces_dict.items() if
             name == surface_name and inspect.isclass(cls) and issubclass(cls, SurfaceBase)),
            None
        )

        # Find the correct dynamics class
        dynamics_class = next(
            (cls for name, cls in dynamics_dict.items() if
             name == dynamics_name and inspect.isclass(cls) and issubclass(cls, DynamicsBase)),
            None
        )

        if surface_class is None:
            raise ValueError(f"Unknown surface: {surface_name}. Ensure it exists in 'surfaces.py'.")
        if dynamics_class is None:
            raise ValueError(f"Unknown dynamics: {dynamics_name}. Ensure it exists in 'dynamics.py'.")

        # Instantiate the correct surface and dynamics classes
        surface = surface_class()
        dynamics = dynamics_class()
        # dynamics = dynamics_class(surface, **params.get("dynamics_params", {}))

        return ToyData(surface, dynamics, large_dim)

    def _load_model_weights(self, pretrained_dir):
        """
        Loads the model weights from the saved state dictionaries.

        Args:
            pretrained_dir (str): Directory containing the pretrained model weights.
        """
        for model_type in ["vanilla", "first", "second"]:
            model_path = os.path.join(pretrained_dir, f"ae_diffusion_{model_type}.pth")
            self.models[model_type].load_state_dict(torch.load(model_path, map_location=self.device))

        # Load ambient networks
        self.ambient_drift.load_state_dict(
            torch.load(os.path.join(pretrained_dir, "ambient_drift.pth"), map_location=self.device)
        )
        self.ambient_diffusion.load_state_dict(
            torch.load(os.path.join(pretrained_dir, "ambient_diffusion.pth"), map_location=self.device)
        )


if __name__ == "__main__":
    from ae.experiments.errors.manifold_errors import GeometryError
    from ae.experiments.errors.sde_errors import DynamicsError
    eps_grid_size= 10
    num_test = 20000
    tn = 0.5
    ntime = 1000
    npaths = 30
    model_dir = "../examples/surfaces/trained_models/Paraboloid/LangevinHarmonicOscillator/trained_20250307-231744_h[16]_df[16]_dr[16]_lr0.001_epochs9000_not_annealed"
    trainer = Trainer.load_from_pretrained(model_dir)
    device = "cpu"
    geometry = GeometryError(trainer.toy_data, trainer, 1., device)
    geometry.compute_and_plot_errors(eps_grid_size, num_test, None, device)
    dynamics_error = DynamicsError(trainer.toy_data, trainer)
    gt, at, aes, gt_local, aes_local = dynamics_error.sample_path_generator.generate_paths(tn, ntime, npaths, None)
    # View the ambient sample paths
    dynamics_error.sample_path_plotter.plot_ambient_sample_paths(gt, aes, at)
    # View first step kernel density:
    dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, False)
    # View the terminal kernel densities:
    dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, True)
    dynamics_error.sample_path_plotter.plot_sample_paths(gt_local, aes_local, None, False)
    dynamics_error.sample_path_plotter.plot_kernel_density(gt_local, aes_local, None, False)
    dynamics_error.sample_path_plotter.plot_kernel_density(gt_local, aes_local, None, True)
