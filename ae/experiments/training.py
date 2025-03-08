import os

import torch
import torch.nn as nn

from ae.experiments.datagen import ToyData
from ae.experiments.helpers import setup_experiment_dir
from ae.models.ambient_sdes import AmbientDriftNetwork, AmbientDiffusionNetwork
from ae.models.autoencoder import AutoEncoder
from ae.models.fitting import ThreeStageFit, fit_model
from ae.models.local_neural_sdes import LatentNeuralSDE, AutoEncoderDiffusion
from ae.models.losses import AmbientDriftLoss, AmbientDiffusionLoss
from ae.models.losses import LossWeights, LocalDiffusionLoss, LocalDriftLoss


class Trainer:
    def __init__(self, toy_data: ToyData, params: dict, device="cpu", anneal_tag="not_annealed"):
        self.toy_data = toy_data
        self.device = torch.device(device)
        self.params = params
        self.anneal_tag = anneal_tag
        self.exp_dir = self._setup_experiment()
        self._initialize_models()

    def _setup_experiment(self):
        base_dir = f"trained_models/{self.toy_data.surface.__class__.__name__}/" \
                   f"{self.toy_data.dynamics.__class__.__name__}"
        return setup_experiment_dir(self.params, base_dir, self.anneal_tag)

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
        self.losses = {
            "vanilla": weights_vanilla,
            "first": weights_first_order,
            "second": weights_second_order,
        }
        self.diffusion_loss = LocalDiffusionLoss("fro")
        self.drift_loss = LocalDriftLoss()

    def train(self, anneal_weights=None):
        fit3 = ThreeStageFit(self.params["lr"],
                             self.params["epochs_ae"],
                             self.params["epochs_diffusion"],
                             self.params["epochs_drift"],
                             self.params["weight_decay"],
                             self.params["batch_size"],
                             self.params["print_freq"])
        data = self.toy_data.generate_data(self.params["num_points"], self.params["intrinsic_dim"], device=self.device)

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
        fit_model(self.ambient_drift, AmbientDriftLoss(), data["x"], data["mu"], self.params["lr"],
                  self.params["epochs_drift"],
                  self.params["print_freq"], self.params["weight_decay"], self.params["batch_size"])
        fit_model(self.ambient_diffusion, AmbientDiffusionLoss(), data["x"], data["cov"], self.params["lr"],
                  self.params["epochs_drift"],
                  self.params["print_freq"], self.params["weight_decay"], self.params["batch_size"])

    def save_models(self):
        for model_type, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(self.exp_dir, f"ae_diffusion_{model_type}.pth"))
        torch.save(self.ambient_drift.state_dict(), os.path.join(self.exp_dir, "ambient_drift.pth"))
        torch.save(self.ambient_diffusion.state_dict(), os.path.join(self.exp_dir, "ambient_diffusion.pth"))
        print("Models successfully saved.")
