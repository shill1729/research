import torch
import torch.nn as nn

from typing import List

from ae.models import FeedForwardNeuralNet


class AmbientDriftNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 drift_act: nn.Module,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Drift architecture
        drift_neurons = [input_dim] + hidden_dims + [output_dim]
        drift_acts = [drift_act] * len(hidden_dims) + [None]
        self.drift = FeedForwardNeuralNet(drift_neurons, drift_acts)

    def forward(self, z):
        return self.drift.forward(z)

    def drift_numpy(self, t, z):
        w = torch.tensor(z, dtype=torch.float32)
        return self.drift.forward(w).detach().numpy()


class AmbientDiffusionNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 diffusion_act: nn.Module,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Diffusion architecture
        diffusion_neurons = [input_dim] + hidden_dims + [output_dim*output_dim]
        diffusion_acts = [diffusion_act] * len(hidden_dims) + [None]
        self.diffusion = FeedForwardNeuralNet(diffusion_neurons, diffusion_acts)

    def forward(self, z):
        return self.diffusion.forward(z).view((z.size(0), self.output_dim, self.output_dim))

    def diffusion_numpy(self, t, z):
        w = torch.tensor(z, dtype=torch.float32)
        d = self.output_dim
        return self.diffusion.forward(w).view((d, d)).detach().numpy()
