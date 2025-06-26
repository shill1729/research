# scripts/experiment_config.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import json
import os

@dataclass
class ExperimentConfig:
    tag: str
    model_type: str  # "ae_sde_0", "ae_sde_1", "ae_sde_2", "euclidean"
    manifold_class: str
    dynamics_class: str
    bounds: Tuple[float, float]
    intrinsic_dim: int
    extrinsic_dim: int
    hidden_dims: List[int]
    drift_layers: List[int]
    diff_layers: List[int]
    encoder_activation: str
    decoder_activation: str
    drift_activation: str
    diffusion_activation: str
    n_train: int
    n_test: int
    lr: float
    weight_decay: float
    batch_size: int
    epochs: Dict[str, int]
    loss_weights: Dict[str, float]
    seed: int

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "config.json")
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return ExperimentConfig(**data)
