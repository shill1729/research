import os

import torch

from ae.toydata import ToyData
from ae.experiments import Trainer
from ae.toydata.local_dynamics import *
from ae.toydata.surfaces import *

# TODO: I recently switched back to ambient MSE's for the AE-SDEs. Make sure you are aware of this.
#  TODO: try more neurons, training more
device = torch.device("cpu")
train_seed = None
test_seed = None
embedding_seed = 17
norm = "fro"
eps_max = 0.5
# Set large dim = 5,10, 100 for embedding into higher dimension. Note this is imported into 'inference.py'! So make
# sure it lines up.
# TODO: currently does not work for the boundary initial point of the sample paths. So it is broken for now.
large_dim = None
embed = False # Bool for embedding or not
print("Current working directory of train.py")
print(os.getcwd())

if __name__ == "__main__":

    # torch.manual_seed(train_seed)
    # Point cloud parameters
    num_points = 30
    num_test = 20000
    batch_size = int(num_points/2)
    eps_grid_size = 10
    # The intrinsic and extrinsic dimensions.
    extrinsic_dim, intrinsic_dim = 3, 2
    hidden_dims = [32]
    diffusion_layers = [32]
    drift_layers = [32]
    lr = 0.001
    weight_decay = 0.
    epochs_ae = 9000
    epochs_diffusion = 9000
    epochs_drift = 9000
    print_freq = 1000
    # Diffeo weight for accumulative orders
    diffeo_weight_12 = 0.2 # this is the separate diffeo_weight for just the First order and second order
    # First order weight: 0.08 was good
    tangent_angle_weight = 0.01
    # Second order weights accumulative
    tangent_angle_weight2 = 0.02 # the first order weight for the second order model, if accumulating penalties
    tangent_drift_weight = 0.02
    surface = RationalSurface()
    dynamics = LangevinHarmonicOscillator()
    if embed:
        extrinsic_dim = large_dim

    # Main below
    params = {
        "num_points": num_points,
        "num_test": num_test,
        "batch_size": batch_size,
        "eps_max": eps_max,
        "eps_grid_size": eps_grid_size,
        "extrinsic_dim": extrinsic_dim,
        "intrinsic_dim": intrinsic_dim,
        "hidden_dims": hidden_dims,
        "diffusion_layers": diffusion_layers,
        "drift_layers": drift_layers,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_ae": epochs_ae,
        "epochs_diffusion": epochs_diffusion,
        "epochs_drift": epochs_drift,
        "print_freq": print_freq,
        "tangent_angle_weight": tangent_angle_weight,
        "tangent_angle_weight2": tangent_angle_weight2,
        "tangent_drift_weight": tangent_drift_weight,
        "diffeo_weight": diffeo_weight_12,
        "norm": norm
    }

    # TODO refactor this so we're not just commenting out things
    # No annealing:
    anneal_weights = None

    # Linearly increasing tangent drift weight
    # anneal_weights = {"tangent_drift_weight": lambda epoch: tangent_drift_weight * (epoch / epochs_ae) if epoch >
    # np.round(epochs_ae / 5) else 0.}

    # Constant tangent drift weight after warm start
    # anneal_weights = {"tangent_drift_weight": lambda epoch: tangent_drift_weight if epoch >
    #                                                                                 np.round(epochs_ae / 4) else 0.}

    anneal_tag = "annealed_2nd" if anneal_weights is not None else "not_annealed"

    toydata = ToyData(surface, dynamics, large_dim, embedding_seed)
    trainer = Trainer(toydata, params, device, anneal_tag, embed=embed)
    trainer.train(anneal_weights)
    print(trainer.exp_dir)
    trainer.save_models()
