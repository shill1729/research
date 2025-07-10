import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Union, List, Tuple, Optional

from ae.models import AutoEncoderDiffusion, LocalCovarianceLoss, LocalDriftLoss
from ae.models.losses.losses_autoencoder import TotalLoss, LossWeights
from ae.utils import set_grad_tracking


def fit_model(model: nn.Module,
              loss: nn.Module,
              input_data: Tensor,
              targets: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None,
              lr: float = 0.001,
              epochs: int = 1000,
              print_freq: int = 1000,
              weight_decay: float = 0.,
              batch_size: int = None,
              anneal_weights=None,
              early_stopping=True,
              val_split: float = 0.1,
              patience: int = 6000,
              min_delta: float = 0.0) -> None:
    """
    Fit any auto-differentiable model with optional early stopping based on validation loss.

    :param model: A nn.Module with .parameters()
    :param loss: a loss function implemented in torch that takes in (input_data, targets)
    :param input_data: input data assumed to be shape (n, D)
    :param targets: targets assumed to be a tuple containing tensors of shape (n, D) or (n, Dxd) or etc.
    :param lr: learning rate
    :param epochs: number of training epochs
    :param print_freq: print frequency of loss
    :param weight_decay: weight decay
    :param batch_size: batch size for training
    :param anneal_weights:
    :param early_stopping: boolean for early stopping based on validation loss
    :param val_split: fraction of data to reserve for validation (between 0 and 1)
    :param patience: number of epochs with no improvement after which training will be stopped
    :param min_delta: minimum change in validation loss to qualify as an improvement
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # Determine batch size
    if batch_size is None or batch_size > len(input_data):
        batch_size = len(input_data)

    # Build dataset
    if targets is None:
        dataset = TensorDataset(input_data)
    elif isinstance(targets, (list, tuple)):
        dataset = TensorDataset(input_data, *targets)
    else:
        dataset = TensorDataset(input_data, targets)

    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Anneal weights if applicable
        if isinstance(loss, TotalLoss) and anneal_weights is not None:
            for weight_name, schedule in anneal_weights.items():
                if hasattr(loss.weights, weight_name):
                    if callable(schedule):
                        setattr(loss.weights, weight_name, schedule(epoch))
                    elif isinstance(schedule, list) and len(schedule) > epoch:
                        setattr(loss.weights, weight_name, schedule[epoch])

        # Training loop
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch[0]
            extra_targets = batch[1:] if len(batch) > 1 else None
            extra_targets = extra_targets[0] if extra_targets and len(extra_targets) == 1 else extra_targets
            loss_value = loss(model, inputs, extra_targets)
            loss_value.backward()
            optimizer.step()
            epoch_loss += loss_value.item()

        # Compute training RMSE
        train_rmse = (epoch_loss / len(train_loader)) ** 0.5

        # Validation loop (no torch.no_grad() because jacobians wrt input are computed using auto-diff)
        model.eval()
        val_loss_sum = 0.0
        device = input_data.device
        for batch in val_loader:
            inputs = batch[0].to(device)
            extra_targets = batch[1:] if len(batch) > 1 else None
            extra_targets = (extra_targets[0].to(device)
                             if extra_targets and len(extra_targets) == 1
                             else [t.to(device) for t in extra_targets]) \
                if extra_targets else None
            # forward‐and‐loss still creates a computational graph, but we never call backward()
            val_loss = loss(model, inputs, extra_targets)
            val_loss_sum += val_loss.item()

        val_mean_loss = val_loss_sum / len(val_loader)
        val_rmse = val_mean_loss ** 0.5

        # Reporting
        if epoch % print_freq == 0:
            print(f'Epoch {epoch}: Train RMSE={train_rmse:.6f}, Val RMSE={val_rmse:.6f}')

        # Early stopping check
        if early_stopping:
            if val_mean_loss < best_val_loss - min_delta:
                best_val_loss = val_mean_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch} (no improvement in {patience} epochs)')
                    break

    return None


class ThreeStageFit:
    def __init__(self, lr, epochs_ae, epochs_diffusion, epochs_drift, weight_decay, batch_size, print_freq):
        self.lr = lr
        self.epochs_ae = epochs_ae
        self.epochs_diffusion = epochs_diffusion
        self.epochs_drift = epochs_drift
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.print_freq = print_freq

    def three_stage_fit(self, ae_diffusion: AutoEncoderDiffusion, weights: LossWeights, x, mu, cov, p, orthonormal_frame, anneal_weights=None, norm="fro", device="cpu", ambient_cov_mse=False, ambient_drift_mse=False):
        ae_loss = TotalLoss(weights, norm, device)
        ae_diffusion.to(device)
        diffusion_loss = LocalCovarianceLoss(norm, ambient=ambient_cov_mse).to(device)
        drift_loss = LocalDriftLoss(ambient=ambient_drift_mse).to(device)

        # Train the AE.
        set_grad_tracking(ae_diffusion.autoencoder, True)
        set_grad_tracking(ae_diffusion.latent_sde.drift_net, False)
        set_grad_tracking(ae_diffusion.latent_sde.diffusion_net, False)
        print("Training autoencoder...")
        fit_model(model=ae_diffusion.autoencoder,
                  loss=ae_loss,
                  input_data=x,
                  targets=(p, orthonormal_frame, cov, mu),
                  lr=self.lr,
                  epochs=self.epochs_ae,
                  print_freq=self.print_freq,
                  weight_decay=self.weight_decay,
                  batch_size=self.batch_size,
                  anneal_weights=anneal_weights
                  )
        z = ae_diffusion.autoencoder.encoder(x)
        dpi = ae_diffusion.autoencoder.encoder_jacobian(x).detach()
        dphi = ae_diffusion.autoencoder.decoder_jacobian(z).detach()
        set_grad_tracking(ae_diffusion.autoencoder, False)
        set_grad_tracking(ae_diffusion.latent_sde.drift_net, False)
        set_grad_tracking(ae_diffusion.latent_sde.diffusion_net, True)

        # TODO: create option to either use Dpi Sigma Dpi^T or g^{-1} Dphi^T \Sigma (g^{-1} D\phi^T)^T
        # Penroose option
        # g = torch.bmm(dphi.mT, dphi)
        # ginv = torch.linalg.inv(g)
        # dpi_min = torch.bmm(ginv, dphi.mT)
        encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)

        # Encoded covariance using jacobian of encoder
        if ambient_cov_mse:
            loss_space = "ambient"
        else:
            loss_space = "latent"
        print("Training latent infinitesimal covariance via "+loss_space+" MSE...")
        fit_model(model=ae_diffusion,
                  loss=diffusion_loss,
                  input_data=x,
                  targets=(cov, encoded_cov, dphi),
                  lr=self.lr,
                  epochs=self.epochs_diffusion,
                  print_freq=self.print_freq,
                  weight_decay=self.weight_decay,
                  batch_size=self.batch_size
                  )

        # Get pre-trained AE components for drift training
        z = ae_diffusion.autoencoder.encoder(x)
        decoder_jacobian = ae_diffusion.autoencoder.decoder_jacobian(z).detach()
        decoder_hessian = ae_diffusion.autoencoder.decoder_hessian(z).detach()

        # Turn off autoencoder and diffusion.
        set_grad_tracking(ae_diffusion.autoencoder, False)
        set_grad_tracking(ae_diffusion.latent_sde.diffusion_net, False)
        set_grad_tracking(ae_diffusion.latent_sde.drift_net, True)

        if ambient_drift_mse:
            loss_space = "ambient"
        else:
            loss_space = "latent"
        print("Training latent infinitesimal drift via "+loss_space+" MSE...")
        fit_model(model=ae_diffusion,
                  loss=drift_loss,
                  input_data=x,
                  targets=(mu, encoded_cov, decoder_jacobian, decoder_hessian),
                  lr=self.lr,
                  epochs=self.epochs_drift,
                  print_freq=self.print_freq,
                  weight_decay=self.weight_decay,
                  batch_size=self.batch_size
                  )
        return ae_diffusion
