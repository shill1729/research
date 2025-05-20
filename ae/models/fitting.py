from typing import Union, List, Tuple, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from ae.models import AutoEncoderDiffusion
from ae.models.losses import TotalLoss, LocalCovarianceLoss, LocalDriftLoss, LossWeights
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
              anneal_weights=None) -> None:
    """
    Fit any auto-differentiable model.

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
    :return:
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # If batch_size is None or larger than dataset, use the full dataset as one batch
    if batch_size is None or batch_size > len(input_data):
        batch_size = len(input_data)

    # Create TensorDataset and DataLoader
    if targets is None:
        dataset = TensorDataset(input_data)
    elif isinstance(targets, (list, tuple)):
        dataset = TensorDataset(input_data, *targets)
    else:
        dataset = TensorDataset(input_data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs + 1):
        model.train()
        epoch_loss = 0.0  # Reset the epoch loss at the start of each epoch

        # Anneal weights if applicable
        if isinstance(loss, TotalLoss) and anneal_weights is not None:
            for weight_name, schedule in anneal_weights.items():
                if hasattr(loss.weights, weight_name):
                    if callable(schedule):
                        setattr(loss.weights, weight_name, schedule(epoch))
                    elif isinstance(schedule, list) and len(schedule) > epoch:
                        setattr(loss.weights, weight_name, schedule[epoch])

        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch[0]
            # Extract remaining tensors in the batch as extra targets
            extra_targets = batch[1:] if len(batch) > 1 else None
            extra_targets = extra_targets[0] if extra_targets and len(extra_targets) == 1 else extra_targets
            loss_value = loss(model, inputs, extra_targets)
            loss_value.backward()
            optimizer.step()
            epoch_loss += loss_value.item()  # Accumulate batch loss into epoch loss


        # Print average loss for the epoch if print_freq is met
        if epoch % print_freq == 0:
            # print(f'Epoch: {epoch}: Train-Loss: {epoch_loss / len(dataloader):.6f}')
            # print(f'Epoch: {epoch}: Train-Loss: {epoch_loss:.6f}')
            total_batches = len(dataloader)
            mean_mse = epoch_loss / total_batches
            rmse = mean_mse ** 0.5
            print(f'Epoch {epoch}: RMSE: {rmse:.6f}')
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

    def three_stage_fit(self, ae_diffusion: AutoEncoderDiffusion, weights: LossWeights, x, mu, cov, p, orthonormal_frame, anneal_weights=None, norm="fro", device="cpu"):
        ae_loss = TotalLoss(weights, norm, device)
        ae_diffusion.to(device)
        diffusion_loss = LocalCovarianceLoss(norm).to(device)
        drift_loss = LocalDriftLoss().to(device)

        # Train the AE.
        set_grad_tracking(ae_diffusion.autoencoder, True)
        set_grad_tracking(ae_diffusion.latent_sde.drift_net, False)
        set_grad_tracking(ae_diffusion.latent_sde.diffusion_net, False)
        print("Training autoencoder")
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
        dpi = ae_diffusion.autoencoder.encoder_jacobian(x).detach()
        set_grad_tracking(ae_diffusion.autoencoder, False)
        set_grad_tracking(ae_diffusion.latent_sde.diffusion_net, True)
        encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
        print("Training diffusion")
        fit_model(model=ae_diffusion,
                  loss=diffusion_loss,
                  input_data=x,
                  targets=encoded_cov, # TODO: we need to decide on using latent drift or ambient drift for the AE-SDE
                  lr=self.lr,
                  epochs=self.epochs_diffusion,
                  print_freq=self.print_freq,
                  weight_decay=self.weight_decay,
                  batch_size=self.batch_size
                  )
        set_grad_tracking(ae_diffusion.latent_sde.diffusion_net, False)
        set_grad_tracking(ae_diffusion.latent_sde.drift_net, True)
        print("Training drift")
        fit_model(model=ae_diffusion,
                  loss=drift_loss,
                  input_data=x,
                  targets=(mu, cov),
                  lr=self.lr,
                  epochs=self.epochs_drift,
                  print_freq=self.print_freq,
                  weight_decay=self.weight_decay,
                  batch_size=self.batch_size
                  )
        return ae_diffusion
