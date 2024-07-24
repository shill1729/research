import os
import itertools
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from shillml.diffgeo import RiemannianManifold
from shillml.dynae import *
from shillml.newptcld import PointCloud

from data_processing import process_data
from sde_coefficients import *
from surfaces import *

# Define parameter ranges for grid search
contractive_weights = [0.0001, 0.001, 0.01]
tangent_bundle_weights = [0.1, 1.0, 10.0]
normal_bundle_weights = [0.001, 0.01, 0.1]
learning_rates = [0.0001, 0.001, 0.01]

# Reuse parameters from original experiments
surface = "paraboloid"
num_pts = 30
num_test = 100
seed = 17
bd_epsilon = 0.25
alpha = -1
beta = 1
epochs = 30000
diffusion_epochs = 20000
drift_epochs = 20000
print_freq = 1000
weight_decay = 0.01
extrinsic_dim = 3
intrinsic_dim = 2
h1 = [64, 64]
h2 = [32, 32]
h3 = [32, 32]
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
drift_act = nn.Tanh()
diffusion_act = nn.Tanh()
final_coef_act = None
matrix_norm = "fro"
normalize = False

# Set up the surface and manifold
if surface in surface_bounds:
    chart = globals()[surface]
    bounds = surface_bounds[surface]
    large_bounds = [(b[0] - bd_epsilon, b[1] + bd_epsilon) for b in bounds]
else:
    raise ValueError("Invalid surface")

u, v = sp.symbols("u v", real=True)
local_coordinates = sp.Matrix([u, v])
manifold = RiemannianManifold(local_coordinates, chart)
true_drift = manifold.local_bm_drift() + manifold.metric_tensor().inv() * double_well_potential
true_diffusion = manifold.local_bm_diffusion()

# Generate data
point_cloud = PointCloud(manifold, bounds=bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x, w, mu, cov, _ = point_cloud.generate(n=num_pts, seed=seed)
x, mu, cov, P = process_data(x, mu, cov, d=intrinsic_dim)

# Generate interpolation and extrapolation data
x_interp, _, mu_interp, cov_interp, _ = point_cloud.generate(n=num_test, seed=None)
x_interp, mu_interp, cov_interp, P_interp = process_data(x_interp, mu_interp, cov_interp, d=intrinsic_dim)

point_cloud_extrap = PointCloud(manifold, bounds=large_bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x_extrap, _, mu_extrap, cov_extrap, _ = point_cloud_extrap.generate(n=num_pts, seed=None)
x_extrap, mu_extrap, cov_extrap, P_extrap = process_data(x_extrap, mu_extrap, cov_extrap, d=intrinsic_dim)


def grid_search():
    best_params = None
    best_score = float('inf')
    results = []

    total_iterations = len(contractive_weights) * len(tangent_bundle_weights) * len(normal_bundle_weights) * len(
        learning_rates)
    pbar = tqdm(total=total_iterations, desc="Grid Search Progress")

    for cw, tbw, nbw, lr in itertools.product(contractive_weights, tangent_bundle_weights, normal_bundle_weights,
                                              learning_rates):
        try:
            # Set up CTBAE
            ctbae = ContractiveTangentBundleAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
            ctbae_loss = CTBAELoss(cw, tbw, P)

            # Train CTBAE
            fit_model(ctbae, ctbae_loss, x, x, lr, epochs, print_freq, weight_decay)

            # Set up and train Latent SDE (diffusion)
            latent_sde = LatentNeuralSDE(intrinsic_dim, h2, h3, drift_act, diffusion_act, final_coef_act)
            model_diffusion = AutoEncoderDiffusion(latent_sde, ctbae)
            diffusion_loss = DiffusionLoss(nbw, norm=matrix_norm, normalize=normalize)

            dpi = ctbae.encoder.jacobian_network(x).detach()
            encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
            fit_model(model_diffusion, diffusion_loss, x, (cov, mu, encoded_cov), lr, diffusion_epochs, print_freq,
                      weight_decay)

            # Train Latent SDE (drift)
            toggle_model(model_diffusion.latent_sde.diffusion_net, False)
            model_drift = AutoEncoderDrift(latent_sde, ctbae)
            drift_loss = DriftMSELoss()
            fit_model(model_drift, drift_loss, x, mu, lr, drift_epochs, print_freq, weight_decay)

            # Define helper function for computing losses
            def compute_losses(x, mu, cov, P):
                # Compute in-bound test loss
                mse_loss = nn.MSELoss()
                tbloss = TangentBundleLoss()
                contraction = ContractiveRegularization()
                ctbae_loss_test = CTBAELoss(cw, tbw, P)
                x_test_reconstructed, dpi, P_model = ctbae(x)
                encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT).detach()
                mse = mse_loss(x, x_test_reconstructed).item()
                tb_loss = tbloss.forward(P_model, P).item()
                contraction_value = contraction.forward(dpi).item()
                ctbae_loss_test_value = ctbae_loss_test(ctbae(x), x).item()
                covariance_loss = CovarianceMSELoss(norm=matrix_norm)
                normal_bundle_loss = NormalBundleLoss()
                model_mu_test = model_drift(x)
                model_cov, N, q, bbt = model_diffusion.forward(x)
                tangent_vector = mu - 0.5 * q
                normal_proj_vector = torch.bmm(N, tangent_vector.unsqueeze(2))
                normal_bundle_loss_value = normal_bundle_loss(normal_proj_vector).item()
                cov_mse_loss = covariance_loss.forward(model_cov, cov).item()
                total_diffusion_loss = diffusion_loss.forward(model_diffusion(x), (cov, mu, encoded_cov)).item()
                drift_mse = drift_loss.forward(model_mu_test, mu).item()
                # print("Test Reconstruction Error = " + str(mse))
                # print("Test Tangent Bundle Error = " + str(tb_loss))
                # print("Test Contraction Error = " + str(contractive_weight * contraction_value))
                # print("Test Total CTBAE Error = " + str(ctbae_loss_test_value))
                # print("\nTest Ambient Covariance Fro-sq Error = " + str(cov_mse_loss))
                # print("Test Drift-Curvature Error = " + str(normal_bundle_loss_value))
                # print("Test Total diffusion Error = " + str(total_diffusion_loss))
                # print("Total drift Error = " + str(drift_mse))
                loss_tuple = (mse, tb_loss, cw * contraction_value, ctbae_loss_test_value, cov_mse_loss,
                              normal_bundle_loss_value, total_diffusion_loss, drift_mse)
                return loss_tuple

            # Evaluate
            interp_loss = compute_losses(x_interp, mu_interp, cov_interp, P_interp)
            extrap_loss = compute_losses(x_extrap, mu_extrap, cov_extrap, P_extrap)

            # Calculate overall score (sum of all losses)
            score = sum(interp_loss) + sum(extrap_loss)

            # Store results
            results.append({
                'params': {'cw': cw, 'tbw': tbw, 'nbw': nbw, 'lr': lr},
                'interp_loss': interp_loss,
                'extrap_loss': extrap_loss,
                'score': score
            })

            # Update best parameters if necessary
            if score < best_score:
                best_score = score
                best_params = {'cw': cw, 'tbw': tbw, 'nbw': nbw, 'lr': lr}

        except Exception as e:
            print(f"Error occurred with parameters cw={cw}, tbw={tbw}, nbw={nbw}, lr={lr}: {str(e)}")

        pbar.update(1)

    pbar.close()
    return results, best_params, best_score


def generate_results_table(results, best_params, surface):
    # Sort results by score
    sorted_results = sorted(results, key=lambda x: x['score'])

    # Generate LaTeX table
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n"
    latex_table += "CW & TBW & NBW & LR & Interp Error & Extrap Error & Total Score \\\\ \\hline\n"

    for result in sorted_results[:10]:  # Show top 10 results
        params = result['params']
        interp_error = sum(result['interp_loss'])
        extrap_error = sum(result['extrap_loss'])
        latex_table += f"{params['cw']:.4f} & {params['tbw']:.4f} & {params['nbw']:.4f} & {params['lr']:.4f} & "
        latex_table += f"{interp_error:.6f} & {extrap_error:.6f} & {result['score']:.6f} \\\\ \\hline\n"

    latex_table += "\\end{tabular}\n"

    # Create a detailed caption
    caption = f"Top 10 parameter combinations for CTBAE+SDE Model on {surface} surface. "
    caption += f"CW: Contractive Weight, TBW: Tangent Bundle Weight, NBW: Normal Bundle Weight, LR: Learning Rate. "
    caption += f"Network dimensions: {extrinsic_dim} (extrinsic) to {intrinsic_dim} (intrinsic). "
    caption += f"CTBAE hidden layers: {h1}. Drift SDE hidden layers: {h2}. Diffusion SDE hidden layers: {h3}. "
    caption += f"Best parameters: CW={best_params['cw']:.4f}, TBW={best_params['tbw']:.4f}, "
    caption += f"NBW={best_params['nbw']:.4f}, LR={best_params['lr']:.4f}."

    latex_table += f"\\caption{{{caption}}}\n"
    latex_table += "\\label{tab:grid_search_results}\n\\end{table}"

    # Save LaTeX table to a file
    os.makedirs(f"plots/{surface}/grid_search", exist_ok=True)
    with open(f"plots/{surface}/grid_search/results_table.tex", "w") as f:
        f.write(latex_table)

    print(f"Results table saved to plots/{surface}/grid_search/results_table.tex")


if __name__ == "__main__":
    print("Starting grid search...")
    results, best_params, best_score = grid_search()

    print(f"Grid search completed.")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    generate_results_table(results, best_params, surface)

    # Optionally, you can plot the best model here
    # You'd need to retrain the model with the best parameters
    # and then use the plotting functions from the original experiments