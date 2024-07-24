# Experiment to test where contractive/tangent bundle AE beats vanilla AE
import os

import matplotlib.pyplot as plt

from shillml.diffgeo import RiemannianManifold
from shillml.dynae import *
from shillml.newptcld import PointCloud

from data_processing import process_data
from sde_coefficients import *
from surfaces import *

surface = "paraboloid"
num_pts = 30
num_test = 100
seed = 17
bd_epsilon = 0.5
# Encoder region and quiver length
alpha = -1
beta = 1
# Regularization: contract and tangent
contractive_weight = 0.001
tangent_bundle_weight = 1.
lr = 0.001
epochs = 50000
print_freq = 1000
weight_decay = 0.01
# Network structure
extrinsic_dim = 3
intrinsic_dim = 2
h1 = [64, 64]
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
if seed is not None:
    torch.manual_seed(seed)
u, v = sp.symbols("u v", real=True)
local_coordinates = sp.Matrix([u, v])
# Assume that 'surface' is the user's input
if surface in surface_bounds:
    # Set the chart dynamically
    chart = globals()[surface]
    # Set the bounds from the dictionary
    bounds = surface_bounds[surface]
    # Set large_bounds (if needed)
    large_bounds = [(b[0] - bd_epsilon, b[1] + bd_epsilon) for b in bounds]
else:
    raise ValueError("Invalid surface")

# Initialize the manifold and choose the dynamics
manifold = RiemannianManifold(local_coordinates, chart)
true_drift = manifold.local_bm_drift() + manifold.metric_tensor().inv() * double_well_potential
true_diffusion = manifold.local_bm_diffusion()

# Initialize the point cloud
point_cloud = PointCloud(manifold, bounds=bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x, w, mu, cov, _ = point_cloud.generate(n=num_pts, seed=seed)
x, mu, cov, P = process_data(x, mu, cov, d=intrinsic_dim)

# Interpolation testing point cloud
x_interp, _, mu_interp, cov_interp, _ = point_cloud.generate(n=num_test, seed=None)
x_interp, mu_interp, cov_interp, P_interp = process_data(x_interp, mu_interp, cov_interp, d=intrinsic_dim)

# Extrapolation testing point cloud
point_cloud = PointCloud(manifold, bounds=large_bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x_extrap, _, mu_extrap, cov_extrap, _ = point_cloud.generate(n=num_pts, seed=None)
x_extrap, mu_extrap, cov_extrap, P_extrap = process_data(x_extrap, mu_extrap, cov_extrap, d=intrinsic_dim)

# Define models
ae = AutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
cae = ContractiveAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
tbae = TangentBundleAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
ctbae = ContractiveTangentBundleAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
# Define the loss functions for each auto encoder
mse_loss = nn.MSELoss()
cae_loss = CAELoss(contractive_weight)
tbae_loss = TBAELoss(tangent_bundle_weight, P)
ctbae_loss = CTBAELoss(contractive_weight, tangent_bundle_weight, P)

# Run the program.
if __name__ == "__main__":
    fit_model(ae, mse_loss, x, x, lr, epochs, print_freq, weight_decay)
    fit_model(cae, cae_loss, x, x, lr, epochs, print_freq, weight_decay)
    fit_model(tbae, tbae_loss, x, x, lr, epochs, print_freq, weight_decay)
    fit_model(ctbae, ctbae_loss, x, x, lr, epochs, print_freq, weight_decay)

    # Compute interpolation error:
    tbae_loss = TBAELoss(tangent_bundle_weight, P_interp)
    ctbae_loss = CTBAELoss(contractive_weight, tangent_bundle_weight, P_interp)
    interpolation_error_ae = mse_loss.forward(ae.forward(x_interp), x_interp).item()
    interpolation_error_cae = cae_loss.forward(cae.forward(x_interp), x_interp)
    interpolation_error_tbae = tbae_loss.forward(tbae.forward(x_interp), x_interp)
    interpolation_error_ctbae = ctbae_loss.forward(ctbae.forward(x_interp), x_interp)
    # Compute extrapolation error:
    tbae_loss = TBAELoss(tangent_bundle_weight, P_extrap)
    ctbae_loss = CTBAELoss(contractive_weight, tangent_bundle_weight, P_extrap)
    extrapolation_error_ae = mse_loss.forward(ae.forward(x_extrap), x_extrap).item()
    extrapolation_error_cae = cae_loss.forward(cae.forward(x_extrap), x_extrap)
    extrapolation_error_tbae = tbae_loss.forward(tbae.forward(x_extrap), x_extrap)
    extrapolation_error_ctbae = ctbae_loss.forward(ctbae.forward(x_extrap), x_extrap)

    # After calculating all errors
    # Create a dictionary to store all errors
    errors = {
        "AE": {"Interpolation": interpolation_error_ae, "Extrapolation": extrapolation_error_ae},
        "CAE": {"Interpolation": interpolation_error_cae, "Extrapolation": extrapolation_error_cae},
        "TBAE": {"Interpolation": interpolation_error_tbae, "Extrapolation": extrapolation_error_tbae},
        "CTBAE": {"Interpolation": interpolation_error_ctbae, "Extrapolation": extrapolation_error_ctbae}
    }

    # Print errors to console
    print("\nErrors:")
    for model, error_types in errors.items():
        for error_type, error_value in error_types.items():
            print(f"{model} {error_type} Error: {error_value:.6f}")

    # Generate LaTeX table
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
    latex_table += "Model & Interpolation Error & Extrapolation Error \\\\ \\hline\n"

    for model, error_types in errors.items():
        latex_table += f"{model} & {error_types['Interpolation']:.6f} & {error_types['Extrapolation']:.6f} \\\\ \\hline\n"

    latex_table += "\\end{tabular}\n"

    # Create a detailed caption
    caption = f"Interpolation and Extrapolation Errors for Different Autoencoder Models. "
    caption += f"Surface: {surface}. "
    caption += f"Network dimensions: {extrinsic_dim} (extrinsic) to {intrinsic_dim} (intrinsic). "
    caption += f"Hidden layers: {h1}. "
    caption += f"Encoder activation: {encoder_act.__class__.__name__}. "
    caption += f"Decoder activation: {decoder_act.__class__.__name__}. "
    caption += f"Training epochs: {epochs}. "
    caption += f"Learning rate: {lr}. "
    caption += f"Weight decay: {weight_decay}. "
    caption += f"Contractive weight: {contractive_weight}. "
    caption += f"Tangent bundle weight: {tangent_bundle_weight}. "
    caption += f"Training region bounds: {bounds}. "
    caption += f"Extrapolation region bounds: {large_bounds}. "
    caption += f"Number of training points: {num_pts}. "
    caption += f"Number of test points: {num_test}. "
    caption += f"Random seed: {seed}."

    latex_table += f"\\caption{{{caption}}}\n"
    latex_table += "\\label{tab:ae_errors}\n\\end{table}"

    # Print LaTeX table to console
    print("\nLaTeX Table:")
    print(latex_table)
    # Create the directory structure
    os.makedirs(f"plots/{surface}/autoencoder", exist_ok=True)
    # Optionally, save LaTeX table to a file
    with open(f"plots/{surface}/autoencoder/error_table.tex", "w") as f:
        f.write(latex_table)
    torch.save(ctbae.state_dict(), f"plots/{surface}/autoencoder/ctbae.pth")

    # Detach for plots!
    x = x.detach()
    # Comparing the four AEs
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2])
    ae.plot_surface(alpha, beta, 30, ax, "AE")

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2])
    cae.plot_surface(alpha, beta, 30, ax, "CAE")

    ax = fig.add_subplot(2, 2, 3, projection="3d")
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2])
    tbae.plot_surface(alpha, beta, 30, ax, "TBAE")

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2])
    ctbae.plot_surface(alpha, beta, 30, ax, "CTBAE")
    plt.show()