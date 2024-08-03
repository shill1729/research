# Experiment to test where contractive/tangent bundle AE beats vanilla AE
import os

import matplotlib.pyplot as plt
import torch

from shillml.diffgeo import RiemannianManifold
from shillml.autoencoders import *
from shillml.losses import *
from shillml.newptcld import PointCloud

from data_processing import process_data
from sde_coefficients import *
from surfaces import *

surface = "paraboloid"
num_pts = 30
num_test = 100
seed = None
bd_epsilon = 0.5

# Encoder region and quiver length
alpha = -1
beta = 1
# Regularization: contract and tangent
contractive_weight = 0.001
second_order_weight = 0.001
tangent_bundle_weight = 0.01
curvature_weight = 0.01
lr = 0.0001
epochs = 20000
print_freq = 5000
weight_decay = 0.
# Network structure
extrinsic_dim = 3
intrinsic_dim = 2
h1 = [32]
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
true_drift = manifold.local_bm_drift() - manifold.metric_tensor().inv() * double_well_potential
true_diffusion = manifold.local_bm_diffusion()
# true_drift = drift_zero
# true_diffusion = diffusion_identity

# Initialize the point cloud
point_cloud = PointCloud(manifold, bounds=bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x, w, mu, cov, _ = point_cloud.generate(n=num_pts, seed=seed)
x, mu, cov, P = process_data(x, mu, cov, d=intrinsic_dim)
# point_cloud.plot_sample_paths()
# point_cloud.plot_drift_vector_field(x, None, mu, 0.1)

# Interpolation testing point cloud
x_interp, _, mu_interp, cov_interp, _ = point_cloud.generate(n=num_test, seed=None)
x_interp, mu_interp, cov_interp, P_interp = process_data(x_interp, mu_interp, cov_interp, d=intrinsic_dim)

# Extrapolation testing point cloud
point_cloud = PointCloud(manifold, bounds=large_bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x_extrap, _, mu_extrap, cov_extrap, _ = point_cloud.generate(n=num_test, seed=None)
x_extrap, mu_extrap, cov_extrap, P_extrap = process_data(x_extrap, mu_extrap, cov_extrap, d=intrinsic_dim)

# Define models
ae = AutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
cae = ContractiveAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
c2ae = SecondOrderContractiveAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
tbae = TangentBundleAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
ctbae = ContractiveTangentBundleAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
curve_ae = CurvatureCTBAutoEncoder(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
curve2_ae = CC2TBAE(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)


# Define the loss functions for each auto encoder
mse_loss = nn.MSELoss()
cae_loss = CAELoss(contractive_weight)
c2ae_loss = CAEHLoss(contractive_weight, second_order_weight)
tbae_loss = TBAELoss(tangent_bundle_weight, P)
ctbae_loss = CTBAELoss(contractive_weight, tangent_bundle_weight, P)
curve_loss = CurvatureCTBAELoss(contractive_weight, tangent_bundle_weight, curvature_weight, P)
curve2_loss = CC2TBAELoss(contractive_weight, second_order_weight, tangent_bundle_weight, curvature_weight, P)

# Run the program.
if __name__ == "__main__":
    fit_model(ae, mse_loss, x, x, lr, epochs, print_freq, weight_decay)
    fit_model(cae, cae_loss, x, x, lr, epochs, print_freq, weight_decay)
    fit_model(c2ae, c2ae_loss, x, x, lr, epochs, print_freq, weight_decay)
    fit_model(tbae, tbae_loss, x, x, lr, epochs, print_freq, weight_decay)
    fit_model(ctbae, ctbae_loss, x, x, lr, epochs, print_freq, weight_decay)
    fit_model(curve_ae, curve_loss, x, [x, mu, cov], lr, epochs, print_freq, weight_decay)
    fit_model(curve2_ae, curve2_loss, x, (x, mu, cov), lr, epochs, print_freq, weight_decay)

    # Compute interpolation error:
    tbae_loss = TBAELoss(tangent_bundle_weight, P_interp)
    ctbae_loss = CTBAELoss(contractive_weight, tangent_bundle_weight, P_interp)
    curve_loss = CurvatureCTBAELoss(contractive_weight, tangent_bundle_weight, curvature_weight, P_interp)
    curve2_loss = CC2TBAELoss(contractive_weight, second_order_weight, tangent_bundle_weight, curvature_weight,
                              P_interp)
    interpolation_error_ae = mse_loss.forward(ae.forward(x_interp), x_interp).item()
    interpolation_error_cae = cae_loss.forward(cae.forward(x_interp), x_interp)
    interpolation_error_c2ae = c2ae_loss.forward(c2ae.forward(x_interp), x_interp)
    interpolation_error_tbae = tbae_loss.forward(tbae.forward(x_interp), x_interp)
    interpolation_error_ctbae = ctbae_loss.forward(ctbae.forward(x_interp), x_interp)
    interpolation_error_curve_ae = curve_loss.forward(curve_ae.forward(x_interp),
                                                      (x_interp, mu_interp, cov_interp))
    interpolation_error_curve2_ae = curve2_loss.forward(curve2_ae.forward(x_interp),
                                                        (x_interp, mu_interp, cov_interp))

    # Compute extrapolation error:
    tbae_loss = TBAELoss(tangent_bundle_weight, P_extrap)
    ctbae_loss = CTBAELoss(contractive_weight, tangent_bundle_weight, P_extrap)
    curve_loss = CurvatureCTBAELoss(contractive_weight, tangent_bundle_weight, curvature_weight, P_extrap)
    curve2_loss = CC2TBAELoss(contractive_weight, second_order_weight, tangent_bundle_weight, curvature_weight,
                              P_extrap)
    extrapolation_error_ae = mse_loss.forward(ae.forward(x_extrap), x_extrap).item()
    extrapolation_error_cae = cae_loss.forward(cae.forward(x_extrap), x_extrap)
    extrapolation_error_c2ae = c2ae_loss.forward(c2ae.forward(x_extrap), x_extrap)
    extrapolation_error_tbae = tbae_loss.forward(tbae.forward(x_extrap), x_extrap)
    extrapolation_error_ctbae = ctbae_loss.forward(ctbae.forward(x_extrap), x_extrap)
    extrapolation_error_curve_ae = curve_loss.forward(curve_ae.forward(x_extrap),
                                                      (x_extrap, mu_extrap, cov_extrap))
    extrapolation_error_curve2_ae = curve2_loss.forward(curve2_ae.forward(x_extrap),
                                                        (x_extrap, mu_extrap, cov_extrap))

    interp_diffeo_error = [model.compute_diffeo_error(x_interp).detach() for model in
                           [ae, cae, c2ae, tbae, ctbae, curve_ae]]
    extrap_diffeo_error = [model.compute_diffeo_error(x_extrap).detach() for model in
                           [ae, cae, c2ae, tbae, ctbae, curve_ae]]
    print("Interpolation diffeo error")
    print(interp_diffeo_error)
    print("Extrapolation diffeo error")
    print(extrap_diffeo_error)
    # After calculating all errors
    # Create a dictionary to store all errors
    errors = {
        "AE": {"Interpolation": interpolation_error_ae, "Extrapolation": extrapolation_error_ae},
        "CAE": {"Interpolation": interpolation_error_cae, "Extrapolation": extrapolation_error_cae},
        "C2AE": {"Interpolation": interpolation_error_c2ae, "Extrapolation": extrapolation_error_c2ae},
        "TBAE": {"Interpolation": interpolation_error_tbae, "Extrapolation": extrapolation_error_tbae},
        "CTBAE": {"Interpolation": interpolation_error_ctbae, "Extrapolation": extrapolation_error_ctbae},
        "CurveAE": {"Interpolation": interpolation_error_curve_ae, "Extrapolation": extrapolation_error_curve_ae},
        "Curve2AE": {"Interpolation": interpolation_error_curve2_ae, "Extrapolation": extrapolation_error_curve2_ae}
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
    caption += f"2nd-order Contraction weight: {second_order_weight}. "
    caption += f"Tangent bundle weight: {tangent_bundle_weight}. "
    caption += f"Curvature weight: {curvature_weight}. "
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
    torch.save(curve_ae.state_dict(), f"plots/{surface}/autoencoder/curve2_ae.pth")

    # Detach for plots!
    x = x.detach()
    x_extrap = x_extrap.detach()
    # Comparing the four AEs
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    ae.plot_surface(alpha, beta, 30, ax, "AE")

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    cae.plot_surface(alpha, beta, 30, ax, "CAE")

    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    cae.plot_surface(alpha, beta, 30, ax, "C2AE")

    ax = fig.add_subplot(2, 3, 4, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    tbae.plot_surface(alpha, beta, 30, ax, "TBAE")

    ax = fig.add_subplot(2, 3, 5, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    ctbae.plot_surface(alpha, beta, 30, ax, "CTBAE")

    ax = fig.add_subplot(2, 3, 6, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    curve2_ae.plot_surface(alpha, beta, 30, ax, "CC2TBAE")
    plt.show()
