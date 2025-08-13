import os
import json
import importlib
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

from ae.models import AutoEncoder, LatentNeuralSDE, AutoEncoderDiffusion
from ae.toydata import RiemannianManifold, PointCloud
from ae.utils import process_data
from ae.experiment_classes.errors.ae_errors import compute_all_losses_for_model
from ae.examples.experiments_for_exam.training_script import (
    curve_list, compare_mse,
    encoder_act, decoder_act, final_act,
    drift_act, diffusion_act
)
# TODO additionally save plots and files (only .tex) directly to our path to candidacy exam
TARGET_DIR = "/Users/seanhill/Documents/Typesetting/LaTex Notes/CandidacyExam/plots"

LATEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
    "-": r"\textminus{}"  # only for header/index text, not numeric values
}

def _latex_escape(s: str) -> str:
    """Escape LaTeX-special chars in plain text fields."""
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in str(s))

def save_latex_tabular(curve_name: str, fname: str, df: pd.DataFrame, display_name_map: dict):
    """
    Save DataFrame as a LaTeX tabular with booktabs. Boldface the minimum value in each row.
    Columns are ordered as ['vanilla','diffeo','first_order','second_order'] if present,
    and their headers are mapped through display_name_map.
    """
    curve_dir = os.path.join(PLOTS_BASEDIR, curve_name)
    os.makedirs(curve_dir, exist_ok=True)
    target_dir = os.path.join(TARGET_DIR, curve_name)
    os.makedirs(target_dir, exist_ok=True)

    # Determine column order (only those actually present)
    preferred_order = ["vanilla", "diffeo", "first_order", "second_order"]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df.loc[:, cols]

    # Header display names (escaped)
    header_names = [ _latex_escape(display_name_map.get(c, c)) for c in df.columns ]

    # Compute row-wise minima ignoring NaNs
    # Keep a mask for where the min occurs (ties: bold all minima)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    row_mins = numeric_df.min(axis=1)
    is_min = numeric_df.eq(row_mins, axis=0)

    # Format numbers; keep plain strings for LaTeX insertion
    def fmt(x):
        if pd.isna(x):
            return ""
        # choose a compact, safe format
        return f"{x:.3g}" # 3 sig fig
        # return f"{float(x):.3f}"  # 3 decimals
    # Build LaTeX lines
    lines = []
    lines.append(r"\begin{tabular}{l" + "r"*len(df.columns) + "}")
    lines.append(r"\toprule")
    header = ["Error type"] + header_names
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for idx, row in df.iterrows():
        cells = []
        # Leftmost index label
        cells.append(_latex_escape(idx))
        # Data cells with bold on minima
        for c in df.columns:
            val = numeric_df.at[idx, c]
            s = fmt(val)
            if s == "":
                cells.append(s)
            else:
                if pd.notna(val) and bool(is_min.at[idx, c]):
                    cells.append(r"\textbf{" + s + "}")
                else:
                    cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    # Save the .tex file to our python project
    tex_path = os.path.join(curve_dir, f"{fname}.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote LaTeX table to: {tex_path}")
    # Save the .tex file to our beamer directory
    tex_path2 = os.path.join(target_dir, f"{fname}.tex")
    with open(tex_path2, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote LaTeX table to: {tex_path2}")

# ==== Config ====
BASE_SAVEDIR = "saved_models"
PLOTS_BASEDIR = "assessment_plots"
os.makedirs(PLOTS_BASEDIR, exist_ok=True)

n_test = 20000
num_grid = 100
eps_min = -0.1
eps_max = 0.5
eps_grid_size = 20
test_seed = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if compare_mse:
    display_name_map = {
        "vanilla": "1st-latent",
        "diffeo": "2nd-latent",
        "first_order": "1st-ambient",
        "second_order": "2nd-ambient"
    }
else:
    display_name_map = {
        "vanilla": "vanilla",
        "diffeo": "diffeo",
        "first_order": "first_order",
        "second_order": "second_order"
    }


def savefig(curve_name, plot_name, base_dir=None):
    """Save current matplotlib figure to curve's plot directory."""
    if base_dir is None:
        curve_dir = os.path.join(PLOTS_BASEDIR, curve_name)
        os.makedirs(curve_dir, exist_ok=True)
    else:
        curve_dir = os.path.join(TARGET_DIR, curve_name)
        os.makedirs(curve_dir, exist_ok=True)
    print("Saving to "+curve_dir)
    print(plot_name)
    plt.savefig(os.path.join(curve_dir, f"{plot_name}.png"), dpi=300)


def savecsv(curve_name, fname, df):
    """Save DataFrame to curve's plot directory."""
    curve_dir = os.path.join(PLOTS_BASEDIR, curve_name)
    os.makedirs(curve_dir, exist_ok=True)
    df.to_csv(os.path.join(curve_dir, f"{fname}.csv"), index=True, index_label="Error type")


def load_models_for_curve(save_dir, config):
    """Load all AE-SDE model variants for a given curve."""
    intrinsic_dim = config["intrinsic_dim"]
    extrinsic_dim = config["extrinsic_dim"]
    hidden_dims = config["hidden_dims"]
    drift_layers = config["drift_layers"]
    diff_layers = config["diff_layers"]

    model_names = ["vanilla", "diffeo", "first_order", "second_order"]
    aedf_models = {}
    for name in model_names:
        subdir = os.path.join(save_dir, name)
        ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, final_act=final_act)
        latent_sde = LatentNeuralSDE(intrinsic_dim, drift_layers, diff_layers, drift_act, diffusion_act, encoder_act=None)
        aedf = AutoEncoderDiffusion(latent_sde, ae)
        aedf.load_state_dict(torch.load(os.path.join(subdir, "aedf.pth"), map_location=device))
        aedf.eval()
        aedf_models[name] = aedf
    return aedf_models


def run_assessment_for_curve(curve):
    curve_name = curve.__class__.__name__
    print(f"\n=== Assessing curve: {curve_name} ===")

    save_dir = os.path.join(BASE_SAVEDIR, curve_name)
    with open(os.path.join(save_dir, "config.json"), "r") as f:
        config = json.load(f)

    intrinsic_dim = config["intrinsic_dim"]
    manifold_name = config["manifold"]
    dynamics_name = config["dynamics"]

    # Dynamically load manifold and dynamics
    geom_mod = importlib.import_module("ae.toydata.curves" if intrinsic_dim == 1 else "ae.toydata.surfaces")
    ManifoldClass = getattr(geom_mod, manifold_name)
    curve_instance = ManifoldClass()

    dynamics_mod = importlib.import_module("ae.toydata.local_dynamics")
    DynamicsClass = getattr(dynamics_mod, dynamics_name)
    dynamics = DynamicsClass()

    manifold = RiemannianManifold(curve_instance.local_coords(), curve_instance.equation())
    local_drift = dynamics.drift(manifold)
    local_diffusion = dynamics.diffusion(manifold)

    # Test dataset
    point_cloud = PointCloud(manifold, curve_instance.bounds(), local_drift, local_diffusion, True)
    x, _, mu, cov, local_x = point_cloud.generate(n=n_test, seed=test_seed)
    x, mu, cov, p, n, h = process_data(x, mu, cov, d=intrinsic_dim)

    # Load trained models
    aedf_models = load_models_for_curve(save_dir, config)

    # === Count parameters above a given epsilon threshold for autoencoder only ===
    eps_threshold = 0.01  # adjust this if you want a looser/tighter definition of "zero"

    threshold_counts = {}
    for name, model in aedf_models.items():
        count = 0
        for param in model.autoencoder.parameters():
            count += (param.abs() > eps_threshold).sum().item()
        threshold_counts[name] = {f"Params > {eps_threshold}": count}

    threshold_df = pd.DataFrame.from_dict(threshold_counts, orient="index")
    savecsv(curve_name, "autoencoder_params_above_threshold", threshold_df)

    # === Interpolation Losses ===
    losses = {}
    for name, model in aedf_models.items():
        loss_for_model = compute_all_losses_for_model(model, x, mu, cov, p, intrinsic_dim)
        losses[name] = loss_for_model
    interp_df = pd.DataFrame.from_dict(losses, orient="columns")
    # TODO some how we lost the row names when saving
    # print(interp_df)
    # raise ValueError("stop")
    savecsv(curve_name, "interpolation_losses", interp_df)
    # --- replace your CSV save for interpolation losses with:
    save_latex_tabular(curve_name, "interpolation_losses", interp_df, display_name_map)

    # === Extrapolation Losses ===
    eps_grid = torch.linspace(0.01, eps_max, steps=eps_grid_size)
    bounds = curve_instance.bounds()
    base_lower = torch.tensor([b[0] for b in bounds])
    base_upper = torch.tensor([b[1] for b in bounds])
    shell_results = []

    shell_upper_bounds = [(b[0] - eps_max, b[1] + eps_max) for b in bounds]
    point_cloud_shell = PointCloud(manifold, shell_upper_bounds, local_drift, local_diffusion, True)
    x_all, _, mu_all, cov_all, local_x_all = point_cloud_shell.generate(n=n_test, seed=test_seed)
    local_x_all = torch.tensor(local_x_all, dtype=x.dtype, device=x.device)
    x_all, mu_all, cov_all, p_all, _, _ = process_data(x_all, mu_all, cov_all, d=intrinsic_dim)

    for eps in eps_grid:
        lower = base_lower - eps
        upper = base_upper + eps
        outside_training = torch.any((local_x_all < base_lower - eps_min) | (local_x_all > base_upper + eps_min), dim=1)
        within_eps = torch.all((local_x_all >= lower) & (local_x_all <= upper), dim=1)
        mask = outside_training & within_eps
        if mask.sum() == 0:
            continue

        x_shell, mu_shell, cov_shell, p_shell = x_all[mask], mu_all[mask], cov_all[mask], p_all[mask]
        shell_loss = {"epsilon": eps.item(), "n_points": mask.sum().item()}
        for name, model in aedf_models.items():
            loss_vals = compute_all_losses_for_model(model, x_shell, mu_shell, cov_shell, p_shell, intrinsic_dim)
            for k, v in loss_vals.items():
                shell_loss[f"{name}/{k}"] = v
        shell_results.append(shell_loss)

    shell_df = pd.DataFrame(shell_results)

    # === Grouped extrapolation error plots ===
    groups = {
        "Reconstruction": ["Reconstruction"],
        "Tangent geometry": ["Tangent penalty", "Ito penalty", "Diffeomorphism Error"],
        "Conditioning": [
            "Min Smallest Decoder SV", "Max Largest Decoder SV",
            "Min Smallest Encoder SV", "Max Largest Encoder SV"
        ],
        "Metric Geometry": ["Moore-Penrose error"],
        "Ambient Consistency": ["Ambient Cov Errors", "Ambient Drift Errors"]
    }

    for group_name, keys in groups.items():
        if group_name == "Conditioning":
            fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(len(keys), 1, figsize=(8, 8), sharex=True)
            if len(keys) == 1:
                axes = [axes]

        for ax, key in zip(axes, keys):
            for model in aedf_models:
                col = f"{model}/{key}"
                if col in shell_df.columns:
                    ax.plot(shell_df["epsilon"] + eps_min, shell_df[col],
                            label=display_name_map[model], marker='o')
            ax.axvline(0., color='k', linestyle='--', linewidth=1)
            ax.set_ylabel(key)
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Epsilon")
        fig.suptitle(f"{group_name} vs Epsilon")
        plt.tight_layout()
        savefig(curve_name, f"extrapolation_{group_name.replace(' ', '_').lower()}")
        savefig(curve_name, f"extrapolation_{group_name.replace(' ', '_').lower()}", TARGET_DIR)
        plt.close()

    # === Manifold Reconstruction Plot ===
    if intrinsic_dim == 1:
        phi_u, _ = point_cloud_shell.get_curve(num_grid=num_grid)
        phi_u_tensor = torch.tensor(phi_u, dtype=torch.float32, device=device)
    elif intrinsic_dim == 2:
        phi_u, u_grid = point_cloud_shell.get_surface(num_grid=num_grid)
        phi_u_tensor = torch.tensor(phi_u.reshape(-1, phi_u.shape[-1]), dtype=torch.float32, device=device)
        print("Shape of gt parameterization")
        print(phi_u.shape)
    else:
        raise ValueError("only 1 and 2d manifolds implemented")


    fig, ax = plt.subplots(figsize=(8, 6))
    # TODO: plot surfaces
    if intrinsic_dim == 1:
        ax.plot(phi_u[:, 0], phi_u[:, 1], label="Ground Truth", color="black", linewidth=2)
        for name, model in aedf_models.items():
            with torch.no_grad():
                decoded = model.autoencoder.decoder(model.autoencoder.encoder(phi_u_tensor)).cpu().numpy()
            ax.plot(decoded[:, 0], decoded[:, 1], label=display_name_map[name], linestyle="--")
        ax.set_title(f"Manifold Reconstruction: {curve_name}")
        ax.legend()
        savefig(curve_name, "manifold_reconstruction")
        savefig(curve_name, "manifold_reconstruction", TARGET_DIR)
        plt.close()
    elif intrinsic_dim == 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            phi_u[:, :, 0], phi_u[:, :, 1], phi_u[:, :, 2], color='gray', alpha=0.3, rstride=1, cstride=1, linewidth=0
        )
        for name, model in aedf_models.items():
            with torch.no_grad():
                encoded = model.autoencoder.encoder(phi_u_tensor.view(-1, phi_u_tensor.shape[-1]))
                decoded = model.autoencoder.decoder(encoded).cpu().numpy().reshape(phi_u.shape)
            ax.plot_surface(
                decoded[:, :, 0], decoded[:, :, 1], decoded[:, :, 2],
                label=display_name_map[name], alpha=0.5, rstride=1, cstride=1
            )
        ax.set_title("Surface Reconstruction via AutoEncoder")
        plt.tight_layout()
        ax.legend()
        savefig(curve_name, "manifold_reconstruction")
        savefig(curve_name, "manifold_reconstruction", TARGET_DIR)
        plt.close()


    # === Drift Field Plot (1D only right now) ===
    # TODO: 2d drift plot
    model_names = ["Vanilla", "First Order", "Second Order"]
    model_keys = ["vanilla", "first_order", "second_order"]
    extrinsic_dim = config["extrinsic_dim"]
    if intrinsic_dim == 1:
        mu_gt = np.zeros((num_grid, extrinsic_dim))
        for i, u in enumerate(np.linspace(bounds[0][0], bounds[0][1], num_grid)):
            mu_gt[i] = point_cloud_shell.np_extrinsic_drift(u)[:, 0]
        idx = np.arange(0, num_grid, 3)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, key in zip(axes, ["vanilla", "first_order", "second_order"]):
            ax.plot(phi_u[:, 0], phi_u[:, 1], 'k-', alpha=0.4)
            ax.quiver(phi_u[idx, 0], phi_u[idx, 1], mu_gt[idx, 0], mu_gt[idx, 1],
                      angles='xy', scale_units='xy', width=0.002, color='black', alpha=0.6)
            mu_pred = aedf_models[key].compute_ambient_drift(phi_u_tensor).detach().cpu().numpy()
            ax.quiver(phi_u[idx, 0], phi_u[idx, 1], mu_pred[idx, 0], mu_pred[idx, 1],
                      angles='xy', scale_units='xy', width=0.003, color='red', alpha=0.4)
            ax.set_title(f"{display_name_map[key]} vs GT")
        savefig(curve_name, "drift_field")
        savefig(curve_name, "drift_field", TARGET_DIR)
        plt.close()
    elif intrinsic_dim == 2:
        # Reshape phi_u back to grid for plotting
        U1, U2 = u_grid  # This is assumed to be (U1, U2) from get_surface
        phi_u_grid = phi_u.reshape(num_grid, num_grid, extrinsic_dim)

        # Compute ground truth drift field on the grid
        mu_gt = np.zeros((num_grid, num_grid, extrinsic_dim))
        for i in range(num_grid):
            for j in range(num_grid):
                u = U1[i, j]
                v = U2[i, j]
                mu_gt[i, j, :] = point_cloud_shell.np_extrinsic_drift(u, v)[:, 0]

        # Compute model drifts
        mu_models = {}
        for name in model_keys:
            mu_pred = aedf_models[name].compute_ambient_drift(phi_u_tensor).detach().numpy()
            mu_models[name] = mu_pred.reshape(num_grid, num_grid, extrinsic_dim)

        # Downsample for cleaner vector field plots
        skip = 5  # every 5th grid point
        X = phi_u_grid[::skip, ::skip, 0]
        Y = phi_u_grid[::skip, ::skip, 1]
        Z = phi_u_grid[::skip, ::skip, 2]

        U_gt = mu_gt[::skip, ::skip, 0]
        V_gt = mu_gt[::skip, ::skip, 1]
        W_gt = mu_gt[::skip, ::skip, 2]

        for key, name in zip(model_keys, model_names):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Surface plot
            ax.plot_surface(
                phi_u_grid[:, :, 0], phi_u_grid[:, :, 1], phi_u_grid[:, :, 2],
                color='lightgray', alpha=0.3, rstride=1, cstride=1, linewidth=0
            )

            # Ground truth quiver
            ax.quiver(
                X, Y, Z, U_gt, V_gt, W_gt,
                length=0.1, normalize=True, color='black', alpha=0.6, label='Ground Truth'
            )

            # Model drift quiver
            U_m = mu_models[key][::skip, ::skip, 0]
            V_m = mu_models[key][::skip, ::skip, 1]
            W_m = mu_models[key][::skip, ::skip, 2]
            ax.quiver(
                X, Y, Z, U_m, V_m, W_m,
                length=0.1, normalize=True, color='red', alpha=0.5, label=display_name_map[key]
            )

            ax.set_title(f"{display_name_map[key]} Drift Field vs Ground Truth")
            plt.tight_layout()
            # plt.show()
            savefig(curve_name, "drift_field")
            savefig(curve_name, "drift_field", TARGET_DIR)
            plt.close()




if __name__ == "__main__":
    for curve in curve_list:
        run_assessment_for_curve(curve)