import os
import json
from datetime import datetime
from pathlib import Path
def get_time_horizon_name(tn):
    """
    Determine the time horizon category based on tn value

    Args:
        tn (float): Maximum time horizon

    Returns:
        str: Time horizon category name
    """
    if tn <= 0.01:
        return "/very_short_term/"
    elif 0.01 < tn <= 0.25:
        return "/short_term/"
    elif 0.25 < tn <= 0.75:
        return "/medium_term/"
    elif 0.75 < tn <= 1.:
        return "/long_term/"
    else:
        return "/very_long_term/"


def save_plot(fig, exp_dir, plot_name):
    """Save a matplotlib figure in the experiment directory."""
    plot_path = os.path.join(exp_dir, f"{plot_name}.png")
    fig.savefig(plot_path)
    print(f"Saved plot: {plot_path}")


def save_table(latex_content, exp_dir, table_name):
    """Save LaTeX table in the experiment directory."""
    os.makedirs(exp_dir, exist_ok=True)
    table_path = os.path.join(exp_dir, f"{table_name}.tex")
    with open(table_path, "w") as f:
        f.write(latex_content)
    print(f"Saved LaTeX table: {table_path}")


def print_dict(my_dict: dict):
    for key, values in my_dict.items():
        print(key)
        print(values)
        # print("\n")


def get_experiment_name(params):
    """Generate a unique name for an experiment based on key hyperparameters."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    key_params = [
        f"h{params['hidden_dims']}",
        f"df{params['diffusion_layers']}",
        f"dr{params['drift_layers']}",
        f"lr{params['lr']}",
        f"epochs{params['epochs_ae']}",
    ]
    return f"trained_{timestamp}_" + "_".join(key_params)


def setup_experiment_dir(params, base_dir="trained_models", anneal_tag="not_annealed"):
    """Create a directory for storing experiment data and save the config."""
    exp_name = get_experiment_name(params)
    exp_dir = os.path.join(base_dir, exp_name)+"_"+str(anneal_tag)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=4)
    return exp_dir

# def setup_experiment_dir(params, base_dir=Path("trained_models"), anneal_tag="not_annealed"):
#     """Create a directory for storing experiment data and save the config."""
#     exp_name = get_experiment_name(params)
#     exp_dir = base_dir / f"{exp_name}_{anneal_tag}"
#     exp_dir.mkdir(parents=True, exist_ok=True)
#
#     with open(exp_dir / "config.json", "w") as f:
#         json.dump(params, f, indent=4)
#
#     return str(exp_dir)
