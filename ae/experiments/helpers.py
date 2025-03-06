import os
import json
from datetime import datetime


def save_plot(fig, exp_dir, plot_name):
    """Save a matplotlib figure in the experiment directory."""
    plot_path = os.path.join(exp_dir, f"{plot_name}.png")
    fig.savefig(plot_path)
    print(f"Saved plot: {plot_path}")


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
