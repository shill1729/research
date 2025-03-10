# TODO: does this work?
import torch
import numpy as np
import sys
from ae.experiments.datagen import ToyData
from ae.experiments.training import Trainer
from ae.toydata.local_dynamics import *
from ae.toydata.surfaces import *
from ae.experiments.manifold_errors import GeometryError
from ae.experiments.sde_errors import DynamicsError
from ae.experiments.path_computations import compute_increments
from ae.experiments.helpers import get_time_horizon_name, print_dict


def interactive_menu():
    print("\n=== Interactive Autoencoder Training & Analysis ===")
    print("1. Configure & Train Model")
    print("2. Load & Analyze Pre-trained Model")
    print("3. Exit")

    choice = input("Select an option: ")
    return choice


def configure_training():
    print("\n=== Model Training Configuration ===")
    num_points = int(input("Enter number of points (default: 30): ") or 30)
    num_test = int(input("Enter number of test samples (default: 20000): ") or 20000)
    batch_size = int(input("Enter batch size (default: 20): ") or 20)
    lr = float(input("Enter learning rate (default: 0.001): ") or 0.001)
    epochs = int(input("Enter number of epochs (default: 9000): ") or 9000)
    surface = Paraboloid()
    dynamics = LangevinHarmonicOscillator()
    device = torch.device("cpu")

    params = {
        "num_points": num_points,
        "num_test": num_test,
        "batch_size": batch_size,
        "lr": lr,
        "epochs_ae": epochs,
        "epochs_diffusion": epochs,
        "epochs_drift": epochs,
    }

    toydata = ToyData(surface, dynamics)
    trainer = Trainer(toydata, params, device, "not_annealed")
    print("Starting training...")
    trainer.train(None)
    trainer.save_models()
    print("Training complete! Model saved at:", trainer.exp_dir)


def analyze_pretrained_model():
    model_dir = input("Enter the model directory: ")
    trainer = Trainer.load_from_pretrained(model_dir)
    device = "cpu"
    show = False
    eps_grid_size = 10
    num_test = 20000
    h = 0.001
    n_paths = 900
    time_horizons = [0.01, 0.1, 0.5, 1.]

    print("Running Geometry Error Analysis...")
    geometry = GeometryError(trainer.toy_data, trainer, 1., device, show=show)
    geometry.compute_and_plot_errors(eps_grid_size, num_test, None, device)

    for tn in time_horizons:
        print("\nAnalyzing Dynamics Error for time horizon:", tn)
        n_time = int(np.ceil(tn / h))
        dynamics_error = DynamicsError(trainer.toy_data, trainer, tn, show=show)
        gt, at, aes, gt_local, aes_local = dynamics_error.sample_path_generator.generate_paths(tn, n_time, n_paths,
                                                                                               None)

        if n_paths < 1000:
            dynamics_error.sample_path_plotter.plot_sample_paths(gt, aes, at, True, "ambient")
        dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, False)
        dynamics_error.sample_path_plotter.plot_kernel_density(gt, aes, at, True)

        results = dynamics_error.analyze_statistical_properties(gt, at, aes, tn)
        dynamics_error.sample_path_plotter.run_all_analyses(results)
    print("Analysis complete.")


def main():
    while True:
        choice = interactive_menu()
        if choice == "1":
            configure_training()
        elif choice == "2":
            analyze_pretrained_model()
        elif choice == "3":
            print("Exiting application.")
            sys.exit()
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
