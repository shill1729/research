# scripts/run_experiment.py
import argparse
import os
import datetime
from experiments.scripts.experiment_config import ExperimentConfig
from experiments.scripts.experiment_runner import run_full_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{timestamp}_{config.tag}"
    experiment_dir = os.path.join("experiments", experiment_name)

    run_full_experiment(config, experiment_dir)

if __name__ == '__main__':
    main()
