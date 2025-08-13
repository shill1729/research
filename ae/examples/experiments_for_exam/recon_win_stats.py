"""
This module loops through all of the directories in 'assessment_plots' directory and finds
the lowest error in the interpolation_losses.csv for each type of error desired and then
counts those 'wins' while also computing average and median errors for each column.
"""
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Set the base directory containing the folders (Parabola, BellCurve, etc.)
base_dir = "assessment_plots"

metrics = [
    "Reconstruction",
    "Tangent penalty",
    "Ito penalty",
    "Diffeomorphism Error",
    "Ambient Cov Errors",
    "Ambient Drift Errors"
]

for metric in metrics:

    # Track winners, error sums, and error lists for medians
    winners = []
    error_sums = defaultdict(float)
    error_counts = defaultdict(int)
    error_lists = defaultdict(list)

    # Loop through each folder
    for folder in os.listdir(base_dir):

        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            csv_path = os.path.join(folder_path, "interpolation_losses.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                # Ensure we look only at the metric row
                metric_row = df[df["Error type"] == metric]
                if not metric_row.empty:
                    cols = ["vanilla", "diffeo", "first_order", "second_order"]
                    errors = metric_row[cols].iloc[0]

                    # Record the winner
                    min_col = errors.idxmin()
                    winners.append(min_col)

                    # Accumulate sums and store values for median
                    for col in cols:
                        val = errors[col]
                        error_sums[col] += val
                        error_counts[col] += 1
                        error_lists[col].append(val)
            else:
                raise FileExistsError("interpolation_losses.csv not found")

    # Count wins for each column
    win_counts = Counter(winners)

    # Prepare summary DataFrame
    summary_data = []
    for col in ["vanilla", "diffeo", "first_order", "second_order"]:
        avg_error = error_sums[col] / error_counts[col] if error_counts[col] else float('nan')
        std_error = np.std(error_lists[col], ddof=1) if len(error_lists[col]) > 1 else float('nan')
        median_error = np.median(error_lists[col]) if error_lists[col] else float('nan')
        summary_data.append({
            "Column": col,
            "Wins": win_counts[col],
            "Percent win": win_counts[col]/(error_counts[col]),
            "Average Error": avg_error,
            "Std dev Error": std_error,
            "Median Error": median_error
        })

    summary_df = pd.DataFrame(summary_data)

    print(f"\n=== Summary of Smallest {metric} Error Wins, Average, and Median Errors ===")
    print(summary_df)