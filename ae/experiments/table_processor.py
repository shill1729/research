import pandas as pd
import os


def format_row_name(row_name):
    # Format row names to match the desired LaTeX table style
    formatted_name = row_name.lower().replace(" loss", "").replace("encoder ", "").replace("tangent ", "tangent ")
    formatted_name = formatted_name.replace("alignment", "").replace("diffeomorphism1", "diffeo")
    return formatted_name.strip()


def format_and_convert_csv_to_latex(csv_file_path, output_latex_file):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file_path, index_col=0)

    # Format row names
    df.index = df.index.map(format_row_name)

    # Identify the minimum in each row and apply bold formatting
    def highlight_min(row):
        # Convert row to numeric for comparison, ignoring errors
        numeric_row = pd.to_numeric(row, errors='coerce')
        # Find the index of the minimum value
        min_index = numeric_row.idxmin()
        # Apply bold formatting to the minimum value
        return [
            f"\\textbf{{{value:.5e}}}" if idx == min_index else f"{value:.5e}"
            for idx, value in zip(row.index, numeric_row)
        ]

    # Apply formatting to each row
    formatted_data = df.apply(highlight_min, axis=1)

    # Re-create the DataFrame with formatted data
    formatted_df = pd.DataFrame(formatted_data.tolist(), columns=df.columns, index=df.index)

    # Convert the formatted DataFrame to LaTeX table
    latex_table = formatted_df.to_latex(
        index=True,  # Keep the index as the first column
        escape=False,  # Ensure LaTeX commands like \textbf{} are not escaped
        column_format="l" + "c" * len(df.columns)  # Left align index, center align columns
    )

    # Save the LaTeX table to a file
    with open(output_latex_file, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_latex_file}")


def process_csv_folder(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all CSV files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_name = os.path.splitext(file_name)[0] + ".tex"
            output_file_path = os.path.join(output_folder, output_file_name)

            # Process each CSV file
            format_and_convert_csv_to_latex(input_file_path, output_file_path)

    print(f"Processed all CSV files in {input_folder} and saved LaTeX tables to {output_folder}")


# Example usage
process_csv_folder("experiment_results/ProductSurface/extrap", "output_latex_folder")
