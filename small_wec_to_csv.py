import json
from pathlib import Path
import os
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image

from matplotlib.ticker import MultipleLocator

sns.set_theme()

FIG_DPI = 300  # Set a high DPI for better quality in saved figures
FIG_DPI = 100  # Set a high DPI for better quality in saved figures


def extract_scalar_values(data, parent_key="", metadata_dict=None):
    """Extract scalar values from nested JSON structure."""
    if metadata_dict is None:
        metadata_dict = {}

    result = {}

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}_{key}" if parent_key else key

            if isinstance(value, dict):
                if "value" in value:
                    result[full_key] = value["value"]
                    metadata_dict[full_key] = {
                        "units": value.get("units", []),
                        "info": value.get("info", []),
                    }
                else:
                    nested_result = extract_scalar_values(
                        value, full_key, metadata_dict
                    )
                    result.update(nested_result)
            elif isinstance(value, (int, float, str, bool)):
                result[full_key] = value

    return result


def load_json_smart_encoding(json_file_path):
    """Load JSON with smart encoding detection."""
    try:
        # Try UTF-8 first (most common)
        with open(json_file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except UnicodeDecodeError:
        # Fall back to ISO-8859-1 for files with special characters
        print(
            f"Process {os.getpid()}: UTF-8 failed for {json_file_path}, trying ISO-8859-1"
        )
        with open(json_file_path, "r", encoding="iso-8859-1") as file:
            return json.load(file)


def process_json_to_dataframe(json_file_path):
    """Process JSON file and convert to DataFrame with metadata."""
    data = load_json_smart_encoding(json_file_path)

    print(f"Process {os.getpid()}: Loaded {len(data)} records from {json_file_path}")

    all_records = []
    metadata = {}

    for i, record in enumerate(data):
        if record:
            scalar_values = extract_scalar_values(record, metadata_dict=metadata)
            if scalar_values:
                all_records.append(scalar_values)
            if "wec_type" in record:
                metadata["wec_type"] = record["wec_type"]
            if "wec_scale" in record:
                metadata["wec_scale"] = record["wec_scale"]

        if (i + 1) % 100 == 0:
            print(f"Process {os.getpid()}: Processed {i + 1} records...")

    print(
        f"Process {os.getpid()}: Successfully processed {len(all_records)} non-empty records"
    )

    df = pd.DataFrame(all_records)
    print(f"Process {os.getpid()}: DataFrame created with shape: {df.shape}")

    return df, metadata


def create_2d_binned_matrix(
    df,
    x_col,
    y_col,
    data_col,
    x_start,
    x_end,
    x_bin_size,
    y_start,
    y_end,
    y_bin_size,
    agg_func="mean",
):
    """Create a 2D binned matrix from DataFrame data."""
    x_bins = np.arange(x_start, x_end + x_bin_size, x_bin_size)
    y_bins = np.arange(y_start, y_end + y_bin_size, y_bin_size)

    x_labels = [f"[{x_bins[i]:.0f}-{x_bins[i+1]:.0f})" for i in range(len(x_bins) - 1)]
    y_labels = [f"[{y_bins[i]:.1f}-{y_bins[i+1]:.1f})" for i in range(len(y_bins) - 1)]

    df_copy = df.copy()
    df_copy[x_col] = pd.cut(
        df_copy[x_col], bins=x_bins, labels=x_labels, include_lowest=True
    )
    df_copy[y_col] = pd.cut(
        df_copy[y_col], bins=y_bins, labels=y_labels, include_lowest=True
    )

    if agg_func == "mean":
        binned_data = df_copy.groupby([y_col, x_col], observed=True)[data_col].mean()
    elif agg_func == "sum":
        binned_data = df_copy.groupby([y_col, x_col], observed=True)[data_col].sum()
    elif agg_func == "count":
        binned_data = df_copy.groupby([y_col, x_col], observed=True)[data_col].count()
    elif agg_func == "median":
        binned_data = df_copy.groupby([y_col, x_col], observed=True)[data_col].median()
    else:
        binned_data = df_copy.groupby([y_col, x_col], observed=True)[data_col].agg(
            agg_func
        )

    matrix_df = binned_data.unstack(fill_value=np.nan)
    matrix_df = matrix_df.reindex(index=matrix_df.index[::-1])

    return matrix_df


def plot_binned_matrix_heatmap(
    matrix_df,
    title="Binned Data Matrix",
    x_label="X Variable",
    y_label="Y Variable",
    x_units="",
    y_units="",
    data_units="",
    figsize=(14, 8),
    cmap="viridis",
    annotate=True,
    annotation_format=".2f",
    colorbar_label=None,
    font_size=9,
    title_size=14,
    rotation_x=0,
    rotation_y=0,
    mask_nan=True,
    vmin=None,
    vmax=None,
):
    """Plot a binned matrix as a heatmap with comprehensive customization options."""
    if matrix_df.empty:
        raise ValueError("Input matrix is empty")

    fig, ax = plt.subplots(figsize=figsize)
    plot_data = matrix_df.copy()

    mask = None
    if mask_nan:
        mask = plot_data.isna()

    if colorbar_label is not None and data_units:
        colorbar_label = f"{colorbar_label} [{data_units}]"

    sns.heatmap(
        plot_data,
        annot=annotate,
        fmt=annotation_format,
        cmap=cmap,
        mask=mask,
        cbar_kws={"label": colorbar_label},
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": font_size},
    )

    x_label_with_units = f"{x_label} [{x_units}]" if x_units else x_label
    y_label_with_units = f"{y_label} [{y_units}]" if y_units else y_label

    ax.set_xlabel(x_label_with_units, fontsize=font_size)
    ax.set_ylabel(y_label_with_units, fontsize=font_size)
    ax.set_title(title, fontsize=title_size, pad=20)

    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=rotation_x, ha="center", fontsize=font_size
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), rotation=rotation_y, ha="right", fontsize=font_size
    )

    ax.grid(False)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(which="minor", color="#bbb", linewidth=0.75)

    plt.tight_layout()
    return fig


def process_single_file(args):
    """Process a single JSON file - designed for multiprocessing."""
    input_file, output_data_directory, output_viz_directory = args

    try:
        print(f"Process {os.getpid()}: Starting {input_file.name}")

        # Process the JSON file
        df, metadata = process_json_to_dataframe(input_file)

        # Create run label and directories
        run_label = input_file.stem.replace(".mat", "")
        scale = run_label.split("_")[-1]
        run_label = f"{scale}_scale_{run_label.replace(scale, '')}"
        run_label = run_label.replace("_data_", "")

        by_run_directory = Path(output_viz_directory, "by_run", run_label)
        by_run_directory.mkdir(parents=True, exist_ok=True)
        by_run_energy_period_directory = Path(by_run_directory, "energy_period")
        by_run_peak_period_directory = Path(by_run_directory, "peak_period")
        by_run_energy_period_directory.mkdir(parents=True, exist_ok=True)
        by_run_peak_period_directory.mkdir(parents=True, exist_ok=True)

        # Save data
        this_data_output_path = Path(output_data_directory, run_label)
        this_data_output_path.mkdir(parents=True, exist_ok=True)
        # Create output directories
        by_run_data_peak_period_directory = Path(
            output_data_directory, run_label, "matrices", "peak_period"
        )
        by_run_data_peak_period_directory.mkdir(parents=True, exist_ok=True)
        by_run_data_energy_period_directory = Path(
            output_data_directory, run_label, "matrices", "energy_period"
        )
        by_run_data_energy_period_directory.mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(this_data_output_path, f"{run_label}_columnar.csv"), index=False)

        # Process each column
        processed_columns = 0
        for col in list(df.columns):
            by_col_directory = Path(output_viz_directory, "by_col", col)
            by_col_directory.mkdir(parents=True, exist_ok=True)
            by_col_energy_period_directory = Path(by_col_directory, "energy_period")
            by_col_peak_period_directory = Path(by_col_directory, "peak_period")
            by_col_energy_period_directory.mkdir(parents=True, exist_ok=True)
            by_col_peak_period_directory.mkdir(parents=True, exist_ok=True)

            this_units = ""
            if (
                col in metadata
                and "units" in metadata[col]
                and len(metadata[col]["units"]) > 0
            ):
                this_units = metadata[col]["units"][0]

            format_string = ".2f"
            if pd.api.types.is_numeric_dtype(df[col]):
                col_max = df[col].max()
                col_min = df[col].min()
                if col_max > 100 or col_min < -100:
                    format_string = ".1f"
                if col_max > 1000 or col_min < -1000:
                    format_string = ".0f"

            # Process peak period matrix
            try:
                this_matrix = create_2d_binned_matrix(
                    df,
                    x_col="peak_period",
                    x_start=0,
                    x_end=23,
                    x_bin_size=1,
                    y_col="wave_height",
                    y_start=0,
                    y_end=9,
                    y_bin_size=0.5,
                    data_col=col,
                    agg_func="mean",
                )

                unique_result = count_numeric_df_uniqueness(this_matrix)
                if unique_result["count"] == 1:
                    metadata[col] = unique_result["values"][0]
                    continue

                save_matrix = this_matrix.copy()
                save_matrix = save_matrix.sort_index(ascending=False)
                save_matrix.index.name = f"X:Peak Period Bins [s], Y:Wave Height Bins [m], Values: {col.replace('_', ' ').title()} [{this_units}]"

                save_matrix.to_csv(
                    Path(
                        by_run_data_peak_period_directory,
                        f"{run_label}_{col}_peak_period_matrix.csv",
                    )
                )
                plot_binned_matrix_heatmap(
                    this_matrix,
                    title=f"{metadata['wec_type']} - {metadata['wec_scale']}m Scale\n{col.replace('_', ' ').title()}",
                    x_label="Peak Period",
                    y_label="Wave Height",
                    colorbar_label=col.replace("_", " ").title(),
                    x_units="s",
                    y_units="m",
                    data_units=this_units,
                    cmap="viridis",
                    annotation_format=format_string,
                )

                plt.savefig(
                    Path(
                        by_run_peak_period_directory,
                        f"{run_label}_{col}_peak_period_matrix.png",
                    ),
                    dpi=FIG_DPI,
                )
                plt.savefig(
                    Path(
                        by_col_peak_period_directory,
                        f"{run_label}_{col}_peak_period_matrix.png",
                    ),
                    dpi=FIG_DPI,
                )
                plt.close()
                this_matrix = create_2d_binned_matrix(
                    df,
                    x_col="energy_period",
                    x_start=0,
                    x_end=23,
                    x_bin_size=1,
                    y_col="wave_height",
                    y_start=0,
                    y_end=9,
                    y_bin_size=0.5,
                    data_col=col,
                    agg_func="mean",
                )

                # Save the matrix to CSV
                save_matrix = this_matrix.copy()
                save_matrix = save_matrix.sort_index(ascending=False)
                save_matrix.index.name = f"X:Energy Period Bins [s], Y:Wave Height Bins [m], Values: {col.replace('_', ' ').title()} [{this_units}]"
                save_matrix.to_csv(
                    Path(
                        by_run_data_energy_period_directory,
                        f"{run_label}_{col}_energy_period_matrix.csv",
                    )
                )

                plot_binned_matrix_heatmap(
                    this_matrix,
                    title=f"Small Scale WEC Performance Modeling\n{metadata['wec_type']} - {metadata['wec_scale']}m Scale\n{col.replace('_', ' ').title()}",
                    x_label="Energy Period",
                    y_label="Wave Height",
                    colorbar_label=col.replace("_", " ").title(),
                    x_units="s",
                    y_units="m",
                    data_units=this_units,
                    cmap="viridis",
                    annotation_format=format_string,
                )

                plt.savefig(
                    Path(
                        by_run_energy_period_directory,
                        f"{run_label}_{col}_energy_period_matrix.png",
                    ),
                    dpi=FIG_DPI,
                )
                plt.savefig(
                    Path(
                        by_col_energy_period_directory,
                        f"{run_label}_{col}_energy_period_matrix.png",
                    ),
                    dpi=FIG_DPI,
                )
                plt.close()

            except Exception:
                continue

            processed_columns += 1

        # Save metadata
        metadata_output_path = Path(this_data_output_path, f"{run_label}_metadata.json")
        with open(metadata_output_path, "w") as meta_file:
            json.dump(metadata, meta_file, indent=2)

        print(
            f"Process {os.getpid()}: Completed {input_file.name} - processed {processed_columns} columns"
        )
        return f"Success: {input_file.name}"

    except Exception as e:
        error_msg = f"Error processing {input_file.name}: {str(e)}"
        print(f"Process {os.getpid()}: {error_msg}")
        return error_msg


def optimize_png_output(directory_with_png_files):
    """Optimize PNG files in a directory using pngquant."""
    png_files = sorted(list(directory_with_png_files.rglob("*.png")))
    print(f"Optimizing {len(png_files)} PNG files in {directory_with_png_files}")

    n_files = len(png_files)
    for i, png_file in enumerate(png_files):
        if i % 10 == 0:
            print(f"Optimizing file {i + 1}/{n_files}: {png_file.name}")
        with Image.open(png_file) as img:
            img.save(png_file, "PNG", optimize=True)


def count_numeric_df_uniqueness(df):
    # Validate all values are scalar numeric or NaN
    for col in df.columns:
        for idx, value in df[col].items():
            if pd.isna(value):
                continue

            # Check if value is scalar
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                raise ValueError(f"Non-scalar value found at [{idx}, '{col}']: {value}")

            # Check if value is numeric
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(
                    f"Non-numeric value found at [{idx}, '{col}']: {value} (type: {type(value)})"
                )

    # Get all numeric values (excluding NaN)
    all_values = []
    for col in df.columns:
        numeric_values = df[col].dropna().tolist()
        all_values.extend(numeric_values)

    # Get unique values and count
    unique_values = list(set(all_values))
    unique_values.sort()  # Sort for consistent output

    return {"count": len(unique_values), "values": unique_values}


def main():
    """Main function with multiprocessing."""
    # Setup directories
    input_directory = Path("./data/00_raw/")
    output_data_directory = Path("./data/b1_vap/")
    output_data_directory.mkdir(parents=True, exist_ok=True)
    output_viz_directory = Path("./viz/")
    output_viz_directory.mkdir(parents=True, exist_ok=True)

    # Get input files
    input_paths = sorted(list(input_directory.glob("*.json")))

    if not input_paths:
        print("No JSON files found in input directory")
        return

    print(f"Found {len(input_paths)} files to process")

    # Determine number of processes (use 75% of available cores)
    num_processes = max(1, int(mp.cpu_count() * 0.75))
    print(f"Using {num_processes} processes")

    # Prepare arguments for each file
    process_args = [
        (path, output_data_directory, output_viz_directory) for path in input_paths
    ]

    # Process files in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, process_args)

    # Optimize PNG files in the output directory
    optimize_png_output(output_viz_directory)

    # Print results summary
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)

    successful = [r for r in results if r.startswith("Success")]
    failed = [r for r in results if not r.startswith("Success")]

    print(f"Successfully processed: {len(successful)} files")
    if failed:
        print(f"Failed to process: {len(failed)} files")
        for failure in failed:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
