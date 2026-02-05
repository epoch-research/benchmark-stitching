#!/usr/bin/env python3
"""
Regularization stability analysis for benchmark stitching.

This script analyzes how the statistical model's parameter estimates change
with different regularization strengths. It sweeps through regularization
values and tracks the average magnitudes of capabilities (C), difficulties (D),
and slopes (α) to understand parameter stability.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import scores_df
from fit import fit_statistical_model


# ============================================================================
# STYLING SETUP
# ============================================================================


def setup_custom_style():
    """Set up custom graph styling for all plots."""
    custom_colors = [
        "#00A5A6",  # teal
        "#E03D90",  # pink
        "#FC6538",  # orange
        "#6A3ECB",  # purple
        "#0058DC",  # blue
        "#EA8D00",  # yellow
        "#B087F4",  # lightPurple
        "#279E27",  # green
        "#009AF1",  # lightBlue
        "#015D90",  # darkBlue
        "#EA4831",  # red
        "#E1C700",  # yellow2
        "#46FFFF",  # turquoise
        "#63F039",  # lightGreen
    ]

    sns.set_palette(custom_colors)

    sns.set_theme(
        style="whitegrid",
        palette=custom_colors,
        context="notebook",
    )

    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 120,
            "axes.titley": 1.02,
            "axes.titlesize": 14,
            "axes.titlelocation": "center",
            "axes.titlepad": 0,
            "axes.labelsize": 12,
            "axes.labelpad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.top": False,
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.right": False,
            "legend.fontsize": 10,
            "legend.loc": "upper left",
            "legend.frameon": True,
            "legend.borderaxespad": 0,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "lines.markeredgecolor": "auto",
            "lines.markeredgewidth": 0.5,
            "errorbar.capsize": 3,
            "font.family": "Arial",
            "font.sans-serif": ["DejaVu Sans"],
            "grid.alpha": 0.3,
            "grid.linestyle": "-",
            "grid.color": "lightgray",
        }
    )

    return custom_colors


def save_plot(output_path: Path, dpi: int = 300, bbox_inches: str = "tight"):
    """Save the current plot as both PNG and PDF."""
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)

    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches=bbox_inches)

    return png_path, pdf_path


# ============================================================================
# REGULARIZATION SWEEP ANALYSIS
# ============================================================================


def run_regularization_sweep(
    lambda_min: float = -5,
    lambda_max: float = 0,
    n_points: int = 51,
    anchor_benchmark: str = "Winogrande",
    anchor_difficulty: float = 0,
    anchor_slope: float = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep through regularization strengths and record parameter magnitudes.

    Args:
        lambda_min: Minimum log10 regularization strength (default: -5)
        lambda_max: Maximum log10 regularization strength (default: 0)
        n_points: Number of points in the sweep (default: 51)
        anchor_benchmark: Benchmark to use as anchor (default: "Winogrande")
        anchor_difficulty: Anchor difficulty value (default: 0)
        anchor_slope: Anchor slope value (default: 1)
        verbose: Whether to print progress (default: True)

    Returns:
        DataFrame with columns: regularizer_exp, avg_C_magnitude, avg_D_magnitude, avg_alpha_magnitude
    """
    lambda_exps = np.linspace(lambda_min, lambda_max, n_points)
    results = []

    if verbose:
        print(f"Running regularization sweep from λ=10^{lambda_min} to λ=10^{lambda_max}")
        print(f"Number of points: {n_points}")
        print(f"Anchor benchmark: {anchor_benchmark}")
        print()

    for i, lambda_exp in enumerate(lambda_exps):
        if verbose and (i % 10 == 0 or i == len(lambda_exps) - 1):
            print(f"  Progress: {i + 1}/{n_points} (λ=10^{lambda_exp:.2f})")

        df, df_model, df_bench = fit_statistical_model(
            scores_df,
            anchor_mode="benchmark",
            anchor_benchmark=anchor_benchmark,
            anchor_difficulty=anchor_difficulty,
            anchor_slope=anchor_slope,
            regularization_strength=10**lambda_exp,
        )

        results.append(
            {
                "regularizer_exp": lambda_exp,
                "avg_C_magnitude": df_model["estimated_capability"].abs().mean(),
                "avg_D_magnitude": df_bench["estimated_difficulty"].abs().mean(),
                "avg_alpha_magnitude": df_bench["estimated_slope"].abs().mean(),
            }
        )

    return pd.DataFrame(results)


def create_stability_plots(
    data: pd.DataFrame,
    output_dir: Path,
    colors: list = None,
) -> dict:
    """
    Create visualization plots for regularization stability analysis.

    Args:
        data: DataFrame from run_regularization_sweep
        output_dir: Directory to save plots
        colors: Custom color palette (optional)

    Returns:
        Dictionary with paths to saved plots
    """
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = {}

    # Plot 1: Parameter magnitude tradeoffs (2x2 grid like original notebook)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)

    # C vs D magnitude
    ax = sns.lineplot(
        data=data,
        x="avg_C_magnitude",
        y="avg_D_magnitude",
        hue="regularizer_exp",
        palette="rocket",
        marker="o",
        ax=axes[0, 0],
    )
    sns.move_legend(ax, loc="best", title="λ=10^x")
    axes[0, 0].set_xlabel("Avg |C| (Capability Magnitude)")
    axes[0, 0].set_ylabel("Avg |D| (Difficulty Magnitude)")
    axes[0, 0].set_title("Capability vs Difficulty Magnitude")

    # C vs alpha magnitude
    ax = sns.lineplot(
        data=data,
        x="avg_C_magnitude",
        y="avg_alpha_magnitude",
        hue="regularizer_exp",
        palette="rocket",
        marker="o",
        ax=axes[1, 0],
    )
    sns.move_legend(ax, loc="best", title="λ=10^x")
    axes[1, 0].set_xlabel("Avg |C| (Capability Magnitude)")
    axes[1, 0].set_ylabel("Avg |α| (Slope Magnitude)")
    axes[1, 0].set_title("Capability vs Slope Magnitude")

    # alpha vs D magnitude
    ax = sns.lineplot(
        data=data,
        x="avg_alpha_magnitude",
        y="avg_D_magnitude",
        hue="regularizer_exp",
        palette="rocket",
        marker="o",
        ax=axes[0, 1],
    )
    sns.move_legend(ax, loc="best", title="λ=10^x")
    axes[0, 1].set_xlabel("Avg |α| (Slope Magnitude)")
    axes[0, 1].set_ylabel("Avg |D| (Difficulty Magnitude)")
    axes[0, 1].set_title("Slope vs Difficulty Magnitude")

    # Hide the fourth subplot
    axes[1, 1].set_visible(False)

    plt.tight_layout()
    png_path, pdf_path = save_plot(output_dir / "parameter_magnitude_tradeoffs")
    saved_plots["tradeoffs"] = (png_path, pdf_path)
    plt.close()

    # Plot 2: Parameter magnitudes vs regularization strength
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Capability magnitude vs lambda
    axes[0].plot(
        data["regularizer_exp"],
        data["avg_C_magnitude"],
        marker="o",
        color=colors[0],
        linewidth=2,
        markersize=4,
    )
    axes[0].set_xlabel("log₁₀(λ)")
    axes[0].set_ylabel("Avg |C| (Capability Magnitude)")
    axes[0].set_title("Capability Magnitude vs Regularization")
    axes[0].grid(True, alpha=0.3)

    # Difficulty magnitude vs lambda
    axes[1].plot(
        data["regularizer_exp"],
        data["avg_D_magnitude"],
        marker="o",
        color=colors[1],
        linewidth=2,
        markersize=4,
    )
    axes[1].set_xlabel("log₁₀(λ)")
    axes[1].set_ylabel("Avg |D| (Difficulty Magnitude)")
    axes[1].set_title("Difficulty Magnitude vs Regularization")
    axes[1].grid(True, alpha=0.3)

    # Slope magnitude vs lambda
    axes[2].plot(
        data["regularizer_exp"],
        data["avg_alpha_magnitude"],
        marker="o",
        color=colors[2],
        linewidth=2,
        markersize=4,
    )
    axes[2].set_xlabel("log₁₀(λ)")
    axes[2].set_ylabel("Avg |α| (Slope Magnitude)")
    axes[2].set_title("Slope Magnitude vs Regularization")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path, pdf_path = save_plot(output_dir / "parameter_magnitudes_vs_lambda")
    saved_plots["magnitudes"] = (png_path, pdf_path)
    plt.close()

    # Plot 3: All parameters on same plot (normalized)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize each parameter to [0, 1] for comparison
    c_norm = (data["avg_C_magnitude"] - data["avg_C_magnitude"].min()) / (
        data["avg_C_magnitude"].max() - data["avg_C_magnitude"].min()
    )
    d_norm = (data["avg_D_magnitude"] - data["avg_D_magnitude"].min()) / (
        data["avg_D_magnitude"].max() - data["avg_D_magnitude"].min()
    )
    a_norm = (data["avg_alpha_magnitude"] - data["avg_alpha_magnitude"].min()) / (
        data["avg_alpha_magnitude"].max() - data["avg_alpha_magnitude"].min()
    )

    ax.plot(
        data["regularizer_exp"],
        c_norm,
        marker="o",
        color=colors[0],
        linewidth=2,
        markersize=4,
        label="Capability (C)",
    )
    ax.plot(
        data["regularizer_exp"],
        d_norm,
        marker="s",
        color=colors[1],
        linewidth=2,
        markersize=4,
        label="Difficulty (D)",
    )
    ax.plot(
        data["regularizer_exp"],
        a_norm,
        marker="^",
        color=colors[2],
        linewidth=2,
        markersize=4,
        label="Slope (α)",
    )

    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("Normalized Magnitude (0-1)")
    ax.set_title("Normalized Parameter Magnitudes vs Regularization Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path, pdf_path = save_plot(output_dir / "normalized_parameters_vs_lambda")
    saved_plots["normalized"] = (png_path, pdf_path)
    plt.close()

    return saved_plots


def write_summary(
    data: pd.DataFrame,
    output_dir: Path,
    anchor_benchmark: str,
) -> Path:
    """
    Write analysis summary to text file.

    Args:
        data: DataFrame from run_regularization_sweep
        output_dir: Directory to save summary
        anchor_benchmark: Anchor benchmark used

    Returns:
        Path to summary file
    """
    summary_path = output_dir / "stability_summary.txt"

    with open(summary_path, "w") as f:
        f.write("Regularization Stability Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Anchor benchmark: {anchor_benchmark}\n")
        f.write(f"Regularization range: λ = 10^{data['regularizer_exp'].min():.1f} to 10^{data['regularizer_exp'].max():.1f}\n")
        f.write(f"Number of points: {len(data)}\n\n")

        f.write("Parameter Magnitude Ranges\n")
        f.write("-" * 30 + "\n")
        f.write(f"Capability |C|: {data['avg_C_magnitude'].min():.4f} - {data['avg_C_magnitude'].max():.4f}\n")
        f.write(f"Difficulty |D|: {data['avg_D_magnitude'].min():.4f} - {data['avg_D_magnitude'].max():.4f}\n")
        f.write(f"Slope |α|:      {data['avg_alpha_magnitude'].min():.4f} - {data['avg_alpha_magnitude'].max():.4f}\n\n")

        # Find the regularization value that gives the most stable (middle) estimates
        mid_idx = len(data) // 2
        f.write("Middle regularization point (λ = 10^{:.2f}):\n".format(data.iloc[mid_idx]["regularizer_exp"]))
        f.write(f"  Avg |C|: {data.iloc[mid_idx]['avg_C_magnitude']:.4f}\n")
        f.write(f"  Avg |D|: {data.iloc[mid_idx]['avg_D_magnitude']:.4f}\n")
        f.write(f"  Avg |α|: {data.iloc[mid_idx]['avg_alpha_magnitude']:.4f}\n\n")

        # Compute sensitivity (change per unit log-lambda)
        c_sensitivity = (data["avg_C_magnitude"].max() - data["avg_C_magnitude"].min()) / (
            data["regularizer_exp"].max() - data["regularizer_exp"].min()
        )
        d_sensitivity = (data["avg_D_magnitude"].max() - data["avg_D_magnitude"].min()) / (
            data["regularizer_exp"].max() - data["regularizer_exp"].min()
        )
        a_sensitivity = (data["avg_alpha_magnitude"].max() - data["avg_alpha_magnitude"].min()) / (
            data["regularizer_exp"].max() - data["regularizer_exp"].min()
        )

        f.write("Parameter Sensitivity (change per unit log₁₀(λ))\n")
        f.write("-" * 30 + "\n")
        f.write(f"Capability |C|: {c_sensitivity:.4f}\n")
        f.write(f"Difficulty |D|: {d_sensitivity:.4f}\n")
        f.write(f"Slope |α|:      {a_sensitivity:.4f}\n")

    return summary_path


def run_stability_analysis(
    output_dir: Path = None,
    lambda_min: float = -5,
    lambda_max: float = 0,
    n_points: int = 51,
    anchor_benchmark: str = "Winogrande",
) -> dict:
    """
    Run the complete stability analysis.

    Args:
        output_dir: Output directory (default: outputs/stability/)
        lambda_min: Minimum log10 regularization strength
        lambda_max: Maximum log10 regularization strength
        n_points: Number of points in sweep
        anchor_benchmark: Anchor benchmark to use

    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = Path("outputs/stability")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("REGULARIZATION STABILITY ANALYSIS")
    print("=" * 60)
    print()

    # Run the regularization sweep
    data = run_regularization_sweep(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        n_points=n_points,
        anchor_benchmark=anchor_benchmark,
    )

    # Save raw data
    data_path = output_dir / "regularization_sweep_data.csv"
    data.to_csv(data_path, index=False)
    print(f"\nSaved sweep data to: {data_path}")

    # Create plots
    print("\nCreating visualizations...")
    colors = setup_custom_style()
    saved_plots = create_stability_plots(data, output_dir, colors)

    for name, (png, pdf) in saved_plots.items():
        print(f"  - {name}: {png.name}, {pdf.name}")

    # Write summary
    summary_path = write_summary(data, output_dir, anchor_benchmark)
    print(f"\nSaved summary to: {summary_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")

    return {
        "data": data,
        "plots": saved_plots,
        "summary_path": summary_path,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze regularization stability for benchmark stitching model"
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=-5,
        help="Minimum log10 regularization strength (default: -5)",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=0,
        help="Maximum log10 regularization strength (default: 0)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=51,
        help="Number of points in regularization sweep (default: 51)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="Winogrande",
        help="Anchor benchmark to use (default: Winogrande)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/stability/)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_stability_analysis(
        output_dir=output_dir,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        n_points=args.n_points,
        anchor_benchmark=args.anchor,
    )
