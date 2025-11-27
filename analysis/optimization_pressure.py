#!/usr/bin/env python3
"""
Benchmark anchor approach for optimization pressure analysis.

This script implements the benchmark anchor approach to analyze optimization pressure
by comparing model capabilities between optimized and unoptimized data partitions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import scores_df
from fit import fit_statistical_model


# ============================================================================
# STYLING SETUP
# ============================================================================


def setup_custom_style():
    """Set up custom graph styling for all plots."""
    # Custom color palette
    custom_colors = [
        '#00A5A6',  # teal
        '#E03D90',  # pink
        '#FC6538',  # orange
        '#6A3ECB',  # purple
        '#0058DC',  # blue
        '#EA8D00',  # yellow
        '#B087F4',  # lightPurple
        '#279E27',  # green
        '#009AF1',  # lightBlue
        '#015D90',  # darkBlue
        '#EA4831',  # red
        '#E1C700',  # yellow2
        '#46FFFF',  # turquoise
        '#63F039',  # lightGreen
    ]

    sns.set_palette(custom_colors)

    # Seaborn global settings
    sns.set_theme(
        style="whitegrid",
        palette=custom_colors,
        context="notebook"
    )

    # Matplotlib global settings (rcParams)
    plt.rcParams.update({
        # Figure
        "figure.figsize": (8, 5),
        "figure.dpi": 120,

        # Axes
        "axes.titley": 1.02,
        "axes.titlesize": 14,
        "axes.titlelocation": 'center',
        "axes.titlepad": 0,
        "axes.labelsize": 12,
        "axes.labelpad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Ticks
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.top": False,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": False,

        # Legend
        "legend.fontsize": 10,
        "legend.loc": "upper left",
        "legend.frameon": True,
        "legend.borderaxespad": 0,

        # Lines and markers
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "lines.markeredgecolor": 'auto',
        "lines.markeredgewidth": 0.5,

        # Error bars
        "errorbar.capsize": 3,

        # Font
        "font.family": "Arial",
        "font.sans-serif": ["DejaVu Sans"],

        # Grid
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.color": "lightgray",
    })

    return custom_colors


def fit_subset_with_benchmark_anchor_ref_init(
    df_subset: pd.DataFrame,
    anchor_benchmark: str,
    slope_init: float = 1.0,
    regularization_strength: float = 0.1,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit on a subset using a benchmark anchor; initialize D and α from the reference fit
    for all benchmarks present; fix anchor α to its reference
    Returns (model_capabilities_df, benchmark_params_df).
    """
    df = df_subset.copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("df_subset is empty")

    valid_model_ids = df["model_id"].unique()
    benchmark_ids = df["benchmark_id"].unique()
    model_id_to_idx = {m: i for i, m in enumerate(valid_model_ids)}
    bench_id_to_idx = {b: i for i, b in enumerate(benchmark_ids)}

    num_models = len(valid_model_ids)
    num_benchmarks = len(benchmark_ids)

    model_idx_data = np.array([model_id_to_idx[m] for m in df["model_id"]])
    bench_idx_data = np.array([bench_id_to_idx[b] for b in df["benchmark_id"]])
    observed_scores = df["performance"].to_numpy()

    try:
        anchor_bench_id = df.loc[
            df["benchmark"] == anchor_benchmark, "benchmark_id"
        ].iloc[0]
    except IndexError as e:
        raise ValueError(f"Anchor '{anchor_benchmark}' not in subset") from e
    anchor_idx = bench_id_to_idx[anchor_bench_id]

    def logistic(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    rng = np.random.default_rng(random_state)
    initial_C = rng.normal(0.0, 0.1, size=num_models)

    bench_id_to_name = (
        df.drop_duplicates("benchmark_id")
        .set_index("benchmark_id")["benchmark"]
        .to_dict()
    )
    initial_D = np.zeros(num_benchmarks)
    initial_alpha_full = np.zeros(num_benchmarks)
    for b_id, idx in bench_id_to_idx.items():
        bname = bench_id_to_name[b_id]
        initial_D[idx] = float(REF_D_BY_NAME.get(bname, 0.0))
        initial_alpha_full[idx] = float(REF_ALPHA_BY_NAME.get(bname, slope_init))

    anchor_slope_ref = float(REF_ALPHA_BY_NAME.get(anchor_benchmark, 1.0))
    initial_alpha_free = np.delete(initial_alpha_full, anchor_idx)

    def split_params(theta: np.ndarray):
        C = theta[:num_models]
        D = theta[num_models : num_models + num_benchmarks]
        alpha_free = theta[num_models + num_benchmarks :]
        alpha = np.insert(alpha_free, anchor_idx, anchor_slope_ref)
        return C, D, alpha

    def residuals(theta: np.ndarray) -> np.ndarray:
        C, D, alpha = split_params(theta)
        preds = logistic(
            alpha[bench_idx_data] * (C[model_idx_data] - D[bench_idx_data])
        )
        resids = preds - observed_scores
        if regularization_strength > 0:
            reg = (
                regularization_strength
                * (
                    np.sum(C**2)
                    + np.sum(D**2)
                    + np.sum(alpha[np.arange(len(alpha)) != anchor_idx] ** 2)
                )
                / (num_models + num_benchmarks + num_benchmarks - 1)
            )
            reg_penalty = np.sqrt(reg) if reg > 0 else 0.0
            return np.append(resids, reg_penalty)
        return resids

    initial_theta = np.concatenate([initial_C, initial_D, initial_alpha_free])

    lower_bounds = np.concatenate(
        [
            np.full(num_models, -10.0),
            np.full(num_benchmarks, -10.0),
            np.full(num_benchmarks - 1, 0.1),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.full(num_models, 10.0),
            np.full(num_benchmarks, 10.0),
            np.full(num_benchmarks - 1, 10.0),
        ]
    )

    result = least_squares(
        residuals,
        initial_theta,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        verbose=0,
    )

    theta_hat = result.x
    C_hat = theta_hat[:num_models]
    D_hat = theta_hat[num_models : num_models + num_benchmarks]
    alpha_free_hat = theta_hat[num_models + num_benchmarks :]
    alpha_hat = np.insert(alpha_free_hat, anchor_idx, anchor_slope_ref)

    # anchor_D_ref = float(REF_D_BY_NAME.get(anchor_benchmark, 0.0))
    # shift = D_hat[anchor_idx] - anchor_D_ref
    # C_hat = C_hat - shift
    # D_hat = D_hat - shift

    id_to_model = (
        df.drop_duplicates("model_id").set_index("model_id")["model"].to_dict()
    )
    model_capabilities_df = (
        pd.DataFrame({"model_id": valid_model_ids, "estimated_capability": C_hat})
        .assign(model=lambda d: d["model_id"].map(id_to_model))
        .sort_values("estimated_capability", ascending=False)
        .reset_index(drop=True)
    )

    bench_params_df = (
        pd.DataFrame(
            {
                "benchmark_id": benchmark_ids,
                "estimated_difficulty": D_hat,
                "estimated_slope": alpha_hat,
            }
        )
        .assign(benchmark_name=lambda d: d["benchmark_id"].map(bench_id_to_name))
        .sort_values("estimated_difficulty")
        .reset_index(drop=True)
    )

    return model_capabilities_df, bench_params_df


def ensure_anchor_present(
    part_df: pd.DataFrame, anchor_name: str, source_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Include the anchor benchmark rows if missing from the partition.

    Args:
        part_df: Partition dataframe to check
        anchor_name: Name of the anchor benchmark
        source_df: Source dataframe to use for adding anchor (defaults to scores_df)
    """
    if anchor_name in part_df["benchmark"].unique():
        return part_df
    if source_df is None:
        source_df = scores_df
    return pd.concat(
        [part_df, source_df[source_df["benchmark"] == anchor_name]], ignore_index=True
    )


def _simple_avg(series_list: list[pd.Series]) -> pd.Series:
    """Simple average of a list of pandas Series."""
    if not series_list:
        return pd.Series(dtype=float)
    return pd.concat(series_list, axis=1).mean(axis=1)


def save_plot(output_path: Path, dpi: int = 300, bbox_inches: str = "tight"):
    """Save the current plot as both PNG and PDF."""
    # Save PNG
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)

    # Save PDF
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches=bbox_inches)

    return png_path, pdf_path


def save_results(results: dict, output_dir: Path = None):
    """
    Save analysis results to files.

    Args:
        results: Dictionary containing analysis results
        output_dir: Output directory path (defaults to outputs/optimization_pressure/)
    """
    if output_dir is None:
        output_dir = Path("outputs/optimization_pressure")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create supplementary subfolder for less important outputs
    supplementary_dir = output_dir / "supplementary"
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    # Save approach 1 results (to supplementary folder)
    comp1 = results["comp1"]
    comp1.to_csv(supplementary_dir / "approach1_model_capabilities.csv", index=False)

    # Save approach 1 summary (to supplementary folder)
    summary1 = results["summary1"]
    with open(supplementary_dir / "approach1_summary.txt", "w") as f:
        f.write("Benchmark Anchor Approach 1 Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of models: {summary1['n_models_overlap']}\n")
        f.write(f"Mean difference (opt - unopt): {summary1['mean_diff']:.6f}\n")
        f.write(f"Median difference (opt - unopt): {summary1['median_diff']:.6f}\n")
        f.write(f"Standard deviation: {summary1['std_diff']:.6f}\n")
        f.write(f"T-test p-value: {summary1['t_pvalue']:.6f}\n")
        f.write(f"Wilcoxon p-value: {summary1['wilcoxon_pvalue']:.6f}\n")
        f.write(f"Pearson correlation: {summary1['pearson_r']:.6f}\n")
        f.write(f"Spearman correlation: {summary1['spearman_rho']:.6f}\n")

    # Save execution summary (to supplementary folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(supplementary_dir / f"execution_summary_{timestamp}.txt", "w") as f:
        f.write("Benchmark Anchor Approach Execution Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Approach 1 - Top-10 anchors by coverage\n")
        f.write(
            f"  - Optimized anchors: {results['approach1_results']['n_opt_anchors']}\n"
        )
        f.write(
            f"  - Unoptimized anchors: {results['approach1_results']['n_unopt_anchors']}\n"
        )
        f.write(
            f"  - Models analyzed: {results['approach1_results']['n_models_overlap']}\n"
        )
        f.write(f"  - Mean delta: {results['approach1_results']['delta_mean']:.6f}\n")

    print(f"\nResults saved to: {output_dir}")
    print(
        f"  - supplementary/approach1_model_capabilities.csv: Model capability comparisons"
    )
    print(
        f"  - supplementary/approach1_summary.txt: Statistical summary for approach 1"
    )
    print(f"  - supplementary/execution_summary_{timestamp}.txt: Execution details")


def create_analysis_plots(
    results: dict,
    opt_caps_by_anchor: list,
    unopt_caps_by_anchor: list,
    anchors20: list,
    output_dir: Path = None,
    colors=None,
):
    """
    Create two analysis plots based on opt_caps_by_anchor data.

    Args:
        results: Dictionary containing analysis results
        opt_caps_by_anchor: List of capability estimates for optimized data across anchors
        unopt_caps_by_anchor: List of capability estimates for unoptimized data across anchors
        anchors20: List of anchor benchmark names
        output_dir: Output directory path (defaults to outputs/optimization_pressure/)
        colors: Custom color palette (optional)
    """
    if output_dir is None:
        output_dir = Path("outputs/optimization_pressure")

    # Use default colors if not provided
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create supplementary subfolder for less important outputs
    supplementary_dir = output_dir / "supplementary"
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Creating analysis plots ===")

    # Plot 1: Distribution of model capability differences (averaging over anchors)
    plt.figure(figsize=(10, 6))
    comp1 = results["comp1"]
    plt.hist(comp1["diff"], bins=30, alpha=0.7, color=colors[0], edgecolor="black")
    plt.axvline(
        comp1["diff"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {comp1['diff'].mean():.4f}",
    )
    plt.axvline(0, color="black", linestyle="-", alpha=0.5, label="No difference")
    plt.xlabel("Model Capability Difference (Optimized - Unoptimized)", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.title("Distribution of Model Capability Differences\n(Averaging over anchors)", fontsize=24)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=20)

    # Save plot 1
    plot1_png, plot1_pdf = save_plot(
        output_dir / "model_capability_differences_histogram"
    )
    plt.show()

    # Plot 1.5: Scatter plot of optimized vs unoptimized capabilities
    plt.figure(figsize=(10, 8))
    plt.scatter(comp1["cap_unopt"], comp1["cap_opt"], alpha=0.7, s=50, color=colors[0])

    # Add y=x diagonal line
    min_val = min(comp1["cap_unopt"].min(), comp1["cap_opt"].min())
    max_val = max(comp1["cap_unopt"].max(), comp1["cap_opt"].max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.7,
        linewidth=2,
        label="y=x (no difference)",
    )

    # Add labels for specific models
    labeled_models = [
        "DeepSeek-R1",
        "claude-3-5-sonnet-20241022",
        # "claude-sonnet-4-5-20250929",
        "gemini-2.5-pro-exp-03-25",
        "gpt-5-2025-08-07_high",
        "gpt-4o-2024-08-06",
        # "grok-4-0709",
    ]
    for model in labeled_models:
        if model in comp1["model"].values:
            model_data = comp1[comp1["model"] == model].iloc[0]
            plt.annotate(
                model,
                xy=(model_data["cap_unopt"], model_data["cap_opt"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=20,
                alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.6)
            )

    # Add correlation coefficient
    correlation = comp1["cap_opt"].corr(comp1["cap_unopt"])
    # plt.text(
    #     0.05,
    #     0.95,
    #     f"Correlation: {correlation:.3f}",
    #     transform=plt.gca().transAxes,
    #     fontsize=12,
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    # )

    plt.xlabel("Unoptimized Capability Score", fontsize=20)
    plt.ylabel("Optimized Capability Score", fontsize=20)
    plt.title(
        "Model Capabilities: Optimized vs Unoptimized",
        fontsize=24
    )
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    # Save plot 1.5
    plot1_5_png, plot1_5_pdf = save_plot(output_dir / "capability_scatter_plot")
    plt.show()

    # Plot 1.6: Individual anchor scatter plots
    # Use recent benchmarks that are more likely to exist when date filtering is enabled
    # Fallback to older ones if newer ones aren't available
    potential_plot_anchors = [
        "WeirdML",  # Likely recent
        "SimpleBench",  # Likely recent
        "Balrog",  # Likely recent
        "LiveBench",  # Recent
        "MMLU",  # Older but widely used
        "GSM8K",  # Older but widely used
        "Winogrande",  # Older
    ]
    available_anchors = []

    # Check which anchors are available in our data
    for anchor in potential_plot_anchors:
        if anchor in anchors20:
            available_anchors.append(anchor)

    if available_anchors:
        print(f"\nCreating individual anchor scatter plots for: {available_anchors}")

        # Create subplots for individual anchors
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Models to label
        labeled_models = [
            "DeepSeek-R1",
            "claude-3-5-sonnet-20241022",
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro-exp-03-25",
            "gpt-5-2025-08-07_high",
            "gpt-4o-2024-08-06",
            "grok-4-0709",
        ]

        # Dictionary to store data for each anchor
        anchor_data = {}

        for i, anchor in enumerate(available_anchors):
            if i >= len(axes):
                break

            # Find the anchor index in our data
            anchor_idx = anchors20.index(anchor)
            opt_caps = opt_caps_by_anchor[anchor_idx]
            unopt_caps = unopt_caps_by_anchor[anchor_idx]

            # Find common models for this anchor
            common_models = opt_caps.index.intersection(unopt_caps.index)
            if len(common_models) > 0:
                opt_common = opt_caps.loc[common_models]
                unopt_common = unopt_caps.loc[common_models]

                # Store common models list
                anchor_data[anchor] = {
                    "common_models": sorted(common_models.tolist()),
                    "n_common": len(common_models),
                    "opt_caps": opt_common,
                    "unopt_caps": unopt_common,
                }

                # Create scatter plot for this anchor
                ax = axes[i]
                ax.scatter(unopt_common, opt_common, alpha=0.7, s=30, color=colors[0])

                # Add y=x diagonal line
                min_val = min(unopt_common.min(), opt_common.min())
                max_val = max(unopt_common.max(), opt_common.max())
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "k--",
                    alpha=0.7,
                    linewidth=1,
                )

                # Add labels for specific models
                for model in labeled_models:
                    if model in opt_common.index:
                        ax.annotate(
                            model,
                            xy=(unopt_common[model], opt_common[model]),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=6,
                            alpha=0.8,
                            bbox=dict(
                                boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6
                            ),
                        )

                # Identify models below the diagonal
                below_diagonal = opt_common[opt_common < unopt_common]
                anchor_data[anchor]["below_diagonal"] = below_diagonal

                # Add correlation coefficient
                correlation = opt_common.corr(unopt_common)
                mean_diff = (opt_common - unopt_common).mean()
                n_above_diag = sum(opt_common > unopt_common)

                ax.text(
                    0.05,
                    0.95,
                    f"r={correlation:.3f}\nΔ={mean_diff:.3f}\n{n_above_diag}/{len(common_models)} above",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

                ax.set_xlabel("Unoptimized Capability")
                ax.set_ylabel("Optimized Capability")
                ax.set_title(f"{anchor}\n(n={len(common_models)} models)")
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(available_anchors), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save individual anchor plots (to supplementary folder)
        plot1_6_png, plot1_6_pdf = save_plot(
            supplementary_dir / "individual_anchor_scatter_plots"
        )
        plt.show()

        print(f"\nIndividual anchor plots saved to: {plot1_6_png} and {plot1_6_pdf}")

        # Save common models lists for each anchor (to supplementary folder)
        print("\n=== Saving common models lists for specific anchors ===")
        for anchor in available_anchors:
            if anchor in anchor_data:
                data = anchor_data[anchor]
                # Save common models list
                models_df = pd.DataFrame(
                    {
                        "model": data["common_models"],
                    }
                )
                models_csv_path = (
                    supplementary_dir / f"common_models_{anchor.replace(' ', '_')}.csv"
                )
                models_df.to_csv(models_csv_path, index=False)
                print(
                    f"  - {anchor}: {data['n_common']} common models saved to {models_csv_path.name}"
                )

        # Print models below diagonal for each anchor
        print("\n=== Models below diagonal (opt < unopt) for each specific anchor ===")
        for anchor in available_anchors:
            if anchor in anchor_data:
                data = anchor_data[anchor]
                below_diagonal = data["below_diagonal"]
                opt_common = data["opt_caps"]
                unopt_common = data["unopt_caps"]

                print(f"\n{anchor}:")
                print(f"  Total common models: {data['n_common']}")
                print(f"  Models below diagonal: {len(below_diagonal)}")
                if len(below_diagonal) > 0:
                    # Create a detailed dataframe
                    below_df = pd.DataFrame(
                        {
                            "model": below_diagonal.index,
                            "opt_capability": below_diagonal.values,
                            "unopt_capability": [
                                unopt_common[m] for m in below_diagonal.index
                            ],
                            "difference": [
                                below_diagonal[m] - unopt_common[m]
                                for m in below_diagonal.index
                            ],
                        }
                    )
                    # Sort by difference (most negative first)
                    below_df = below_df.sort_values("difference")

                    # Save to CSV (to supplementary folder)
                    below_csv_path = (
                        supplementary_dir
                        / f"below_diagonal_{anchor.replace(' ', '_')}.csv"
                    )
                    below_df.to_csv(below_csv_path, index=False)
                    print(f"  Saved to: {below_csv_path.name}")

                    # Print the list
                    for _, row in below_df.iterrows():
                        print(
                            f"    - {row['model']}: opt={row['opt_capability']:.3f}, "
                            f"unopt={row['unopt_capability']:.3f}, diff={row['difference']:.3f}"
                        )
                else:
                    print("    (none)")
    else:
        print(
            f"\nNone of the requested anchors ({potential_plot_anchors}) were found in the analysis."
        )
        print(
            f"Available anchors: {anchors20[:10]}..."
        )  # Show first 10 available anchors

    # Plot 2: Mean model capability difference across anchors
    # Calculate mean difference for each anchor
    anchor_differences = []
    anchor_names = []
    anchor_types = []  # Track whether anchor is optimized or unoptimized

    # Get benchmark metadata to determine anchor types
    bench_meta = scores_df.drop_duplicates("benchmark")[["benchmark", "optimized"]]
    opt_bench_set = set(bench_meta[bench_meta["optimized"]]["benchmark"])

    for i, anchor in enumerate(anchors20):
        opt_caps = opt_caps_by_anchor[i]
        unopt_caps = unopt_caps_by_anchor[i]

        # Find common models for this anchor
        common_models = opt_caps.index.intersection(unopt_caps.index)
        if len(common_models) > 0:
            opt_common = opt_caps.loc[common_models]
            unopt_common = unopt_caps.loc[common_models]
            mean_diff = (opt_common - unopt_common).mean()
            anchor_differences.append(mean_diff)
            anchor_names.append(anchor)
            # Determine if this anchor is optimized or unoptimized
            is_optimized = anchor in opt_bench_set
            anchor_types.append("Optimized" if is_optimized else "Unoptimized")

    # Create plot 2
    plt.figure(figsize=(14, 8))
    # Use different colors for optimized vs unoptimized anchors
    bar_colors = [
        colors[7] if anchor_type == "Optimized" else colors[10]  # green for optimized, red for unoptimized
        for anchor_type in anchor_types
    ]
    plt.bar(range(len(anchor_names)), anchor_differences, color=bar_colors, alpha=0.7)
    plt.axhline(0, color="black", linestyle="-", alpha=0.5, label="No difference")
    plt.axhline(
        np.mean(anchor_differences),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall mean: {np.mean(anchor_differences):.4f}",
    )

    plt.xlabel("Anchor Benchmark")
    plt.ylabel("Mean Model Capability Difference (Optimized - Unoptimized)")
    plt.title(
        "Mean Model Capability Difference Across Anchors\n(Averaging over models first)"
    )
    plt.xticks(range(len(anchor_names)), anchor_names, rotation=45, ha="right")

    # Add custom legend for anchor types
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors[7], alpha=0.7, label="Optimized Anchors"),
        Patch(facecolor=colors[10], alpha=0.7, label="Unoptimized Anchors"),
        plt.Line2D(
            [0], [0], color="black", linestyle="-", alpha=0.5, label="No difference"
        ),
        plt.Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Overall mean: {np.mean(anchor_differences):.4f}",
        ),
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot 2
    plot2_png, plot2_pdf = save_plot(output_dir / "anchor_differences_barplot")
    plt.show()

    # Save anchor differences data (to supplementary folder)
    anchor_df = pd.DataFrame(
        {
            "anchor_benchmark": anchor_names,
            "mean_difference": anchor_differences,
            "anchor_type": anchor_types,
        }
    )
    anchor_df.to_csv(supplementary_dir / "anchor_differences.csv", index=False)

    # Print summary statistics
    print("\nPlot 1 Summary (Model differences):")
    print(f"  Mean difference: {comp1['diff'].mean():.6f}")
    print(f"  Median difference: {comp1['diff'].median():.6f}")
    print(f"  Std deviation: {comp1['diff'].std():.6f}")
    print(f"  Range: [{comp1['diff'].min():.6f}, {comp1['diff'].max():.6f}]")

    print("\nPlot 1.5 Summary (Capability scatter plot):")
    print(
        f"  Correlation (optimized vs unoptimized): {comp1['cap_opt'].corr(comp1['cap_unopt']):.6f}"
    )
    print(
        f"  Optimized capabilities - Mean: {comp1['cap_opt'].mean():.6f}, Std: {comp1['cap_opt'].std():.6f}"
    )
    print(
        f"  Unoptimized capabilities - Mean: {comp1['cap_unopt'].mean():.6f}, Std: {comp1['cap_unopt'].std():.6f}"
    )
    print(
        f"  Points above diagonal (optimized > unoptimized): {sum(comp1['cap_opt'] > comp1['cap_unopt'])}/{len(comp1)} ({100*sum(comp1['cap_opt'] > comp1['cap_unopt'])/len(comp1):.1f}%)"
    )

    print("\nPlot 2 Summary (Anchor differences):")
    print(f"  Number of anchors: {len(anchor_differences)}")
    print(f"  Mean difference across anchors: {np.mean(anchor_differences):.6f}")
    print(f"  Std deviation across anchors: {np.std(anchor_differences):.6f}")
    print(
        f"  Range: [{np.min(anchor_differences):.6f}, {np.max(anchor_differences):.6f}]"
    )

    # Summary by anchor type
    opt_anchors = [
        diff
        for diff, anchor_type in zip(anchor_differences, anchor_types)
        if anchor_type == "Optimized"
    ]
    unopt_anchors = [
        diff
        for diff, anchor_type in zip(anchor_differences, anchor_types)
        if anchor_type == "Unoptimized"
    ]

    print("\n  By anchor type:")
    print(
        f"    Optimized anchors ({len(opt_anchors)}): mean={np.mean(opt_anchors):.6f}, std={np.std(opt_anchors):.6f}"
    )
    print(
        f"    Unoptimized anchors ({len(unopt_anchors)}): mean={np.mean(unopt_anchors):.6f}, std={np.std(unopt_anchors):.6f}"
    )

    print("\nPlots saved to:")
    print(f"  - {plot1_png} and {plot1_pdf}")
    print(f"  - {plot1_5_png} and {plot1_5_pdf}")
    if "plot1_6_png" in locals():
        print(f"  - {plot1_6_png} and {plot1_6_pdf} (supplementary)")
    print(f"  - {plot2_png} and {plot2_pdf}")
    print(f"  - {supplementary_dir / 'anchor_differences.csv'} (supplementary)")


def run_benchmark_anchor_approach(
    num_anchors=10, random_seed=None, min_benchmark_date=None
):
    """
    Run the benchmark anchor approach analysis.

    This function implements Approach 1:
    Top-N anchors by coverage within each partition

    Args:
        num_anchors: Number of top anchors to use from each partition (default: 10)
        random_seed: Random seed for reproducibility (default: None)
        min_benchmark_date: Minimum benchmark release date as string (e.g., "2024-01-01")
                          or pd.Timestamp. If None, no date filtering is applied (default: None)
    """
    print("Running Benchmark anchor approach...")

    # Apply date filtering if specified
    working_df = scores_df.copy()
    if min_benchmark_date is not None:
        if isinstance(min_benchmark_date, str):
            min_benchmark_date = pd.to_datetime(min_benchmark_date)
        print(f"\nFiltering benchmarks by release date >= {min_benchmark_date.date()}")

        # Check if benchmark_release_date column exists
        if "benchmark_release_date" not in working_df.columns:
            raise ValueError(
                "benchmark_release_date column not found in scores_df. "
                "Please ensure data_loader.py has merged benchmark release dates."
            )

        # Check for missing dates
        missing_dates = working_df["benchmark_release_date"].isna().sum()
        if missing_dates > 0:
            print(
                f"  Warning: {missing_dates} rows have missing benchmark_release_date"
            )
            print("  These rows will be excluded from date-filtered analysis.")

        # Filter to only include benchmarks from min_benchmark_date onwards
        # Exclude rows with missing dates
        working_df = working_df[
            (working_df["benchmark_release_date"].notna())
            & (working_df["benchmark_release_date"] >= min_benchmark_date)
        ].copy()

        n_before = len(scores_df)
        n_after = len(working_df)
        n_benchmarks_before = scores_df["benchmark"].nunique()
        n_benchmarks_after = working_df["benchmark"].nunique()

        print(f"  Rows: {n_before} -> {n_after} ({100*n_after/n_before:.1f}%)")
        print(
            f"  Benchmarks: {n_benchmarks_before} -> {n_benchmarks_after} ({100*n_benchmarks_after/n_benchmarks_before:.1f}%)"
        )

        # Show distribution by optimized status
        opt_count = (working_df["optimized"]).sum()
        unopt_count = (~working_df["optimized"]).sum()
        opt_benchmarks = working_df[working_df["optimized"]]["benchmark"].nunique()
        unopt_benchmarks = working_df[~working_df["optimized"]]["benchmark"].nunique()
        print(f"  Optimized benchmarks: {opt_benchmarks} ({opt_count} rows)")
        print(f"  Unoptimized benchmarks: {unopt_benchmarks} ({unopt_count} rows)")

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"Using random seed: {random_seed}")

    # Reference fit anchored on a benchmark with good coverage to cache per-benchmark difficulty (D) and slope (α)
    # Auto-select anchor: prefer benchmarks with good coverage, and if date filtering, prefer recent ones
    print("Creating reference fit...")
    # Find benchmark with best coverage
    bench_coverage = (
        working_df.groupby("benchmark")["model"].nunique().sort_values(ascending=False)
    )
    potential_anchors = [
        "Winogrande",
        "MMLU",
        "HellaSwag",
        "SimpleBench",
        "WeirdML",
        "LiveBench",
    ]

    # Select anchor that exists in filtered data, prefer common ones
    anchor_benchmark = None
    for candidate in potential_anchors + bench_coverage.index.tolist():
        if candidate in bench_coverage.index:
            anchor_benchmark = candidate
            break

    if anchor_benchmark is None:
        # Fallback to most covered benchmark
        anchor_benchmark = bench_coverage.index[0]

    print(
        f"Using '{anchor_benchmark}' as reference anchor (coverage: {bench_coverage[anchor_benchmark]} models)"
    )

    ref_df, ref_cap, ref_bench = fit_statistical_model(
        working_df,
        anchor_mode="benchmark",
        anchor_benchmark=anchor_benchmark,
        anchor_difficulty=0.0,
        anchor_slope=1.0,
    )

    global REF_D_BY_NAME, REF_ALPHA_BY_NAME
    REF_D_BY_NAME = dict(
        zip(ref_bench["benchmark_name"], ref_bench["estimated_difficulty"])
    )
    REF_ALPHA_BY_NAME = dict(
        zip(ref_bench["benchmark_name"], ref_bench["estimated_slope"])
    )

    print(f"Reference fit cached: {len(REF_D_BY_NAME)} benchmarks.")

    # Partitions
    opt_df = working_df[working_df["optimized"]].copy()
    unopt_df = working_df[~working_df["optimized"]].copy()

    # Approach 1: Top-N by coverage within each partition
    print(f"\n=== Approach 1: Top-{num_anchors} anchors by coverage ===")
    cov_opt = (
        opt_df.groupby("benchmark")["model"].nunique().sort_values(ascending=False)
    )
    cov_unopt = (
        unopt_df.groupby("benchmark")["model"].nunique().sort_values(ascending=False)
    )
    opt_anchors = cov_opt.head(num_anchors).index.tolist()
    unopt_anchors = cov_unopt.head(num_anchors).index.tolist()
    anchors_total = opt_anchors + unopt_anchors

    # Run fits for each anchor on both partitions (anchor enforced if missing)
    opt_caps_by_anchor = []
    unopt_caps_by_anchor = []
    for anc in anchors_total:
        # Diagnostic printing for requested anchors (only if they exist in filtered data)
        diagnostic_anchors = [
            "MMLU",
            "WeirdML",
            "GSM8K",
            "Winogrande",
            "HellaSwag",
            "Balrog",
            "GPQA Diamond",
            "SimpleBench",
            "LiveBench",
        ]
        if anc in diagnostic_anchors:
            print(f"\n{'='*60}")
            print(f"DIAGNOSTIC: Fitting with anchor: {anc}")
            print(f"{'='*60}")

            # For optimized partition
            print(f"\n--- OPTIMIZED PARTITION ---")
            models_before_opt = set(opt_df["model"].unique())
            print(
                f"Models in partition before ensure_anchor_present: {len(models_before_opt)}"
            )

            opt_with_anchor = ensure_anchor_present(opt_df, anc, source_df=working_df)
            models_after_opt = set(opt_with_anchor["model"].unique())
            print(
                f"Models in partition after ensure_anchor_present: {len(models_after_opt)}"
            )

            added_models_opt = models_after_opt - models_before_opt
            if added_models_opt:
                print(f"Models ADDED by ensure_anchor_present: {len(added_models_opt)}")
                print(
                    f"  Added models: {sorted(added_models_opt)[:10]}{'...' if len(added_models_opt) > 10 else ''}"
                )
            else:
                print(
                    "No models added by ensure_anchor_present (anchor already present)"
                )

            # For unoptimized partition
            print(f"\n--- UNOPTIMIZED PARTITION ---")
            models_before_unopt = set(unopt_df["model"].unique())
            print(
                f"Models in partition before ensure_anchor_present: {len(models_before_unopt)}"
            )

            unopt_with_anchor = ensure_anchor_present(
                unopt_df, anc, source_df=working_df
            )
            models_after_unopt = set(unopt_with_anchor["model"].unique())
            print(
                f"Models in partition after ensure_anchor_present: {len(models_after_unopt)}"
            )

            added_models_unopt = models_after_unopt - models_before_unopt
            if added_models_unopt:
                print(
                    f"Models ADDED by ensure_anchor_present: {len(added_models_unopt)}"
                )
                print(
                    f"  Added models: {sorted(added_models_unopt)[:10]}{'...' if len(added_models_unopt) > 10 else ''}"
                )
            else:
                print(
                    "No models added by ensure_anchor_present (anchor already present)"
                )
        else:
            opt_with_anchor = ensure_anchor_present(opt_df, anc, source_df=working_df)
            unopt_with_anchor = ensure_anchor_present(
                unopt_df, anc, source_df=working_df
            )

        mcap_opt, _ = fit_subset_with_benchmark_anchor_ref_init(opt_with_anchor, anc)
        mcap_unopt, _ = fit_subset_with_benchmark_anchor_ref_init(
            unopt_with_anchor, anc
        )

        # Diagnostic printing for requested anchors - after fitting
        if anc in diagnostic_anchors:
            print(f"\n--- AFTER FITTING ---")
            print(f"Optimized: {len(mcap_opt)} models with capability estimates")
            print(f"Unoptimized: {len(mcap_unopt)} models with capability estimates")

            # Find common models
            opt_models = set(mcap_opt["model"].unique())
            unopt_models = set(mcap_unopt["model"].unique())
            common = opt_models.intersection(unopt_models)
            print(f"Common models (with estimates in both partitions): {len(common)}")
            print(f"{'='*60}\n")

        opt_caps_by_anchor.append(mcap_opt.set_index("model")["estimated_capability"])
        unopt_caps_by_anchor.append(
            mcap_unopt.set_index("model")["estimated_capability"]
        )

    # Average across anchors in each partition
    opt_avg = pd.concat(opt_caps_by_anchor, axis=1).mean(axis=1)
    unopt_avg = pd.concat(unopt_caps_by_anchor, axis=1).mean(axis=1)

    # Difference over overlapping models
    common = opt_avg.index.intersection(unopt_avg.index)
    delta = opt_avg.loc[common] - unopt_avg.loc[common]

    approach1_results = {
        "approach": 1,
        "n_anchors": num_anchors,
        "n_opt_anchors": len(opt_anchors),
        "n_unopt_anchors": len(unopt_anchors),
        "n_models_overlap": int(len(common)),
        "delta_mean": float(delta.mean()),
        "delta_median": float(delta.median()),
        "delta_std": float(delta.std(ddof=0)),
    }
    print(approach1_results)

    # Statistical tests for Approach 1
    print("\n=== Statistical Analysis for Approach 1 ===")
    comp1 = (
        pd.DataFrame({"cap_opt": opt_avg, "cap_unopt": unopt_avg})
        .dropna()
        .assign(diff=lambda d: d["cap_opt"] - d["cap_unopt"])
        .reset_index()
        .rename(columns={"index": "model"})
    )

    # Summary stats (paired)
    t_stat1, t_p1 = ttest_rel(comp1["cap_opt"], comp1["cap_unopt"])
    w_stat1, w_p1 = wilcoxon(comp1["cap_opt"], comp1["cap_unopt"])
    r_pearson1, p_pearson1 = pearsonr(comp1["cap_opt"], comp1["cap_unopt"])
    r_spear1, p_spear1 = spearmanr(comp1["cap_opt"], comp1["cap_unopt"])
    diffs = comp1["diff"].to_numpy()

    summary1 = {
        "approach": 1,
        "n_models_overlap": int(len(comp1)),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "std_diff": float(diffs.std(ddof=0)),
        "t_pvalue": float(t_p1),
        "wilcoxon_pvalue": float(w_p1),
        "pearson_r": float(r_pearson1),
        "spearman_rho": float(r_spear1),
    }
    print(summary1)

    # Show top and bottom differences
    print("\nTop 10 (opt - unopt):")
    top10 = comp1.nlargest(10, "diff")[["model", "cap_opt", "cap_unopt", "diff"]]
    print(top10.to_string(index=False))

    print("\nBottom 10 (opt - unopt):")
    bottom10 = comp1.nsmallest(10, "diff")[["model", "cap_opt", "cap_unopt", "diff"]]
    print(bottom10.to_string(index=False))

    results = {
        "approach1_results": approach1_results,
        "summary1": summary1,
        "comp1": comp1,
        "opt_avg": opt_avg,
        "unopt_avg": unopt_avg,
        "delta": delta,
    }

    # Save results to files
    save_results(results)

    # Create analysis plots (get colors from current style)
    current_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    create_analysis_plots(
        results, opt_caps_by_anchor, unopt_caps_by_anchor, anchors_total, colors=current_colors
    )

    return results, anchors_total


def run_robustness_checks(min_benchmark_date=None):
    """
    Run robustness checks with different numbers of anchors.

    Args:
        min_benchmark_date: Minimum benchmark release date as string (e.g., "2024-01-01")
                          or pd.Timestamp. If None, no date filtering is applied (default: None)
    """
    print("Running robustness checks with different numbers of anchors...")

    anchor_counts = [5, 10, 15]
    results_summary = []

    for num_anchors in anchor_counts:
        print(f"\n{'='*60}")
        print(f"ROBUSTNESS CHECK: {num_anchors} anchors")
        print(f"{'='*60}")

        # Run analysis with this number of anchors
        results = run_benchmark_anchor_approach(
            num_anchors, min_benchmark_date=min_benchmark_date
        )

        # Store summary for comparison
        summary = {
            "n_anchors": num_anchors,
            "delta_mean": results["approach1_results"]["delta_mean"],
            "delta_std": results["approach1_results"]["delta_std"],
            "n_models": results["approach1_results"]["n_models_overlap"],
            "t_pvalue": results["summary1"]["t_pvalue"],
            "wilcoxon_pvalue": results["summary1"]["wilcoxon_pvalue"],
        }
        results_summary.append(summary)

    # Print comparison table
    print(f"\n{'='*80}")
    print("ROBUSTNESS CHECK SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'Anchors':<8} {'Mean Δ':<12} {'Std Δ':<12} {'Models':<8} {'T-test p':<12} {'Wilcoxon p':<12}"
    )
    print("-" * 80)

    for summary in results_summary:
        print(
            f"{summary['n_anchors']:<8} "
            f"{summary['delta_mean']:<12.6f} "
            f"{summary['delta_std']:<12.6f} "
            f"{summary['n_models']:<8} "
            f"{summary['t_pvalue']:<12.2e} "
            f"{summary['wilcoxon_pvalue']:<12.2e}"
        )

    return results_summary


def run_permutation_test_analysis(
    anchor_benchmarks: list[str],
    num_permutations: int = 100,
    random_seed: int = 42,
    output_dir: Path = None,
    min_benchmark_date: str | pd.Timestamp | None = None,
):
    """
    Run permutation test by comparing random benchmark splits to opt/unopt splits.

    For each anchor benchmark, this function:
    1. Computes the observed difference (opt vs unopt partitions)
    2. Creates random benchmark partitions and computes null distribution
    3. Calculates p-value: how often random splits show >= observed effect

    Args:
        anchor_benchmarks: List of benchmark names to use as anchors
        num_permutations: Number of random permutations to test (default: 100)
        random_seed: Random seed for reproducibility (default: 42)
        output_dir: Output directory path (defaults to outputs/optimization_pressure/)
        min_benchmark_date: Minimum benchmark release date as string (e.g., "2024-01-01")
                          or pd.Timestamp. If None, no date filtering is applied (default: None)
    """
    if output_dir is None:
        output_dir = Path("outputs/optimization_pressure")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PERMUTATION TEST ANALYSIS")
    print(f"{'='*80}")
    print(f"Anchor benchmarks: {anchor_benchmarks}")
    print(f"Number of permutations: {num_permutations}")
    print(f"Random seed: {random_seed}")

    # Apply date filtering if specified
    working_df = scores_df.copy()
    if min_benchmark_date is not None:
        if isinstance(min_benchmark_date, str):
            min_benchmark_date = pd.to_datetime(min_benchmark_date)
        print(f"\nFiltering benchmarks by release date >= {min_benchmark_date.date()}")

        # Check if benchmark_release_date column exists
        if "benchmark_release_date" not in working_df.columns:
            raise ValueError(
                "benchmark_release_date column not found in scores_df. "
                "Please ensure data_loader.py has merged benchmark release dates."
            )

        # Check for missing dates
        missing_dates = working_df["benchmark_release_date"].isna().sum()
        if missing_dates > 0:
            print(
                f"  Warning: {missing_dates} rows have missing benchmark_release_date"
            )
            print("  These rows will be excluded from date-filtered analysis.")

        # Filter to only include benchmarks from min_benchmark_date onwards
        # Exclude rows with missing dates
        working_df = working_df[
            (working_df["benchmark_release_date"].notna())
            & (working_df["benchmark_release_date"] >= min_benchmark_date)
        ].copy()
        print(
            f"  Filtered to {len(working_df)} rows, {working_df['benchmark'].nunique()} benchmarks"
        )

    rng = np.random.default_rng(random_seed)

    # Get all benchmarks
    all_benchmarks = working_df["benchmark"].unique()
    opt_df = working_df[working_df["optimized"]].copy()
    unopt_df = working_df[~working_df["optimized"]].copy()

    opt_benchmarks = opt_df["benchmark"].unique()
    unopt_benchmarks = unopt_df["benchmark"].unique()

    print(f"\nDataset info:")
    print(f"  Total benchmarks: {len(all_benchmarks)}")
    print(f"  Optimized benchmarks: {len(opt_benchmarks)}")
    print(f"  Unoptimized benchmarks: {len(unopt_benchmarks)}")

    permutation_results = []

    for anchor in anchor_benchmarks:
        if anchor not in all_benchmarks:
            print(f"\nWarning: Anchor '{anchor}' not found in dataset, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Anchor: {anchor}")
        print(f"{'='*60}")

        # --- Observed difference (opt vs unopt) ---
        print("  Computing observed difference...")
        opt_with_anchor = ensure_anchor_present(opt_df, anchor, source_df=working_df)
        unopt_with_anchor = ensure_anchor_present(
            unopt_df, anchor, source_df=working_df
        )

        mcap_opt_obs, _ = fit_subset_with_benchmark_anchor_ref_init(
            opt_with_anchor, anchor
        )
        mcap_unopt_obs, _ = fit_subset_with_benchmark_anchor_ref_init(
            unopt_with_anchor, anchor
        )

        opt_caps_obs = mcap_opt_obs.set_index("model")["estimated_capability"]
        unopt_caps_obs = mcap_unopt_obs.set_index("model")["estimated_capability"]

        common_models_obs = opt_caps_obs.index.intersection(unopt_caps_obs.index)
        observed_diff = (
            opt_caps_obs.loc[common_models_obs] - unopt_caps_obs.loc[common_models_obs]
        ).mean()

        print(f"    Observed difference: {observed_diff:.4f}")
        print(f"    Common models: {len(common_models_obs)}")

        # --- Null distribution (random partitions) ---
        print(f"  Computing null distribution ({num_permutations} permutations)...")

        # Get benchmarks that can be partitioned (excluding anchor)
        other_benchmarks = [b for b in all_benchmarks if b != anchor]

        null_diffs = []

        for perm_idx in range(num_permutations):
            # Show progress every permutation
            print(
                f"    Progress: {perm_idx + 1}/{num_permutations} permutations ({len(null_diffs)} successful so far)"
            )

            # Randomly partition benchmarks
            shuffled_benchmarks = rng.permutation(other_benchmarks)

            # Split into two partitions of similar size to observed
            split_point = len(shuffled_benchmarks) // 2
            partition1_benchmarks = shuffled_benchmarks[:split_point]
            partition2_benchmarks = shuffled_benchmarks[split_point:]

            # Create partitions including anchor
            partition1_df = working_df[
                working_df["benchmark"].isin(partition1_benchmarks)
            ]
            partition2_df = working_df[
                working_df["benchmark"].isin(partition2_benchmarks)
            ]

            # Add anchor to both partitions
            partition1_df = ensure_anchor_present(
                partition1_df, anchor, source_df=working_df
            )
            partition2_df = ensure_anchor_present(
                partition2_df, anchor, source_df=working_df
            )

            # Fit model on both partitions
            try:
                mcap_p1, _ = fit_subset_with_benchmark_anchor_ref_init(
                    partition1_df, anchor
                )
                mcap_p2, _ = fit_subset_with_benchmark_anchor_ref_init(
                    partition2_df, anchor
                )

                p1_caps = mcap_p1.set_index("model")["estimated_capability"]
                p2_caps = mcap_p2.set_index("model")["estimated_capability"]

                common_models_perm = p1_caps.index.intersection(p2_caps.index)

                if len(common_models_perm) > 0:
                    perm_diff = (
                        p1_caps.loc[common_models_perm]
                        - p2_caps.loc[common_models_perm]
                    ).mean()
                    null_diffs.append(perm_diff)
            except Exception:
                # Skip this permutation if fitting fails
                continue

        print(
            f"    Completed: {len(null_diffs)}/{num_permutations} permutations successful"
        )

        # Calculate p-value
        null_diffs_abs = np.abs(null_diffs)
        observed_diff_abs = np.abs(observed_diff)
        p_value = np.mean(null_diffs_abs >= observed_diff_abs)

        print(f"    Null distribution mean: {np.mean(null_diffs):.4f}")
        print(f"    Null distribution std: {np.std(null_diffs):.4f}")
        print(f"    P-value (two-tailed): {p_value:.4f}")

        # Store results
        permutation_results.append(
            {
                "anchor": anchor,
                "observed_diff": observed_diff,
                "null_mean": np.mean(null_diffs),
                "null_std": np.std(null_diffs),
                "p_value": p_value,
                "n_permutations": len(null_diffs),
                "n_common_models": len(common_models_obs),
                "null_diffs": null_diffs,
            }
        )

    # Save results
    results_df = pd.DataFrame(
        [
            {
                "anchor": r["anchor"],
                "observed_diff": r["observed_diff"],
                "null_mean": r["null_mean"],
                "null_std": r["null_std"],
                "p_value": r["p_value"],
                "n_permutations": r["n_permutations"],
                "n_common_models": r["n_common_models"],
            }
            for r in permutation_results
        ]
    )
    results_df.to_csv(output_dir / "permutation_test_results.csv", index=False)

    # Create visualization
    print(f"\nCreating permutation test visualization...")

    # Get custom colors from current style
    custom_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    n_anchors = len(permutation_results)
    fig, axes = plt.subplots(n_anchors, 1, figsize=(12, 4 * n_anchors))
    if n_anchors == 1:
        axes = [axes]

    for i, result in enumerate(permutation_results):
        ax = axes[i]

        # Plot null distribution
        ax.hist(
            result["null_diffs"],
            bins=30,
            alpha=0.7,
            color=custom_colors[0],
            edgecolor="black",
            label="Null distribution (random partitions)",
        )

        # Plot observed difference
        ax.axvline(
            result["observed_diff"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Observed (opt vs unopt): {result['observed_diff']:.3f}",
        )
        ax.axvline(0, color="black", linestyle="-", alpha=0.3)

        # Add p-value annotation
        ax.text(
            0.98,
            0.95,
            f"p = {result['p_value']:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            ha="right",
            va="top",
        )

        ax.set_xlabel("Mean Capability Difference")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Permutation Test: {result['anchor']}\n"
            f"({result['n_permutations']} permutations, {result['n_common_models']} models)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    perm_png, perm_pdf = save_plot(output_dir / "permutation_test_distribution")
    plt.show()

    print(f"\n{'='*80}")
    print("PERMUTATION TEST SUMMARY")
    print(f"{'='*80}")
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'permutation_test_results.csv'}")
    print(f"  - {perm_png} and {perm_pdf}")

    return permutation_results


def run_random_state_robustness_checks(
    num_anchors=10, num_runs=5, min_benchmark_date=None
):
    """
    Run robustness checks with different random seeds.

    Args:
        num_anchors: Number of anchors to use (default: 10)
        num_runs: Number of random runs to perform (default: 5)
        min_benchmark_date: Minimum benchmark release date as string (e.g., "2024-01-01")
                          or pd.Timestamp. If None, no date filtering is applied (default: None)
    """
    print(f"Running random state robustness checks with {num_anchors} anchors...")

    random_seeds = [42, 123, 456, 789, 999]
    results_summary = []
    anchors_total = None  # Will be set from first run

    for i, seed in enumerate(random_seeds[:num_runs]):
        print(f"\n{'='*60}")
        print(f"RANDOM STATE CHECK {i+1}/{num_runs}: seed={seed}")
        print(f"{'='*60}")

        # Run analysis with this random seed
        results, anchors = run_benchmark_anchor_approach(
            num_anchors, random_seed=seed, min_benchmark_date=min_benchmark_date
        )

        # Store anchors from first run
        if anchors_total is None:
            anchors_total = anchors

        # Store summary for comparison
        summary = {
            "run": i + 1,
            "seed": seed,
            "delta_mean": results["approach1_results"]["delta_mean"],
            "delta_std": results["approach1_results"]["delta_std"],
            "n_models": results["approach1_results"]["n_models_overlap"],
            "t_pvalue": results["summary1"]["t_pvalue"],
            "wilcoxon_pvalue": results["summary1"]["wilcoxon_pvalue"],
        }
        results_summary.append(summary)

    # Print comparison table
    print(f"\n{'='*80}")
    print("RANDOM STATE ROBUSTNESS CHECK SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'Run':<4} {'Seed':<6} {'Mean Δ':<12} {'Std Δ':<12} {'Models':<8} {'T-test p':<12} {'Wilcoxon p':<12}"
    )
    print("-" * 80)

    for summary in results_summary:
        print(
            f"{summary['run']:<4} "
            f"{summary['seed']:<6} "
            f"{summary['delta_mean']:<12.6f} "
            f"{summary['delta_std']:<12.6f} "
            f"{summary['n_models']:<8} "
            f"{summary['t_pvalue']:<12.2e} "
            f"{summary['wilcoxon_pvalue']:<12.2e}"
        )

    # Calculate statistics across runs
    delta_means = [s["delta_mean"] for s in results_summary]
    delta_stds = [s["delta_std"] for s in results_summary]

    print(f"\n{'='*50}")
    print("RANDOM STATE VARIABILITY ANALYSIS")
    print(f"{'='*50}")
    print(f"Mean Δ across runs: {np.mean(delta_means):.6f} ± {np.std(delta_means):.6f}")
    print(f"Std Δ across runs: {np.mean(delta_stds):.6f} ± {np.std(delta_stds):.6f}")
    print(f"Range of Mean Δ: [{np.min(delta_means):.6f}, {np.max(delta_means):.6f}]")
    print(f"Range of Std Δ: [{np.min(delta_stds):.6f}, {np.max(delta_stds):.6f}]")

    # Check if results are consistent
    mean_cv = (
        np.std(delta_means) / np.mean(delta_means) if np.mean(delta_means) != 0 else 0
    )
    print(f"Coefficient of variation (Mean Δ): {mean_cv:.4f}")

    if mean_cv < 0.05:
        print("✓ Results are highly consistent across random states")
    elif mean_cv < 0.10:
        print("✓ Results are reasonably consistent across random states")
    else:
        print("⚠ Results show some variability across random states")

    return results_summary, anchors_total


if __name__ == "__main__":
    # Set up custom styling
    setup_custom_style()

    # Set minimum benchmark date for date-filtered analysis
    # Set to None to disable date filtering (original behavior)
    # Set to "2024-01-01" to only consider benchmarks from 2024 onwards
    MIN_BENCHMARK_DATE = "2024-01-01"  # Change to None to disable date filtering

    if MIN_BENCHMARK_DATE is not None:
        print(f"\n{'='*80}")
        print(
            f"DATE FILTERING ENABLED: Only considering benchmarks from {MIN_BENCHMARK_DATE} onwards"
        )
        print(f"{'='*80}\n")
    else:
        print("\nDate filtering disabled - using all benchmarks\n")

    # Run random state robustness checks
    random_state_results, anchors_total = run_random_state_robustness_checks(
        num_anchors=9, num_runs=1, min_benchmark_date=MIN_BENCHMARK_DATE
    )
    print("\nRandom state robustness checks completed successfully!")

    # Run permutation test on all 20 anchors (top 10 optimized + top 10 unoptimized)
    print("\n" + "=" * 80)
    print("Running permutation test analysis...")
    print("=" * 80)

    print(
        f"Using all {len(anchors_total)} anchors from the optimization pressure analysis:"
    )
    print(f"  Anchors: {anchors_total}")

    permutation_results = run_permutation_test_analysis(
        anchor_benchmarks=anchors_total,
        num_permutations=100,
        random_seed=42,
        min_benchmark_date=MIN_BENCHMARK_DATE,
    )
    print("\nPermutation test analysis completed successfully!")
