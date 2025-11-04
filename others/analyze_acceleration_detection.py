#!/usr/bin/env python3
"""
Acceleration Detection Analysis - Synthetic Data Validation

This script tests the benchmark stitching methodology's ability to detect
acceleration in AI capability development using synthetic data experiments.
It validates whether the method can distinguish between linear and accelerating
capability growth patterns.

This analysis addresses a key methodological question: if AI capabilities
were to accelerate significantly, would our linear forecasting approach
detect this change, and how quickly?

Usage: python analyze_acceleration_detection.py [--num-trials N] [--output-dir DIR]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse

# Local imports
from analysis_utils import (
    setup_analysis_environment,
    setup_plotting_style,
    save_results_summary,
)


def generate_synthetic_data(
    num_models=600,
    num_benchmarks=30,
    speedup_factor_model=2,
    time_range_start=2020,
    time_range_end=2030,
    cutoff_year=2027,
    frac_eval=0.25,
    error_std=0.025,
    elo_change=3.5,
    base_model=0,
    noise_std_model=0.05,
    base_bench=0.5,
    saturation_level=0.05,
    min_alpha=3,
    max_alpha=10,
    frac_accelerate_models=1.0,
):
    """
    Generate synthetic benchmark stitching data with controllable acceleration

    Args:
        speedup_factor_model: Acceleration factor post-cutoff (e.g., 2 = 2x faster)
        cutoff_year: When acceleration begins
        frac_accelerate_models: Fraction of post-cutoff models that accelerate
        Other parameters control data size, noise, and benchmark characteristics
    """
    np.random.seed(42)

    # Generate model release times
    model_times = np.sort(
        np.random.uniform(time_range_start, time_range_end, num_models)
    )
    slope_model = elo_change / (time_range_end - time_range_start)

    # Decide which post-cutoff models accelerate
    random_draws = np.random.rand(num_models)
    accelerate_mask = (model_times >= cutoff_year) & (
        random_draws < frac_accelerate_models
    )

    # Compute model capabilities with acceleration
    model_capabilities = (
        base_model
        + np.where(
            accelerate_mask,
            # Accelerated: baseline until cutoff, then faster slope
            slope_model * (cutoff_year - time_range_start)
            + speedup_factor_model * slope_model * (model_times - cutoff_year),
            # Normal: same slope throughout
            slope_model * (model_times - time_range_start),
        )
        + np.random.normal(0, noise_std_model, num_models)
    )

    models = pd.DataFrame(
        {
            "model_id": np.arange(num_models),
            "date": model_times,
            "model_capabilities": model_capabilities,
            "accelerated": accelerate_mask,
        }
    )

    # Generate benchmarks
    benchmark_times = np.random.uniform(
        time_range_start, time_range_end, num_benchmarks
    )
    slope_bench = elo_change / (time_range_end - time_range_start)
    benchmark_difficulties = base_bench + np.where(
        benchmark_times < cutoff_year,
        slope_bench * (benchmark_times - time_range_start),
        slope_bench * (cutoff_year - time_range_start)
        + 1 * slope_bench * (benchmark_times - cutoff_year),
    )
    benchmark_progress_slopes = np.random.uniform(min_alpha, max_alpha, num_benchmarks)

    benchmarks = pd.DataFrame(
        {
            "benchmark_id": np.arange(num_benchmarks),
            "benchmark_release_date": benchmark_times,
            "benchmark_difficulties": benchmark_difficulties,
            "benchmark_progress_slopes": benchmark_progress_slopes,
        }
    )

    # Generate performance scores using sigmoid function
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    scores = []
    for _, m in models.iterrows():
        raw = logistic(
            benchmarks["benchmark_progress_slopes"]
            * (m["model_capabilities"] - benchmarks["benchmark_difficulties"])
        )
        mask = (raw >= saturation_level) & (raw <= 1 - saturation_level)
        for b_idx, s in zip(benchmarks["benchmark_id"][mask], raw[mask]):
            if np.random.rand() < frac_eval:
                noisy = np.clip(
                    s + np.random.normal(0, error_std), 0, 1 - saturation_level
                )
                scores.append(
                    {
                        "model_id": m["model_id"],
                        "benchmark_id": b_idx,
                        "date": m["date"],
                        "performance": noisy,
                    }
                )

    df_scores = pd.DataFrame(scores)
    return models, benchmarks, df_scores


def estimate_capabilities_from_scores(models, benchmarks, df_scores):
    """
    Estimate model capabilities from synthetic benchmark scores using the same
    statistical model as the main benchmark stitching analysis, with L2 regularization
    and benchmark anchoring
    """
    # Identify valid models (those with benchmark data)
    valid_model_ids = sorted(df_scores["model_id"].unique())
    model_id_to_fit_idx = {m_id: i for i, m_id in enumerate(valid_model_ids)}
    num_valid_models = len(valid_model_ids)
    num_benchmarks = benchmarks.shape[0]

    # Prepare arrays for fitting
    model_idx_for_data = np.array(
        [model_id_to_fit_idx[m] for m in df_scores["model_id"]]
    )
    benchmark_ids_for_data = df_scores["benchmark_id"].values.astype(int)
    observed_scores = df_scores["performance"].values

    # Anchor setup: Use first benchmark as anchor (similar to Winogrande in real data)
    anchor_benchmark_idx = 0
    anchor_difficulty = 0
    anchor_slope = 1.0
    regularization_strength = 0.1

    def logistic(x):
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def split_params(params):
        """Break parameter vector into C, D, and alpha with anchor slope fixed"""
        C = params[:num_valid_models]
        D = params[num_valid_models : num_valid_models + num_benchmarks]
        alpha_free = params[num_valid_models + num_benchmarks :]
        # Insert the fixed anchor slope
        alpha = np.insert(alpha_free, anchor_benchmark_idx, anchor_slope)
        return C, D, alpha

    def residuals(params, model_idx_for_data, benchmark_ids_for_data, observed_scores):
        C, D, alpha = split_params(params)

        c_vals = C[model_idx_for_data]
        d_vals = D[benchmark_ids_for_data]
        alpha_vals = alpha[benchmark_ids_for_data]

        preds = logistic(alpha_vals * (c_vals - d_vals))
        residuals = preds - observed_scores

        # Add L2 regularization
        if regularization_strength > 0:
            reg_term = regularization_strength * (
                np.sum(C**2) +
                np.sum(D**2) +
                np.sum(alpha[alpha != anchor_slope]**2)
            ) / (num_valid_models + num_benchmarks + num_benchmarks - 1)

            reg_penalty = np.sqrt(reg_term) if reg_term > 0 else 0
            return np.append(residuals, reg_penalty)

        return residuals

    # Initial parameter guesses (one fewer alpha since anchor is fixed)
    np.random.seed(42)
    initial_C = np.random.randn(num_valid_models) * 0.1
    initial_D = np.random.randn(num_benchmarks) * 0.1
    initial_alpha = np.ones(num_benchmarks - 1)  # One fewer because anchor is fixed
    initial_params = np.concatenate([initial_C, initial_D, initial_alpha])

    # Set bounds to prevent extreme values
    lower_bounds = np.concatenate([
        np.full(num_valid_models, -10),
        np.full(num_benchmarks, -10),
        np.full(num_benchmarks - 1, 0.1)
    ])
    upper_bounds = np.concatenate([
        np.full(num_valid_models, 10),
        np.full(num_benchmarks, 10),
        np.full(num_benchmarks - 1, 10)
    ])

    # Fit the model
    result = least_squares(
        residuals,
        initial_params,
        args=(model_idx_for_data, benchmark_ids_for_data, observed_scores),
        bounds=(lower_bounds, upper_bounds),
        method="trf",
    )

    # Recover full parameter vectors
    theta_hat = result.x
    C_hat = theta_hat[:num_valid_models]
    D_hat = theta_hat[num_valid_models : num_valid_models + num_benchmarks]
    alpha_free_hat = theta_hat[num_valid_models + num_benchmarks :]
    alpha_hat = np.insert(alpha_free_hat, anchor_benchmark_idx, anchor_slope)

    # Shift to match anchor difficulty
    shift = D_hat[anchor_benchmark_idx] - anchor_difficulty
    C_hat -= shift
    D_hat -= shift

    # Create DataFrame with estimated capabilities
    fitted_C_df = pd.DataFrame(
        {"model_id": valid_model_ids, "unaligned_C": C_hat}
    )

    # Merge with true capabilities and align
    meta = models[["model_id", "model_capabilities", "date"]]
    fitted_C_df = fitted_C_df.merge(meta, on="model_id", how="left")

    # Compute alignment transform to match true capabilities
    a, b = np.polyfit(
        fitted_C_df["unaligned_C"].values, fitted_C_df["model_capabilities"].values, 1
    )

    fitted_C_df["estimated_capability"] = a * fitted_C_df["unaligned_C"] + b

    return fitted_C_df


def piecewise_linear(x, slope1, intercept1, slope2, breakpoint):
    """Piecewise linear function with continuous breakpoint"""
    intercept2 = slope1 * breakpoint + intercept1 - slope2 * breakpoint
    return np.where(x < breakpoint, slope1 * x + intercept1, slope2 * x + intercept2)


def fit_piecewise_linear(x, y, num_breaks=30):
    """
    Fit piecewise linear model by scanning breakpoint candidates
    Returns best parameters and R² score
    """
    best_r2 = -np.inf
    best_params = None

    xs, ys = x, y
    rng = xs.max() - xs.min()
    min_bp = xs.min() + 0.1 * rng
    max_bp = xs.max() - 0.1 * rng
    lower = [-np.inf, -np.inf, -np.inf, min_bp]
    upper = [np.inf, np.inf, np.inf, max_bp]

    for bp in np.linspace(min_bp, max_bp, num_breaks):
        left = xs <= bp
        right = xs > bp
        if left.sum() < 2 or right.sum() < 2:
            continue

        # Initial parameter guesses
        m1, b1 = np.polyfit(xs[left], ys[left], 1)
        m2, _ = np.polyfit(xs[right], ys[right], 1)
        i1 = np.mean(ys[left]) - m1 * np.mean(xs[left])
        p0 = [m1, i1, m2, bp]

        try:
            params, _ = curve_fit(
                piecewise_linear, xs, ys, p0=p0, bounds=(lower, upper), maxfev=2000
            )
            r2 = r2_score(ys, piecewise_linear(xs, *params))
            if r2 > best_r2:
                best_r2 = r2
                best_params = params
        except Exception:
            continue

    return best_params, best_r2


def detect_acceleration_timeline(
    model_df, cutoff_year, num_post_windows=50, min_post=0.1, threshold=1.5
):
    """
    Test acceleration detection as a function of post-cutoff observation time

    Args:
        model_df: DataFrame with model capabilities and dates
        cutoff_year: Known acceleration start time
        num_post_windows: Number of post-cutoff windows to test
        threshold: Slope ratio threshold for detection

    Returns:
        results_df: Detection results for each time window
        detection: (time_to_detect, slope_ratio) or (None, None)
    """
    results = []
    model_df = model_df.copy()
    model_df["time_since_start"] = model_df["date"] - model_df["date"].min()
    end_year = model_df["date"].max()

    post_windows = np.linspace(min_post, end_year - cutoff_year, num_post_windows)

    for T in post_windows:
        # Use data up to cutoff + T years
        df = model_df[model_df["date"] <= cutoff_year + T]
        x = df["time_since_start"].values
        y = df["model_capabilities"].values

        params, r2 = fit_piecewise_linear(x, y)
        if params is not None:
            s1, _, s2, _ = params
            slope_ratio = s2 / s1 if s1 != 0 else np.inf
        else:
            slope_ratio = 1.0
            r2 = 0.0

        results.append(
            {
                "T_after": T,
                "slope_pre": s1 if params is not None else np.nan,
                "slope_post": s2 if params is not None else np.nan,
                "slope_ratio": slope_ratio,
                "r2": r2,
            }
        )

    res_df = pd.DataFrame(results)

    # Find first detection above threshold
    mask = res_df["slope_ratio"] >= threshold
    if mask.any():
        first = res_df[mask].iloc[0]
        detection = (first["T_after"], first["slope_ratio"])
    else:
        detection = (None, None)

    return res_df, detection


def broad_acceleration_experiment(output_dir):
    """
    Test detection when all post-cutoff models accelerate
    """
    print("Running broad acceleration experiment...")

    # Generate data with broad acceleration
    models, benchmarks, df_scores = generate_synthetic_data(
        num_models=600,
        num_benchmarks=30,
        speedup_factor_model=2,
        cutoff_year=2027,
        frac_accelerate_models=1.0,  # All models accelerate
    )

    # Estimate capabilities from synthetic scores
    df_estimated = estimate_capabilities_from_scores(models, benchmarks, df_scores)

    # Test acceleration detection
    results_df, detection = detect_acceleration_timeline(
        df_estimated, cutoff_year=2027, threshold=1.5
    )

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Synthetic data
    ax1.scatter(
        models["date"],
        models["model_capabilities"],
        alpha=0.7,
        label="True capabilities",
        s=30,
    )
    ax1.scatter(
        df_estimated["date"],
        df_estimated["estimated_capability"],
        alpha=0.7,
        label="Estimated capabilities",
        s=30,
    )
    ax1.axvline(
        2027, color="red", linestyle="--", alpha=0.7, label="Acceleration start"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Capability")
    ax1.set_title("Broad Acceleration: True vs Estimated Capabilities")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Detection timeline
    ax2.plot(results_df["T_after"], results_df["slope_ratio"], marker="o", markersize=4)
    ax2.axhline(
        1.5, color="red", linestyle="--", alpha=0.7, label="Detection threshold"
    )
    ax2.set_xlabel("Years after acceleration start")
    ax2.set_ylabel("Slope ratio (post/pre)")
    ax2.set_title("Acceleration Detection Timeline")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        output_dir / "broad_acceleration_analysis.png", dpi=300, bbox_inches="tight"
    )

    return results_df, detection, df_estimated


def narrow_acceleration_experiment(output_dir):
    """
    Test frontier-based detection when only some models accelerate
    """
    print("Running narrow acceleration experiment...")

    # Generate data with narrow acceleration (only 10% of models)
    models, benchmarks, df_scores = generate_synthetic_data(
        num_models=600,
        num_benchmarks=30,
        speedup_factor_model=2,
        cutoff_year=2027,
        frac_accelerate_models=0.1,  # Only 10% accelerate
    )

    # Sort models by date and compute running maximum (frontier)
    df = models.sort_values("date").copy()
    df["running_max"] = df["model_capabilities"].cummax()
    frontier_df = df[df["model_capabilities"] == df["running_max"]]

    # Fit piecewise linear to frontier
    x = frontier_df["date"].values
    y = frontier_df["model_capabilities"].values
    params, r2 = fit_piecewise_linear(x, y, num_breaks=50)

    if params is not None:
        slope1, intercept1, slope2, bp = params

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all models (faded)
        ax.scatter(
            df["date"], df["model_capabilities"], alpha=0.2, label="All models", s=20
        )

        # Plot frontier models
        ax.scatter(
            frontier_df["date"],
            frontier_df["model_capabilities"],
            color="orange",
            label="Frontier models",
            s=40,
        )

        # Plot piecewise fit
        x_fine = np.linspace(x.min(), x.max(), 400)
        y_fine = piecewise_linear(x_fine, *params)
        ax.plot(x_fine, y_fine, "k--", lw=2, label="Piecewise fit")
        ax.axvline(bp, color="gray", linestyle=":", label=f"Detected break @ {bp:.2f}")
        ax.axvline(
            2027,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="True acceleration start",
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Model capability")
        ax.set_title("Narrow Acceleration: Frontier Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "narrow_acceleration_frontier.png",
            dpi=300,
            bbox_inches="tight",
        )

        return {
            "detected_breakpoint": bp,
            "true_breakpoint": 2027,
            "pre_slope": slope1,
            "post_slope": slope2,
            "slope_ratio": slope2 / slope1,
            "r2": r2,
        }
    else:
        return None


def parameter_sweep_experiment(output_dir):
    """
    Systematic parameter sweep to understand detection sensitivity
    """
    print("Running parameter sweep experiment...")

    time_range_start, time_range_end = 2020, 2030
    cutoff_year = 2027

    # Parameter grids
    num_models_list = np.array([40, 60, 80, 100, 120]) * (
        time_range_end - time_range_start
    )
    num_benchmarks_list = np.array([2, 4, 6, 8, 10]) * (
        time_range_end - time_range_start
    )
    acceleration_list = [2.5, 5.0]

    records = []
    total_experiments = (
        len(num_models_list) * len(num_benchmarks_list) * len(acceleration_list)
    )
    experiment_count = 0

    for nm in num_models_list:
        for nb in num_benchmarks_list:
            for accel in acceleration_list:
                experiment_count += 1
                print(
                    f"  Experiment {experiment_count}/{total_experiments}: "
                    f"{int(nm/(time_range_end-time_range_start))} models/yr, "
                    f"{int(nb/(time_range_end-time_range_start))} benchmarks/yr, "
                    f"{accel}x acceleration"
                )

                # Generate synthetic data
                models, _, _ = generate_synthetic_data(
                    num_models=int(nm),
                    num_benchmarks=int(nb),
                    speedup_factor_model=accel,
                    time_range_start=time_range_start,
                    time_range_end=time_range_end,
                    cutoff_year=cutoff_year,
                    frac_accelerate_models=1.0,
                )

                # Test detection
                results_df, (T_det, ratio_det) = detect_acceleration_timeline(
                    models, cutoff_year, num_post_windows=50, threshold=1.5
                )

                records.append(
                    {
                        "models_per_year": int(
                            nm / (time_range_end - time_range_start)
                        ),
                        "benchmarks_per_year": int(
                            nb / (time_range_end - time_range_start)
                        ),
                        "actual_acceleration": accel,
                        "detect_time": T_det,
                        "est_acc_at_detect": ratio_det,
                    }
                )

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(output_dir / "parameter_sweep_results.csv", index=False)

    # Create summary visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, acc in enumerate(acceleration_list):
        # Detection time vs models per year
        ax = axes[i, 0]
        for bt in [2, 4, 6, 8, 10]:
            df = summary_df[
                (summary_df["benchmarks_per_year"] == bt)
                & (summary_df["actual_acceleration"] == acc)
            ]
            ax.plot(
                df["models_per_year"],
                df["detect_time"],
                marker="o",
                label=f"{bt} benchmarks/yr",
            )
        ax.set_xlabel("Models per year")
        ax.set_ylabel("Detection time (years)")
        ax.set_title(f"Detection Time vs Data Density ({acc}x acceleration)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Detection time vs benchmarks per year
        ax = axes[i, 1]
        for mt in [40, 60, 80, 100, 120]:
            df = summary_df[
                (summary_df["models_per_year"] == mt)
                & (summary_df["actual_acceleration"] == acc)
            ]
            ax.plot(
                df["benchmarks_per_year"],
                df["detect_time"],
                marker="o",
                label=f"{mt} models/yr",
            )
        ax.set_xlabel("Benchmarks per year")
        ax.set_ylabel("Detection time (years)")
        ax.set_title(f"Detection Time vs Benchmark Density ({acc}x acceleration)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "parameter_sweep_analysis.png", dpi=300, bbox_inches="tight"
    )

    return summary_df


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(
        description="Test acceleration detection capabilities"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trial runs for each experiment (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/acceleration_detection",
        help="Output directory for results and plots",
    )

    args = parser.parse_args()

    print("Starting acceleration detection analysis...")

    # Setup environment
    setup_analysis_environment()
    setup_plotting_style()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    print("\n1. Broad Acceleration Experiment")
    broad_results, broad_detection, broad_estimated = broad_acceleration_experiment(
        output_dir
    )

    print("\n2. Narrow Acceleration (Frontier) Experiment")
    narrow_results = narrow_acceleration_experiment(output_dir)

    print("\n3. Parameter Sweep Experiment")
    sweep_results = parameter_sweep_experiment(output_dir)

    # Compile summary results
    all_results = {
        "Broad Acceleration": {
            "detection_time": (
                broad_detection[0] if broad_detection[0] is not None else "Not detected"
            ),
            "slope_ratio_at_detection": (
                broad_detection[1] if broad_detection[1] is not None else "N/A"
            ),
            "models_analyzed": len(broad_estimated),
        },
        "Narrow Acceleration (Frontier)": (
            narrow_results if narrow_results else "Failed to fit"
        ),
        "Parameter Sweep": {
            "total_experiments": len(sweep_results),
            "mean_detection_time_2.5x": sweep_results[
                sweep_results["actual_acceleration"] == 2.5
            ]["detect_time"].mean(),
            "mean_detection_time_5x": sweep_results[
                sweep_results["actual_acceleration"] == 5.0
            ]["detect_time"].mean(),
        },
    }

    save_results_summary(all_results, output_dir / "acceleration_detection_summary.txt")

    # Print summary
    print("\n" + "=" * 60)
    print("ACCELERATION DETECTION ANALYSIS SUMMARY")
    print("=" * 60)

    if broad_detection[0] is not None:
        print(
            f"Broad acceleration (all models): Detected after {broad_detection[0]:.2f} years"
        )
        print(f"  Slope ratio at detection: {broad_detection[1]:.2f}")
    else:
        print("Broad acceleration: Not detected within observation window")

    if narrow_results:
        print(
            f"\nNarrow acceleration (frontier): Detected breakpoint at {narrow_results['detected_breakpoint']:.2f}"
        )
        print(f"  True breakpoint: {narrow_results['true_breakpoint']}")
        print(f"  Slope ratio: {narrow_results['slope_ratio']:.2f}")
    else:
        print("\nNarrow acceleration: Detection failed")

    print(f"\nParameter sweep: {len(sweep_results)} experiments completed")
    print(
        f"  Mean detection time (2.5x acceleration): {sweep_results[sweep_results['actual_acceleration'] == 2.5]['detect_time'].mean():.2f} years"
    )
    print(
        f"  Mean detection time (5.0x acceleration): {sweep_results[sweep_results['actual_acceleration'] == 5.0]['detect_time'].mean():.2f} years"
    )

    print(f"\nResults saved to: {output_dir}")

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
