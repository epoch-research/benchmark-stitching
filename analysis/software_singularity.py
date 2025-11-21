"""
Detection Analysis: Parameter Sweep and False Positive Testing

This script runs two main analyses:
1. Parameter sweep across model/benchmark release rates and acceleration factors
2. False positive rate testing to validate detection specificity

This is a focused version that contains only the parameter sweep and false positive
testing components from the full software singularity analysis.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default output directory
OUTPUT_DIR = Path("outputs/software_singularity")

# Global storage for run data (used for visualizations)
RUN_DATA_STORAGE = {}

# ============================================================================
# CUSTOM GRAPH STYLING
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

# ============================================================================
# DATA GENERATION
# ============================================================================


def generate_data(
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
    noise_std_bench=0.05,
    base_bench=0.5,
    saturation_level=0.05,
    min_alpha=3,
    max_alpha=10,
    frac_accelerate_models=1.0,
    random_seed=None,
):
    """
    Generate synthetic benchmark evaluation data with optional acceleration.

    Models released after cutoff_year can have accelerated capability growth.
    Benchmarks track the frontier, with difficulty scaling over time.

    Parameters:
    -----------
    speedup_factor_model : float
        Multiplier for capability growth rate after cutoff (e.g., 2 = 2x faster)
    cutoff_year : float
        Year when acceleration begins
    frac_accelerate_models : float
        Fraction of post-cutoff models that accelerate (0-1)
    noise_std_model : float
        Noise added to model capabilities
    noise_std_bench : float
        Noise added to benchmark difficulties
    random_seed : int or None
        Random seed for reproducibility. If None, uses current numpy random state.

    Returns:
    --------
    models : DataFrame with columns [model_id, date, model_capabilities, accelerated]
    benchmarks : DataFrame with benchmark metadata
    scores : DataFrame with evaluation results
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate models
    model_times = np.sort(
        np.random.uniform(time_range_start, time_range_end, num_models)
    )
    slope_model = elo_change / (time_range_end - time_range_start)

    # Decide which post-cutoff models accelerate
    random_draws = np.random.rand(num_models)
    accelerate_mask = (model_times >= cutoff_year) & (
        random_draws < frac_accelerate_models
    )

    # Compute capabilities with acceleration
    model_capabilities = (
        base_model
        + np.where(
            accelerate_mask,
            # Accelerated: baseline until cutoff, then faster slope
            slope_model * (cutoff_year - time_range_start)
            + speedup_factor_model * slope_model * (model_times - cutoff_year),
            # Normal: same slope the whole time
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
    benchmark_difficulties = (
        base_bench
        + np.where(
            benchmark_times < cutoff_year,
            slope_bench * (benchmark_times - time_range_start),
            slope_bench * (cutoff_year - time_range_start)
            + 1 * slope_bench * (benchmark_times - cutoff_year),
        )
        + np.random.normal(0, noise_std_bench, num_benchmarks)
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

    # Generate evaluation scores using logistic function
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


# ============================================================================
# CAPABILITY ESTIMATION
# ============================================================================


def estimated_capabilities(
    models, benchmarks, df, regularization_strength=0.1, verbose=False
):
    """
    Estimate model capabilities from benchmark scores using logistic regression.

    Fits the model: performance(m,b) = σ(α_b(C_m - D_b))
    where C_m = capability, D_b = difficulty, α_b = slope, σ = sigmoid

    Parameters:
    -----------
    verbose : bool
        If True, print warnings about skipped models. Default False to reduce
        noise during sweeps where this is expected behavior.

    Returns:
    --------
    DataFrame with columns: model_id, date, model_capabilities,
                           estimated_capability, unaligned_C
    """
    # Identify valid models (those with data)
    valid_model_ids = sorted(df["model_id"].unique())
    skipped_model_ids = set(models["model_id"]) - set(valid_model_ids)
    if skipped_model_ids and verbose:
        print(f"  Skipping {len(skipped_model_ids)} models (no data)")

    # Create mapping for valid models
    model_id_to_fit_idx = {m_id: i for i, m_id in enumerate(valid_model_ids)}
    num_valid_models = len(valid_model_ids)
    num_benchmarks = benchmarks.shape[0]

    # Prepare arrays for fitting
    model_idx_for_data = np.array([model_id_to_fit_idx[m] for m in df["model_id"]])
    benchmark_ids_for_data = df["benchmark_id"].values.astype(int)
    observed_scores = df["performance"].values

    # Define logistic and residual functions
    def logistic(x):
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def residuals(params, model_idx, bench_idx, y):
        C = params[:num_valid_models]
        D = params[num_valid_models : num_valid_models + num_benchmarks]
        alpha = params[num_valid_models + num_benchmarks :]

        c_vals = C[model_idx]
        d_vals = D[bench_idx]
        alpha_vals = alpha[bench_idx]

        preds = logistic(alpha_vals * (c_vals - d_vals))
        residuals = preds - y

        # Add L2 regularization
        if regularization_strength > 0:
            reg_term = (
                regularization_strength
                * (np.sum(C**2) + np.sum(D**2) + np.sum(alpha**2))
                / (num_valid_models + num_benchmarks + num_benchmarks)
            )
            reg_penalty = np.sqrt(reg_term) if reg_term > 0 else 0
            return np.append(residuals, reg_penalty)

        return residuals

    # Set initial guesses
    rng = np.random.default_rng(42)
    initial_C = rng.normal(0.0, 0.1, size=num_valid_models)
    initial_D = rng.normal(0.0, 0.1, size=num_benchmarks)
    initial_alpha = np.full(num_benchmarks, 1.0)
    initial_params = np.concatenate([initial_C, initial_D, initial_alpha])

    # Set bounds
    lower_bounds = np.concatenate(
        [
            np.full(num_valid_models, -10),
            np.full(num_benchmarks, -10),
            np.full(num_benchmarks, 0.1),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.full(num_valid_models, 10),
            np.full(num_benchmarks, 10),
            np.full(num_benchmarks, 10),
        ]
    )

    # Fit with bounds
    result = least_squares(
        residuals,
        initial_params,
        args=(model_idx_for_data, benchmark_ids_for_data, observed_scores),
        bounds=(lower_bounds, upper_bounds),
        method="trf",
    )

    estimated_params = result.x
    estimated_C = estimated_params[:num_valid_models]

    # Map fitted capabilities back to original model IDs
    fitted_C_df = pd.DataFrame(
        {"model_id": valid_model_ids, "unaligned_C": estimated_C}
    )

    meta = models[["model_id", "model_capabilities", "date"]]
    fitted_C_df = fitted_C_df.merge(meta, on="model_id", how="left")

    # Compute alignment transform
    a, b = np.polyfit(
        fitted_C_df["unaligned_C"].values, fitted_C_df["model_capabilities"].values, 1
    )

    fitted_C_df["estimated_capability"] = a * fitted_C_df["unaligned_C"] + b

    return fitted_C_df


# ============================================================================
# PIECEWISE LINEAR FITTING
# ============================================================================


def piecewise_linear(x, slope1, intercept1, slope2, breakpoint):
    """
    Piecewise linear function with two segments that meet at breakpoint.
    """
    intercept2 = slope1 * breakpoint + intercept1 - slope2 * breakpoint
    return np.where(x < breakpoint, slope1 * x + intercept1, slope2 * x + intercept2)


def fit_two_segments_fixed_breakpoint(x, y, breakpoint):
    """
    Fit two line segments with a FIXED breakpoint (no optimization of bp).
    Forces continuity at the breakpoint.

    Returns:
    --------
    dict with slope1, intercept1, slope2, intercept2, breakpoint, r2
    or None if insufficient data
    """
    left_mask = x <= breakpoint
    right_mask = x > breakpoint

    # Need at least 2 points in each segment
    if left_mask.sum() < 2 or right_mask.sum() < 2:
        return None

    # Fit left segment
    x_left = x[left_mask]
    y_left = y[left_mask]
    slope1, intercept1 = np.polyfit(x_left, y_left, 1)

    # Fit right segment
    x_right = x[right_mask]
    y_right = y[right_mask]
    slope2, _ = np.polyfit(x_right, y_right, 1)

    # Force continuity
    intercept2 = slope1 * breakpoint + intercept1 - slope2 * breakpoint

    # Compute R²
    y_pred = np.where(x <= breakpoint, slope1 * x + intercept1, slope2 * x + intercept2)
    r2 = r2_score(y, y_pred)

    return {
        "slope1": slope1,
        "intercept1": intercept1,
        "slope2": slope2,
        "intercept2": intercept2,
        "breakpoint": breakpoint,
        "r2": r2,
    }


# ============================================================================
# ACCELERATION DETECTION
# ============================================================================


def detect_acceleration_sequential(
    x,
    y,
    cutoff_year,
    min_acceleration=2.0,
    min_r2=0.6,
    min_gap_years=0.0,
    scan_resolution=50,
    min_points_after=3,
    verbose=False,
):
    """
    Sequentially scan for the FIRST point where we can detect acceleration.

    Scans from cutoff_year onwards and returns the first breakpoint where:
    1. The piecewise fit has R² >= min_r2
    2. The slope ratio (slope2/slope1) >= min_acceleration
    3. The breakpoint is >= cutoff_year + min_gap_years
    4. At least min_points_after data points exist after the breakpoint

    This simulates the temporal process of accumulating evidence over time.

    Returns:
    --------
    dict with fit results, or None if no detection
    """
    # Determine scan range
    min_bp = cutoff_year + min_gap_years
    max_bp = x.max()

    if min_bp >= max_bp:
        if verbose:
            print(f"  Cannot scan: min_bp={min_bp:.2f} >= max_bp={max_bp:.2f}")
        return None

    # Generate candidate breakpoints
    breakpoints = np.linspace(min_bp, max_bp, scan_resolution)

    if verbose:
        print(
            f"  Scanning {len(breakpoints)} breakpoints from {min_bp:.2f} to {max_bp:.2f}"
        )

    # Scan sequentially
    for i, bp in enumerate(breakpoints):
        # Check minimum points requirement
        points_after = (x > bp).sum()
        if points_after < min_points_after:
            continue

        result = fit_two_segments_fixed_breakpoint(x, y, bp)

        if result is None:
            continue

        # Check if slope1 is positive (avoid division issues)
        if result["slope1"] <= 0:
            continue

        # Compute slope ratio
        ratio = result["slope2"] / result["slope1"]

        # Check detection criteria
        if result["r2"] >= min_r2 and ratio >= min_acceleration:
            if verbose:
                print(f"  ✓ DETECTED at breakpoint {bp:.3f}")
                print(f"    Slope before: {result['slope1']:.4f}")
                print(f"    Slope after: {result['slope2']:.4f}")
                print(f"    Ratio: {ratio:.2f}x")
                print(f"    R²: {result['r2']:.4f}")

            result["ratio"] = ratio
            result["detected"] = True
            result["points_after"] = points_after
            return result

    if verbose:
        print(f"  ✗ No detection in {len(breakpoints)} candidates")

    return None


# ============================================================================
# TEMPORAL DETECTION (CORE ANALYSIS)
# ============================================================================


def estimate_detection_for_single_trajectory(
    models_df,
    benchmarks_df,
    scores_df,
    cutoff_year,
    acceleration_factor=2.0,
    min_r2=0.6,
    min_gap_years=0.0,
    scan_resolution=50,
    min_points_after=3,
    verbose=True,
):
    """
    Simulates temporal detection: incrementally reveal data over time and detect
    when we first have enough evidence of acceleration.

    TEMPORAL PROCESS:
    1. Acceleration happens at cutoff_year
    2. Time passes, more models/benchmarks are released
    3. At each time point, we ask: "With data available UP TO NOW, can we detect?"
    4. Return the FIRST time when detection succeeds

    This is the KEY difference from fitting all data at once - we simulate
    the actual process of waiting for evidence to accumulate.

    Returns:
    --------
    dict with keys: detected, years_to_detect, detection_time, breakpoint,
                    slope_before, slope_after, ratio, r2
    """
    # Get models released after cutoff
    post_cutoff_models = models_df[models_df["date"] > cutoff_year].copy()

    if post_cutoff_models.empty:
        if verbose:
            print("  No models released after cutoff year")
        return {
            "detected": False,
            "years_to_detect": None,
            "detection_time": None,
            "breakpoint": None,
            "slope_before": None,
            "slope_after": None,
            "ratio": None,
            "r2": None,
        }

    # Create time checkpoints to test
    min_time = cutoff_year + min_gap_years
    max_time = models_df["date"].max()

    # Use model release times as natural checkpoints
    model_times = sorted(post_cutoff_models["date"].unique())
    checkpoint_times = [t for t in model_times if t >= min_time]

    if not checkpoint_times:
        if verbose:
            print(f"  No data after cutoff + gap ({min_time:.2f})")
        return {
            "detected": False,
            "years_to_detect": None,
            "detection_time": None,
            "breakpoint": None,
            "slope_before": None,
            "slope_after": None,
            "ratio": None,
            "r2": None,
        }

    if verbose:
        print(f"\n  Temporal Detection Simulation")
        print(f"  Cutoff year: {cutoff_year:.2f}")
        print(f"  Testing {len(checkpoint_times)} time checkpoints")
        print(f"  Looking for {acceleration_factor}x acceleration\n")

    # Incrementally reveal data and check for detection at each time point
    for current_time in checkpoint_times:
        # Data available up to current_time
        scores_up_to_now = scores_df[scores_df["date"] <= current_time].copy()

        if scores_up_to_now.empty or len(scores_up_to_now) < 10:
            continue

        # Try to estimate capabilities with data available up to now
        try:
            df_est = estimated_capabilities(models_df, benchmarks_df, scores_up_to_now)
        except:
            continue

        if df_est.empty or len(df_est) < 5:
            continue

        # Compute frontier (running maximum reduces noise)
        df_est = df_est.sort_values("date").copy()
        df_est["frontier"] = df_est["estimated_capability"].cummax()

        x = df_est["date"].values
        y = df_est["frontier"].values

        # Try to detect acceleration with data available up to now
        result = detect_acceleration_sequential(
            x,
            y,
            cutoff_year=cutoff_year,
            min_acceleration=acceleration_factor,
            min_r2=min_r2,
            min_gap_years=min_gap_years,
            scan_resolution=scan_resolution,
            min_points_after=min_points_after,
            verbose=False,
        )

        if result is not None:
            # DETECTION SUCCESSFUL!
            if verbose:
                print(f"  ✓ DETECTED at time {current_time:.3f}")
                print(f"    Years after cutoff: {current_time - cutoff_year:.2f}")
                print(f"    Breakpoint: {result['breakpoint']:.3f}")
                print(f"    Ratio: {result['ratio']:.2f}x")
                print(f"    R²: {result['r2']:.4f}")

            return {
                "detected": True,
                "years_to_detect": current_time - cutoff_year,
                "detection_time": current_time,
                "breakpoint": result["breakpoint"],
                "slope_before": result["slope1"],
                "slope_after": result["slope2"],
                "ratio": result["ratio"],
                "r2": result["r2"],
            }

    # No detection across all time points
    if verbose:
        print(f"  ✗ No detection across {len(checkpoint_times)} time points")

    return {
        "detected": False,
        "years_to_detect": None,
        "detection_time": None,
        "breakpoint": None,
        "slope_before": None,
        "slope_after": None,
        "ratio": None,
        "r2": None,
    }


# ============================================================================
# PARAMETER SWEEP
# ============================================================================


def run_detection_sweep(
    models_per_year_list,
    benchmarks_per_year_list,
    acceleration_factors,
    noise_multipliers,
    n_simulations=5,
    time_range_start=2024,
    horizon_years=6,
    cutoff_year=2027,
    detection_threshold=2.0,
    random_seed_base=42,
    store_runs=True,
):
    """
    Run detection analysis across a grid of parameters.

    Tests different combinations of:
    - Model release rates (models per year)
    - Benchmark release rates (benchmarks per year)
    - True acceleration factors (2x, 3x, etc.)
    - Noise multipliers (1x, 2x, etc.)

    For each combination, runs multiple simulations and computes:
    - Detection success rate
    - Average time to detection

    Returns:
    --------
    DataFrame with columns: models_per_year, benchmarks_per_year, accel_factor,
                           noise_multiplier, detected_fraction, mean_years_to_detect, ...
    """
    global RUN_DATA_STORAGE
    if store_runs:
        RUN_DATA_STORAGE = {}

    results = []
    total_runs = (
        len(models_per_year_list)
        * len(benchmarks_per_year_list)
        * len(acceleration_factors)
        * len(noise_multipliers)
        * n_simulations
    )
    run_idx = 0

    print(f"\nRunning detection sweep: {total_runs} total simulations")
    print(f"{'='*60}\n")

    for models_per_year in models_per_year_list:
        for benchmarks_per_year in benchmarks_per_year_list:
            for accel_factor in acceleration_factors:
                for noise_mult in noise_multipliers:

                    detection_times = []
                    detections = []

                    for sim_idx in range(n_simulations):
                        run_idx += 1

                        # Set random seed for reproducibility
                        seed = random_seed_base + run_idx

                        # Generate synthetic data
                        num_models = int(models_per_year * horizon_years)
                        num_benchmarks = int(benchmarks_per_year * horizon_years)

                        # Apply noise multiplier to base noise parameters
                        base_error_std = 0.025
                        base_noise_std_model = 0.2
                        base_noise_std_bench = 0.2

                        models_df, benchmarks_df, scores_df = generate_data(
                            num_models=num_models,
                            num_benchmarks=num_benchmarks,
                            speedup_factor_model=accel_factor,
                            time_range_start=time_range_start,
                            time_range_end=time_range_start + horizon_years,
                            cutoff_year=cutoff_year,
                            frac_eval=0.25,
                            error_std=base_error_std * noise_mult,
                            elo_change=3.5,
                            noise_std_model=base_noise_std_model * noise_mult,
                            noise_std_bench=base_noise_std_bench * noise_mult,
                            frac_accelerate_models=1.0,
                            random_seed=seed,
                        )

                        # Estimate capabilities
                        df_est = estimated_capabilities(
                            models_df, benchmarks_df, scores_df
                        )

                        # Estimate detection
                        result = estimate_detection_for_single_trajectory(
                            models_df,
                            benchmarks_df,
                            scores_df,
                            cutoff_year=cutoff_year,
                            acceleration_factor=detection_threshold,
                            verbose=False,
                        )

                        # Store run data if requested
                        if store_runs:
                            RUN_DATA_STORAGE[run_idx] = {
                                "models_df": models_df.copy(),
                                "benchmarks_df": benchmarks_df.copy(),
                                "scores_df": scores_df.copy(),
                                "df_est": df_est.copy(),
                                "result": result.copy(),
                                "params": {
                                    "models_per_year": models_per_year,
                                    "benchmarks_per_year": benchmarks_per_year,
                                    "accel_factor": accel_factor,
                                    "noise_multiplier": noise_mult,
                                    "sim_idx": sim_idx,
                                    "seed": seed,
                                    "cutoff_year": cutoff_year,
                                },
                            }

                        detections.append(result["detected"])
                        if result["detected"]:
                            detection_times.append(result["years_to_detect"])

                        if run_idx % 10 == 0:
                            print(f"Progress: {run_idx}/{total_runs} runs complete")

                    # Aggregate results for this parameter combination
                    detected_fraction = np.mean(detections)
                    mean_detection_time = (
                        np.mean(detection_times) if detection_times else None
                    )

                    results.append(
                        {
                            "models_per_year": models_per_year,
                            "benchmarks_per_year": benchmarks_per_year,
                            "accel_factor": accel_factor,
                            "noise_multiplier": noise_mult,
                            "detected_fraction": detected_fraction,
                            "mean_years_to_detect": mean_detection_time,
                            "n_detected": sum(detections),
                            "n_total": n_simulations,
                        }
                    )

    print(f"\nSweep complete!")
    if store_runs:
        print(f"Stored {len(RUN_DATA_STORAGE)} runs for visualization\n")

    return pd.DataFrame(results)


# ============================================================================
# FALSE POSITIVE TESTING
# ============================================================================


def test_false_positive_rate_single(
    n_trials=100,
    num_models=100,
    num_benchmarks=20,
    cutoff_year=2027,
    observation_years=3.0,
    acceleration_factor=2.0,
    min_r2=0.6,
    noise_multiplier=1.0,
    **kwargs,
):
    """
    Test false positive rate for a single configuration.

    This is a helper function used by test_false_positive_rate_sweep.
    """
    # Base noise parameters (matching parameter sweep)
    base_error_std = 0.025
    base_noise_std_model = 0.2
    base_noise_std_bench = 0.2

    # Apply noise multiplier
    error_std = base_error_std * noise_multiplier
    noise_std_model = base_noise_std_model * noise_multiplier
    noise_std_bench = base_noise_std_bench * noise_multiplier

    false_positives = 0
    detection_times = []

    # Calculate time range based on observation window
    time_range_start = cutoff_year - 3.0  # 3 years before cutoff for baseline
    time_range_end = cutoff_year + observation_years

    for trial in range(n_trials):
        # Generate data with NO acceleration (speedup_factor=1)
        models_df, benchmarks_df, scores_df = generate_data(
            num_models=num_models,
            num_benchmarks=num_benchmarks,
            speedup_factor_model=1.0,  # NO ACCELERATION
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            cutoff_year=cutoff_year,
            frac_eval=kwargs.get("frac_eval", 0.25),
            error_std=error_std,
            noise_std_model=noise_std_model,
            noise_std_bench=noise_std_bench,
        )

        # Try to estimate capabilities
        try:
            df_est = estimated_capabilities(models_df, benchmarks_df, scores_df)
            if df_est.empty or len(df_est) < 5:
                continue

            df_est = df_est.sort_values("date").copy()
            df_est["frontier"] = df_est["estimated_capability"].cummax()

            x = df_est["date"].values
            y = df_est["frontier"].values

            # Try to detect
            result = detect_acceleration_sequential(
                x,
                y,
                cutoff_year=cutoff_year,
                min_acceleration=acceleration_factor,
                min_r2=min_r2,
                scan_resolution=50,
                verbose=False,
            )

            if result is not None:
                false_positives += 1
                detection_times.append(result["breakpoint"])
        except:
            continue

    fpr = false_positives / n_trials

    return {
        "false_positive_rate": fpr,
        "n_false_positives": false_positives,
        "n_trials": n_trials,
        "detection_times": detection_times,
    }


def test_false_positive_rate_sweep(
    noise_multipliers=[1.0, 2.0],
    observation_years_list=[0.5, 1.0, 2.0, 3.0, 6.0],
    n_trials=100,
    num_models=100,
    num_benchmarks=20,
    cutoff_year=2027,
    acceleration_factor=2.0,
    min_r2=0.6,
    **kwargs,
):
    """
    Test false positive rate across a grid of noise levels and observation windows.

    Sweeps over different combinations of noise multipliers and observation periods
    to understand how these parameters affect false positive rates.

    Parameters:
    -----------
    noise_multipliers : list of float
        Noise multipliers to test (same as parameter sweep: [1.0, 2.0])
    observation_years_list : list of float
        Observation windows to test (e.g., [0.5, 1.0, 2.0, 3.0, 6.0] years)

    Returns:
    --------
    DataFrame with columns: noise_multiplier, observation_years, false_positive_rate,
                           n_false_positives, n_trials
    """
    results = []
    total_configs = len(noise_multipliers) * len(observation_years_list)
    config_idx = 0

    print(f"Testing false positive rate across {total_configs} configurations...")
    print(f"  Noise multipliers: {noise_multipliers}")
    print(f"  Observation windows: {observation_years_list} years")
    print(f"  {n_trials} trials per configuration\n")

    for noise_mult in noise_multipliers:
        for obs_years in observation_years_list:
            config_idx += 1
            time_range_end = cutoff_year + obs_years

            print(
                f"[{config_idx}/{total_configs}] noise={noise_mult:.1f}x, obs={obs_years:.1f}yr ({cutoff_year:.1f} to {time_range_end:.1f})..."
            )

            result = test_false_positive_rate_single(
                n_trials=n_trials,
                num_models=num_models,
                num_benchmarks=num_benchmarks,
                cutoff_year=cutoff_year,
                observation_years=obs_years,
                acceleration_factor=acceleration_factor,
                min_r2=min_r2,
                noise_multiplier=noise_mult,
                **kwargs,
            )

            results.append(
                {
                    "noise_multiplier": noise_mult,
                    "observation_years": obs_years,
                    "false_positive_rate": result["false_positive_rate"],
                    "n_false_positives": result["n_false_positives"],
                    "n_trials": result["n_trials"],
                }
            )

            print(
                f"  FPR: {result['false_positive_rate']:.1%} ({result['n_false_positives']}/{result['n_trials']})"
            )

    return pd.DataFrame(results)


def test_false_positive_rate(
    n_trials=100,
    num_models=100,
    num_benchmarks=20,
    cutoff_year=2027,
    observation_years=3.0,
    acceleration_factor=2.0,
    min_r2=0.6,
    noise_std_model=0.4,
    noise_std_bench=0.4,
    **kwargs,
):
    """
    Test false positive rate by running detection on data with NO acceleration.

    Generates synthetic data where capabilities grow linearly (no speedup)
    and checks how often the algorithm incorrectly detects acceleration.

    IMPORTANT: False positive rate depends on the observation window (how long
    you observe after the cutoff). Longer observation windows give more
    opportunities for noise to create false detections, so FPR may increase.

    Parameters:
    -----------
    observation_years : float
        How many years to observe after cutoff_year. The detection algorithm
        will scan from cutoff_year to cutoff_year + observation_years.
        Longer observation windows may increase false positive rate.

    Returns:
    --------
    dict with false_positive_rate and details
    """
    false_positives = 0
    detection_times = []

    # Calculate time range based on observation window
    time_range_start = cutoff_year - 3.0  # 3 years before cutoff for baseline
    time_range_end = cutoff_year + observation_years

    print(f"Testing false positive rate ({n_trials} trials, NO acceleration)...")
    print(
        f"  Observation window: {observation_years} years after cutoff ({cutoff_year:.1f} to {time_range_end:.1f})"
    )

    for trial in range(n_trials):
        # Generate data with NO acceleration (speedup_factor=1)
        models_df, benchmarks_df, scores_df = generate_data(
            num_models=num_models,
            num_benchmarks=num_benchmarks,
            speedup_factor_model=1.0,  # NO ACCELERATION
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            cutoff_year=cutoff_year,
            frac_eval=kwargs.get("frac_eval", 0.25),
            error_std=kwargs.get("error_std", 0.025),
            noise_std_model=noise_std_model,
            noise_std_bench=noise_std_bench,
        )

        # Try to estimate capabilities
        try:
            df_est = estimated_capabilities(models_df, benchmarks_df, scores_df)
            if df_est.empty or len(df_est) < 5:
                continue

            df_est = df_est.sort_values("date").copy()
            df_est["frontier"] = df_est["estimated_capability"].cummax()

            x = df_est["date"].values
            y = df_est["frontier"].values

            # Try to detect
            result = detect_acceleration_sequential(
                x,
                y,
                cutoff_year=cutoff_year,
                min_acceleration=acceleration_factor,
                min_r2=min_r2,
                scan_resolution=50,
                verbose=False,
            )

            if result is not None:
                false_positives += 1
                detection_times.append(result["breakpoint"])
        except:
            continue

        if (trial + 1) % 20 == 0:
            print(f"  {trial + 1}/{n_trials} trials complete...")

    fpr = false_positives / n_trials

    print(f"\nResults:")
    print(f"  False Positive Rate: {fpr:.1%} ({false_positives}/{n_trials})")

    return {
        "false_positive_rate": fpr,
        "n_false_positives": false_positives,
        "n_trials": n_trials,
        "detection_times": detection_times,
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_synthetic_data_over_time(models_df, benchmarks_df, output_dir, colors):
    """
    Plot 1: Synthetic data showing model capabilities and benchmark difficulties
    over time.

    Shows the ground truth data with acceleration visible in the model capabilities.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot model capabilities
    ax.scatter(
        models_df["date"],
        models_df["model_capabilities"],
        alpha=0.6,
        s=30,
        label="Model capabilities",
        color=colors[0],  # teal
    )

    # Plot benchmark difficulties
    ax.scatter(
        benchmarks_df["benchmark_release_date"],
        benchmarks_df["benchmark_difficulties"],
        alpha=0.6,
        s=30,
        label="Benchmark difficulties",
        color=colors[2],  # orange
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Capability / Difficulty")
    ax.set_title("Synthetic Data: Model Capabilities and Benchmark Difficulties Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "synthetic_data_over_time.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_detection_demonstration(
    models_df,
    benchmarks_df,
    scores_df,
    cutoff_year,
    acceleration_factor,
    output_dir,
    colors,
):
    """
    Plot 2: Detection algorithm demonstration showing whether acceleration is
    detected and when.

    Shows:
    - All model estimated capabilities (scatter)
    - Frontier points highlighted (larger markers)
    - Piecewise linear fit if detected (line)
    - Cutoff year (vertical line)
    - Detection time (vertical line if detected)
    """
    # Estimate capabilities
    df_est = estimated_capabilities(models_df, benchmarks_df, scores_df, verbose=False)

    if df_est.empty:
        print("  Warning: Could not estimate capabilities for detection plot")
        return

    # Compute frontier
    df_est = df_est.sort_values("date").copy()
    df_est["frontier"] = df_est["estimated_capability"].cummax()

    # Identify frontier points (models where capability equals frontier)
    df_est["is_frontier"] = df_est["estimated_capability"] == df_est["frontier"]

    # Run detection
    result = estimate_detection_for_single_trajectory(
        models_df,
        benchmarks_df,
        scores_df,
        cutoff_year=cutoff_year,
        acceleration_factor=acceleration_factor,
        verbose=False,
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all estimated capabilities (non-frontier) - using teal from first plot
    non_frontier = df_est[~df_est["is_frontier"]]
    ax.scatter(
        non_frontier["date"],
        non_frontier["estimated_capability"],
        alpha=0.4,
        s=30,
        color=colors[0],  # teal (matches model capabilities in first plot)
    )

    # Highlight frontier points
    frontier_points = df_est[df_est["is_frontier"]]
    ax.scatter(
        frontier_points["date"],
        frontier_points["estimated_capability"],
        alpha=0.8,
        s=80,
        color=colors[7],  # green (darker)
        edgecolors='white',
        linewidths=1,
        zorder=5,
    )

    # If detected, plot breakpoint and piecewise fit
    if result["detected"]:
        bp = result["breakpoint"]
        detection_time = result["detection_time"]

        # Plot estimated cutoff (breakpoint)
        ax.axvline(
            bp,
            color=colors[10],  # red
            linestyle="--",
            alpha=0.7,
            linewidth=2,
        )

        # Recompute piecewise fit on the FULL frontier for visualization
        # (The detection used a subset of data, but we want to show the fit on all data)
        x_frontier = df_est["date"].values
        y_frontier = df_est["frontier"].values

        refit_result = fit_two_segments_fixed_breakpoint(x_frontier, y_frontier, bp)

        if refit_result is not None:
            m1 = refit_result["slope1"]
            b1 = refit_result["intercept1"]
            m2 = refit_result["slope2"]
            b2 = refit_result["intercept2"]
        else:
            # Fallback to original slopes if refit fails
            m1 = result["slope_before"]
            m2 = result["slope_after"]
            frontier_at_bp_idx = np.argmin(np.abs(df_est["date"].values - bp))
            y_at_bp = df_est["frontier"].values[frontier_at_bp_idx]
            b1 = y_at_bp - m1 * bp
            b2 = y_at_bp - m2 * bp

        x_range = df_est["date"].values
        x_min, x_max = x_range.min(), x_range.max()
        x_fine = np.linspace(x_min, x_max, 200)
        y_fit = np.where(x_fine < bp, m1 * x_fine + b1, m2 * x_fine + b2)

        # Compute ratio from refit if available
        if refit_result is not None and m1 > 0:
            fit_ratio = m2 / m1
        else:
            fit_ratio = result['ratio']

        ax.plot(
            x_fine,
            y_fit,
            linestyle=":",
            linewidth=2.5,
            color=colors[3],  # purple
            alpha=0.8,
        )

        title = f"Detection time: {detection_time-cutoff_year:.2f} years"
    else:
        title = f"NO DETECTION (looking for {acceleration_factor}x acceleration)"

    ax.set_xlabel("Year")
    ax.set_ylabel("Capability")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "detection_demonstration.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

    # Print detection summary
    print(f"\n  Detection Summary:")
    print(f"    Total models generated: {len(models_df)}")
    print(f"    Models with estimates: {len(df_est)}")
    print(f"    Frontier models: {df_est['is_frontier'].sum()}")
    print(f"    Detected: {result['detected']}")
    if result["detected"]:
        print(f"    Years to detect: {result['years_to_detect']:.2f}")
        print(f"    Detection time: {result['detection_time']:.2f}")
        print(f"    Breakpoint: {result['breakpoint']:.2f}")
        print(f"    Detected ratio: {result['ratio']:.2f}x")
        print(f"    R²: {result['r2']:.3f}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Detection Analysis: Parameter Sweep and False Positive Testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/software_singularity",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick version with fewer simulations"
    )
    parser.add_argument(
        "--false-positive-only",
        action="store_true",
        help="Only run false positive testing, skip parameter sweep",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up custom styling
    colors = setup_custom_style()

    # ========================================================================
    # VISUALIZATION: Generate example plots
    # ========================================================================

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 80)

    # ========================================================================
    # Generate synthetic data (matching notebook cell 9)
    # ========================================================================
    print("\nCreating synthetic data (matching notebook cell 9)...")

    models_broad, benchmarks_broad, df_broad = generate_data(
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
        noise_std_model=0.25,
        noise_std_bench=0.25,
        base_bench=0.5,
        saturation_level=0.05,
        min_alpha=2.5,
        max_alpha=10,
        frac_accelerate_models=0.25,
        random_seed=42,  # Fixed seed for reproducible visualizations
    )

    print(f"  Generated {len(models_broad)} models, {len(benchmarks_broad)} benchmarks")
    print(f"  {len(df_broad)} evaluation scores")

    # ========================================================================
    # Plot 1: Synthetic data over time
    # ========================================================================
    print("\nGenerating Plot 1: Synthetic data over time...")
    plot_synthetic_data_over_time(models_broad, benchmarks_broad, output_dir, colors)

    # ========================================================================
    # Plot 2: Detection demonstration (using same data as Plot 1)
    # ========================================================================
    print("\nGenerating Plot 2: Detection demonstration (using same data)...")
    plot_detection_demonstration(
        models_broad,
        benchmarks_broad,
        df_broad,
        cutoff_year=2027,
        acceleration_factor=2.0,
        output_dir=output_dir,
        colors=colors,
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION PLOTS COMPLETE")
    print("=" * 80)

    if args.false_positive_only:
        print("=" * 80)
        print("FALSE POSITIVE RATE TESTING")
        print("=" * 80)
    else:
        print("=" * 80)
        print("DETECTION ANALYSIS: PARAMETER SWEEP AND FALSE POSITIVE TESTING")
        print("=" * 80)
        print("\nThis analysis runs:")
        print(
            "  - Parameter sweep across model/benchmark release rates and acceleration factors"
        )
        print("  - False positive rate testing to validate detection specificity\n")

    # ========================================================================
    # PART 1: Parameter Sweep
    # ========================================================================

    if not args.false_positive_only:
        print("\n" + "=" * 80)
        print("PART 1: Parameter Sweep")
        print("=" * 80)

        if args.quick:
            print("\nRunning QUICK parameter sweep (fewer simulations)...")
            models_per_year_list = [20, 40]
            benchmarks_per_year_list = [5, 10]
            acceleration_factors = [2, 4]
            noise_multipliers = [1.0, 2.0]
            n_simulations = 2
        else:
            print("\nRunning FULL parameter sweep...")
            models_per_year_list = [10, 30, 50]
            benchmarks_per_year_list = [5, 10, 15]
            acceleration_factors = [2, 4, 8]
            noise_multipliers = [1.0, 2.0]
            n_simulations = 3

        results_df = run_detection_sweep(
            models_per_year_list=models_per_year_list,
            benchmarks_per_year_list=benchmarks_per_year_list,
            acceleration_factors=acceleration_factors,
            noise_multipliers=noise_multipliers,
            n_simulations=n_simulations,
            horizon_years=6,
            cutoff_year=2027,
            detection_threshold=2.0,
        )

        # Save results
        results_path = output_dir / "detection_sweep_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved results: {results_path}")

        # Display summary
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP RESULTS")
        print("=" * 80)
        print(
            results_df.sort_values(
                [
                    "accel_factor",
                    "noise_multiplier",
                    "models_per_year",
                    "benchmarks_per_year",
                ]
            )
        )

    # ========================================================================
    # PART 2: False Positive Testing
    # ========================================================================

    print("\n" + "=" * 80)
    if args.false_positive_only:
        print("False Positive Rate Testing")
    else:
        print("PART 2: False Positive Rate Testing")
    print("=" * 80)

    print("\nRunning false positive rate sweep...")
    print(
        "Note: Observation window affects false positive rate (longer = more opportunities for false detections)"
    )

    # Define sweep parameters
    noise_multipliers = [1.0, 2.0, 4.0]  # Same as parameter sweep, plus higher noise
    observation_years_list = [0.5, 1.0, 2.0, 3.0, 6.0]
    n_trials = 50 if args.quick else 100

    fpr_results_df = test_false_positive_rate_sweep(
        noise_multipliers=noise_multipliers,
        observation_years_list=observation_years_list,
        n_trials=n_trials,
        num_models=100,
        num_benchmarks=20,
        cutoff_year=2027,
        acceleration_factor=2.0,
        min_r2=0.6,
    )

    # Save results
    fpr_results_path = output_dir / "false_positive_rate_sweep_results.csv"
    fpr_results_df.to_csv(fpr_results_path, index=False)
    print(f"\nSaved false positive rate sweep results: {fpr_results_path}")

    # Display summary
    print("\n" + "=" * 80)
    print("FALSE POSITIVE RATE SWEEP RESULTS")
    print("=" * 80)
    print(fpr_results_df.sort_values(["noise_multiplier", "observation_years"]))

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nKey findings:")
    if not args.false_positive_only:
        print("  - Detection time depends on model/benchmark release rates")
        print("  - Higher acceleration is easier/faster to detect")

    # Summary of false positive rates
    min_fpr = fpr_results_df["false_positive_rate"].min()
    max_fpr = fpr_results_df["false_positive_rate"].max()
    mean_fpr = fpr_results_df["false_positive_rate"].mean()
    print(
        f"  - False positive rate range: {min_fpr:.1%} to {max_fpr:.1%} (mean: {mean_fpr:.1%})"
    )
    print("  - Longer observation windows generally increase false positive rate")
    print("  - Higher noise levels generally increase false positive rate")

    print("\nNext steps:")
    if not args.false_positive_only:
        print("  - Check detection_sweep_results.csv for detailed results")
    print("  - Check false_positive_rate_sweep_results.csv for false positive analysis")
    print("  - Adjust parameters based on false positive rate tolerance")
    print("=" * 80)


if __name__ == "__main__":
    main()
