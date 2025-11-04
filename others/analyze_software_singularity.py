"""
Software Singularity Analysis

This script analyzes how quickly we can detect acceleration in AI capability progress
using temporal detection methods. It simulates scenarios where AI capabilities undergo
acceleration (e.g., 2x, 3x speedup) and estimates how long it takes to detect such
changes given different rates of model and benchmark releases.

Key concepts:
- Cutoff year: When acceleration actually happens
- Breakpoint: Where we estimate the acceleration is
- Detection time: When we have enough data to detect
- Years to detect: How long detection takes (detection_time - cutoff_year)

Main analyses:
1. Synthetic data generation with configurable acceleration
2. Temporal detection simulation (incremental data revelation)
3. Parameter sweep across model/benchmark release rates and acceleration factors
4. False positive rate testing and control
5. Visualization of detection results
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from sklearn.metrics import r2_score


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default output directory
OUTPUT_DIR = Path("outputs/software_singularity")

# Global storage for run data (used for visualizations)
RUN_DATA_STORAGE = {}


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
    random_seed=None
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
    accelerate_mask = (model_times >= cutoff_year) & (random_draws < frac_accelerate_models)

    # Compute capabilities with acceleration
    model_capabilities = (
        base_model
        + np.where(
            accelerate_mask,
            # Accelerated: baseline until cutoff, then faster slope
            slope_model * (cutoff_year - time_range_start) +
            speedup_factor_model * slope_model * (model_times - cutoff_year),
            # Normal: same slope the whole time
            slope_model * (model_times - time_range_start)
        )
        + np.random.normal(0, noise_std_model, num_models)
    )

    models = pd.DataFrame({
        'model_id': np.arange(num_models),
        'date': model_times,
        'model_capabilities': model_capabilities,
        'accelerated': accelerate_mask
    })

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
            slope_bench * (cutoff_year - time_range_start) +
            1 * slope_bench * (benchmark_times - cutoff_year)
        )
        + np.random.normal(0, noise_std_bench, num_benchmarks)
    )
    benchmark_progress_slopes = np.random.uniform(
        min_alpha, max_alpha, num_benchmarks
    )

    benchmarks = pd.DataFrame({
        'benchmark_id': np.arange(num_benchmarks),
        'benchmark_release_date': benchmark_times,
        'benchmark_difficulties': benchmark_difficulties,
        'benchmark_progress_slopes': benchmark_progress_slopes
    })

    # Generate evaluation scores using logistic function
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    scores = []
    for _, m in models.iterrows():
        raw = logistic(
            benchmarks['benchmark_progress_slopes']
            * (m['model_capabilities'] - benchmarks['benchmark_difficulties'])
        )
        mask = (raw >= saturation_level) & (raw <= 1 - saturation_level)
        for b_idx, s in zip(benchmarks['benchmark_id'][mask], raw[mask]):
            if np.random.rand() < frac_eval:
                noisy = np.clip(s + np.random.normal(0, error_std), 0, 1 - saturation_level)
                scores.append({
                    'model_id': m['model_id'],
                    'benchmark_id': b_idx,
                    'date': m['date'],
                    'performance': noisy
                })

    df_scores = pd.DataFrame(scores)
    return models, benchmarks, df_scores


# ============================================================================
# CAPABILITY ESTIMATION
# ============================================================================

def estimated_capabilities(models, benchmarks, df, regularization_strength=0.1):
    """
    Estimate model capabilities from benchmark scores using logistic regression.

    Fits the model: performance(m,b) = σ(α_b(C_m - D_b))
    where C_m = capability, D_b = difficulty, α_b = slope, σ = sigmoid

    Returns:
    --------
    DataFrame with columns: model_id, date, model_capabilities,
                           estimated_capability, unaligned_C
    """
    # Identify valid models (those with data)
    valid_model_ids = sorted(df["model_id"].unique())
    skipped_model_ids = set(models["model_id"]) - set(valid_model_ids)
    if skipped_model_ids:
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
        D = params[num_valid_models:num_valid_models + num_benchmarks]
        alpha = params[num_valid_models + num_benchmarks:]

        c_vals = C[model_idx]
        d_vals = D[bench_idx]
        alpha_vals = alpha[bench_idx]

        preds = logistic(alpha_vals * (c_vals - d_vals))
        residuals = preds - y

        # Add L2 regularization
        if regularization_strength > 0:
            reg_term = regularization_strength * (
                np.sum(C**2) + np.sum(D**2) + np.sum(alpha**2)
            ) / (num_valid_models + num_benchmarks + num_benchmarks)
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
    lower_bounds = np.concatenate([
        np.full(num_valid_models, -10),
        np.full(num_benchmarks, -10),
        np.full(num_benchmarks, 0.1)
    ])
    upper_bounds = np.concatenate([
        np.full(num_valid_models, 10),
        np.full(num_benchmarks, 10),
        np.full(num_benchmarks, 10)
    ])

    # Fit with bounds
    result = least_squares(
        residuals,
        initial_params,
        args=(model_idx_for_data, benchmark_ids_for_data, observed_scores),
        bounds=(lower_bounds, upper_bounds),
        method="trf"
    )

    estimated_params = result.x
    estimated_C = estimated_params[:num_valid_models]

    # Map fitted capabilities back to original model IDs
    fitted_C_df = pd.DataFrame({
        "model_id": valid_model_ids,
        "unaligned_C": estimated_C
    })

    meta = models[["model_id", "model_capabilities", "date"]]
    fitted_C_df = fitted_C_df.merge(meta, on="model_id", how="left")

    # Compute alignment transform
    a, b = np.polyfit(
        fitted_C_df["unaligned_C"].values,
        fitted_C_df["model_capabilities"].values,
        1
    )

    fitted_C_df["estimated_capability"] = (
        a * fitted_C_df["unaligned_C"] + b
    )

    return fitted_C_df


# ============================================================================
# PIECEWISE LINEAR FITTING
# ============================================================================

def piecewise_linear(x, slope1, intercept1, slope2, breakpoint):
    """
    Piecewise linear function with two segments that meet at breakpoint.
    """
    intercept2 = slope1 * breakpoint + intercept1 - slope2 * breakpoint
    return np.where(x < breakpoint, slope1*x + intercept1, slope2*x + intercept2)


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
    y_pred = np.where(x <= breakpoint,
                     slope1 * x + intercept1,
                     slope2 * x + intercept2)
    r2 = r2_score(y, y_pred)

    return {
        'slope1': slope1,
        'intercept1': intercept1,
        'slope2': slope2,
        'intercept2': intercept2,
        'breakpoint': breakpoint,
        'r2': r2
    }


# ============================================================================
# ACCELERATION DETECTION
# ============================================================================

def detect_acceleration_sequential(x, y, cutoff_year,
                                  min_acceleration=2.0,
                                  min_r2=0.6,
                                  min_gap_years=0.0,
                                  scan_resolution=50,
                                  min_points_after=3,
                                  verbose=False):
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
        print(f"  Scanning {len(breakpoints)} breakpoints from {min_bp:.2f} to {max_bp:.2f}")

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
        if result['slope1'] <= 0:
            continue

        # Compute slope ratio
        ratio = result['slope2'] / result['slope1']

        # Check detection criteria
        if result['r2'] >= min_r2 and ratio >= min_acceleration:
            if verbose:
                print(f"  ✓ DETECTED at breakpoint {bp:.3f}")
                print(f"    Slope before: {result['slope1']:.4f}")
                print(f"    Slope after: {result['slope2']:.4f}")
                print(f"    Ratio: {ratio:.2f}x")
                print(f"    R²: {result['r2']:.4f}")

            result['ratio'] = ratio
            result['detected'] = True
            result['points_after'] = points_after
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
    verbose=True
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
    post_cutoff_models = models_df[models_df['date'] > cutoff_year].copy()

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
            "r2": None
        }

    # Create time checkpoints to test
    min_time = cutoff_year + min_gap_years
    max_time = models_df['date'].max()

    # Use model release times as natural checkpoints
    model_times = sorted(post_cutoff_models['date'].unique())
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
            "r2": None
        }

    if verbose:
        print(f"\n  Temporal Detection Simulation")
        print(f"  Cutoff year: {cutoff_year:.2f}")
        print(f"  Testing {len(checkpoint_times)} time checkpoints")
        print(f"  Looking for {acceleration_factor}x acceleration\n")

    # Incrementally reveal data and check for detection at each time point
    for current_time in checkpoint_times:
        # Data available up to current_time
        scores_up_to_now = scores_df[scores_df['date'] <= current_time].copy()

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
            x, y,
            cutoff_year=cutoff_year,
            min_acceleration=acceleration_factor,
            min_r2=min_r2,
            min_gap_years=min_gap_years,
            scan_resolution=scan_resolution,
            min_points_after=min_points_after,
            verbose=False
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
                "breakpoint": result['breakpoint'],
                "slope_before": result['slope1'],
                "slope_after": result['slope2'],
                "ratio": result['ratio'],
                "r2": result['r2']
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
        "r2": None
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
    store_runs=True
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
    total_runs = (len(models_per_year_list) * len(benchmarks_per_year_list) *
                 len(acceleration_factors) * len(noise_multipliers) * n_simulations)
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
                        random_seed=seed
                    )

                    # Estimate capabilities
                    df_est = estimated_capabilities(models_df, benchmarks_df, scores_df)

                    # Estimate detection
                    result = estimate_detection_for_single_trajectory(
                        models_df,
                        benchmarks_df,
                        scores_df,
                        cutoff_year=cutoff_year,
                        acceleration_factor=detection_threshold,
                        verbose=False
                    )

                    # Store run data if requested
                    if store_runs:
                        RUN_DATA_STORAGE[run_idx] = {
                            'models_df': models_df.copy(),
                            'benchmarks_df': benchmarks_df.copy(),
                            'scores_df': scores_df.copy(),
                            'df_est': df_est.copy(),
                            'result': result.copy(),
                            'params': {
                                'models_per_year': models_per_year,
                                'benchmarks_per_year': benchmarks_per_year,
                                'accel_factor': accel_factor,
                                'noise_multiplier': noise_mult,
                                'sim_idx': sim_idx,
                                'seed': seed,
                                'cutoff_year': cutoff_year
                            }
                        }

                    detections.append(result["detected"])
                    if result["detected"]:
                        detection_times.append(result["years_to_detect"])

                    if run_idx % 10 == 0:
                        print(f"Progress: {run_idx}/{total_runs} runs complete")

                # Aggregate results for this parameter combination
                detected_fraction = np.mean(detections)
                mean_detection_time = np.mean(detection_times) if detection_times else None

                results.append({
                    "models_per_year": models_per_year,
                    "benchmarks_per_year": benchmarks_per_year,
                    "accel_factor": accel_factor,
                    "noise_multiplier": noise_mult,
                    "detected_fraction": detected_fraction,
                    "mean_years_to_detect": mean_detection_time,
                    "n_detected": sum(detections),
                    "n_total": n_simulations
                })

    print(f"\nSweep complete!")
    if store_runs:
        print(f"Stored {len(RUN_DATA_STORAGE)} runs for visualization\n")

    return pd.DataFrame(results)


# ============================================================================
# FALSE POSITIVE TESTING
# ============================================================================

def test_false_positive_rate(n_trials=100,
                             num_models=100,
                             num_benchmarks=20,
                             cutoff_year=2027,
                             acceleration_factor=2.0,
                             min_r2=0.6,
                             noise_std_model=0.4,
                             noise_std_bench=0.4,
                             **kwargs):
    """
    Test false positive rate by running detection on data with NO acceleration.

    Generates synthetic data where capabilities grow linearly (no speedup)
    and checks how often the algorithm incorrectly detects acceleration.

    This helps tune detection parameters to achieve desired false positive rate.

    Returns:
    --------
    dict with false_positive_rate and details
    """
    false_positives = 0
    detection_times = []

    print(f"Testing false positive rate ({n_trials} trials, NO acceleration)...")

    for trial in range(n_trials):
        # Generate data with NO acceleration (speedup_factor=1)
        models_df, benchmarks_df, scores_df = generate_data(
            num_models=num_models,
            num_benchmarks=num_benchmarks,
            speedup_factor_model=1.0,  # NO ACCELERATION
            time_range_start=2024,
            time_range_end=2030,
            cutoff_year=cutoff_year,
            frac_eval=kwargs.get('frac_eval', 0.25),
            error_std=kwargs.get('error_std', 0.025),
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
                x, y,
                cutoff_year=cutoff_year,
                min_acceleration=acceleration_factor,
                min_r2=min_r2,
                scan_resolution=50,
                verbose=False
            )

            if result is not None:
                false_positives += 1
                detection_times.append(result['breakpoint'])
        except:
            continue

        if (trial + 1) % 20 == 0:
            print(f"  {trial + 1}/{n_trials} trials complete...")

    fpr = false_positives / n_trials

    print(f"\nResults:")
    print(f"  False Positive Rate: {fpr:.1%} ({false_positives}/{n_trials})")

    return {
        'false_positive_rate': fpr,
        'n_false_positives': false_positives,
        'n_trials': n_trials,
        'detection_times': detection_times
    }


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def plot_synthetic_capabilities(models, benchmarks, output_path):
    """Plot true capabilities and difficulties from synthetic data."""
    plt.figure(figsize=(10, 6))
    plt.scatter(models["date"], models["model_capabilities"],
               alpha=0.5, s=30, label="Model capabilities")
    plt.scatter(benchmarks["benchmark_release_date"],
               benchmarks["benchmark_difficulties"],
               alpha=0.5, s=30, label="Benchmark difficulties")
    plt.xlabel("Year")
    plt.ylabel("Capability / Difficulty")
    plt.title("Synthetic Data: True Capabilities and Difficulties")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save as both PDF and PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = str(output_path).replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")
    print(f"Saved plot: {png_path}")


def plot_estimation_quality(df_est, output_path):
    """Plot true vs estimated capabilities."""
    plt.figure(figsize=(8, 8))
    plt.scatter(df_est["model_capabilities"], df_est["estimated_capability"],
               alpha=0.5, s=30)

    # Add diagonal line
    min_val = min(df_est["model_capabilities"].min(), df_est["estimated_capability"].min())
    max_val = max(df_est["model_capabilities"].max(), df_est["estimated_capability"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect estimation')

    plt.xlabel("True Capability")
    plt.ylabel("Estimated Capability")
    plt.title("Estimation Quality")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save as both PDF and PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = str(output_path).replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")
    print(f"Saved plot: {png_path}")


def plot_frontier_analysis(models, df_est, cutoff_year, output_path):
    """Plot capability frontier over time."""
    df = models.sort_values("date").copy()
    df["running_max"] = df["model_capabilities"].cummax()

    plt.figure(figsize=(12, 6))
    plt.scatter(df["date"], df["model_capabilities"],
               alpha=0.2, s=20, label="All models")
    plt.plot(df["date"], df["running_max"],
            'r-', lw=2, label="True frontier")

    if not df_est.empty:
        df_est_sorted = df_est.sort_values("date").copy()
        df_est_sorted["frontier"] = df_est_sorted["estimated_capability"].cummax()
        plt.plot(df_est_sorted["date"], df_est_sorted["frontier"],
                'b-', lw=2, label="Estimated frontier")

    plt.axvline(cutoff_year, color='gray', linestyle='--',
               alpha=0.7, label=f'Cutoff ({cutoff_year})')

    plt.xlabel("Year")
    plt.ylabel("Capability")
    plt.title("Frontier Analysis: True vs Estimated")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save as both PDF and PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = str(output_path).replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")
    print(f"Saved plot: {png_path}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Software Singularity Analysis: Temporal Detection of AI Capability Acceleration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/software_singularity",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version with fewer simulations"
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.0,
        help="Noise multiplier (1.0 = baseline, 2.0 = 2x noise)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SOFTWARE SINGULARITY ANALYSIS")
    print("="*80)
    print("\nThis analysis simulates temporal detection of AI capability acceleration.")
    print("It answers: How quickly can we detect Nx acceleration given different")
    print("model/benchmark release rates?\n")

    # ========================================================================
    # PART 1: Generate and Visualize Synthetic Data
    # ========================================================================

    print("\n" + "="*80)
    print("PART 1: Synthetic Data Generation")
    print("="*80)

    print("\nGenerating synthetic data with 2x acceleration starting in 2027...")
    models, benchmarks, scores = generate_data(
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
        frac_accelerate_models=0.25,  # Only 25% of post-cutoff models accelerate
        random_seed=42  # Fixed seed for reproducible demo
    )

    print(f"  Generated {len(models)} models, {len(benchmarks)} benchmarks, {len(scores)} scores")
    print(f"  {models['accelerated'].sum()} models have accelerated capabilities")

    plot_synthetic_capabilities(
        models, benchmarks,
        output_dir / "synthetic_capabilities_and_difficulties.pdf"
    )

    # ========================================================================
    # PART 2: Test Capability Estimation
    # ========================================================================

    print("\n" + "="*80)
    print("PART 2: Capability Estimation")
    print("="*80)

    print("\nEstimating capabilities from synthetic scores...")
    df_est = estimated_capabilities(models, benchmarks, scores)
    print(f"  Successfully estimated capabilities for {len(df_est)} models")

    plot_estimation_quality(
        df_est,
        output_dir / "estimation_quality.pdf"
    )

    plot_frontier_analysis(
        models, df_est, 2027,
        output_dir / "frontier_analysis.pdf"
    )

    # ========================================================================
    # PART 3: Single Trajectory Detection Example
    # ========================================================================

    print("\n" + "="*80)
    print("PART 3: Single Trajectory Detection Example")
    print("="*80)

    print("\nGenerating test case with 3x acceleration starting in 2027...")
    models_test, benchmarks_test, scores_test = generate_data(
        num_models=100,
        num_benchmarks=20,
        speedup_factor_model=3.0,
        time_range_start=2024,
        time_range_end=2030,
        cutoff_year=2027,
        frac_eval=0.3,
        error_std=0.02,
        elo_change=3.5,
        noise_std_model=0.15,
        noise_std_bench=0.15,
        frac_accelerate_models=1.0,
        random_seed=123  # Fixed seed for reproducible demo
    )

    print("\nRunning temporal detection simulation...")
    result_test = estimate_detection_for_single_trajectory(
        models_test,
        benchmarks_test,
        scores_test,
        cutoff_year=2027,
        acceleration_factor=2.0,
        verbose=True
    )

    if result_test["detected"]:
        print(f"\n✓ Detection successful!")
        print(f"  Detection time: {result_test['detection_time']:.2f}")
        print(f"  Years to detect: {result_test['years_to_detect']:.2f}")
        print(f"  Detected ratio: {result_test['ratio']:.2f}x")
    else:
        print(f"\n✗ No detection in this scenario")

    # ========================================================================
    # PART 4: Parameter Sweep
    # ========================================================================

    print("\n" + "="*80)
    print("PART 4: Parameter Sweep")
    print("="*80)

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
        detection_threshold=2.0
    )

    # Save results
    results_path = output_dir / "detection_sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    # Display summary
    print("\n" + "="*80)
    print("PARAMETER SWEEP RESULTS")
    print("="*80)
    print(results_df.sort_values(["accel_factor", "noise_multiplier", "models_per_year", "benchmarks_per_year"]))

    # ========================================================================
    # PART 5: False Positive Testing
    # ========================================================================

    print("\n" + "="*80)
    print("PART 5: False Positive Rate Testing")
    print("="*80)

    print("\nTesting false positive rate with NO acceleration...")
    fpr_result = test_false_positive_rate(
        n_trials=50 if args.quick else 100,
        num_models=100,
        num_benchmarks=20,
        acceleration_factor=2.0,
        min_r2=0.6,
        noise_std_model=0.2,
        noise_std_bench=0.2
    )

    print(f"\n  False Positive Rate: {fpr_result['false_positive_rate']:.1%}")
    print(f"  ({fpr_result['n_false_positives']}/{fpr_result['n_trials']} trials)")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nKey findings:")
    print("  - Temporal detection simulates real evidence accumulation")
    print("  - Detection time depends on model/benchmark release rates")
    print("  - Higher acceleration is easier/faster to detect")
    print(f"  - False positive rate: ~{fpr_result['false_positive_rate']:.1%} with current settings")
    print("\nNext steps:")
    print("  - Review plots in outputs/software_singularity/")
    print("  - Check detection_sweep_results.csv for detailed results")
    print("  - Adjust parameters based on false positive rate tolerance")
    print("="*80)


if __name__ == "__main__":
    main()
