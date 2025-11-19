"""
Detection Analysis: Parameter Sweep and False Positive Testing

This script runs two main analyses:
1. Parameter sweep across model/benchmark release rates and acceleration factors
2. False positive rate testing to validate detection specificity

This is a focused version that contains only the parameter sweep and false positive
testing components from the full software singularity analysis.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class SimulationConfig:
    """Fixed parameters used across all analyses.

    These parameters are NOT varied in the parameter sweep and should remain
    constant unless you're exploring different experimental setups.
    """

    # ========================================================================
    # Temporal Parameters
    # ========================================================================
    time_range_start: int = 2024
    """Start year for simulation in sweeps and examples"""

    cutoff_year: int = 2027
    """Year when acceleration begins"""

    horizon_years: int = 6
    """Number of years to simulate (time_range_start to time_range_start + horizon_years)"""

    # ========================================================================
    # Data Generation Parameters
    # ========================================================================
    elo_change: float = 3.5
    """Total capability change over the time range"""

    frac_eval: float = 0.25
    """Fraction of (model, benchmark) pairs that are evaluated"""

    # ========================================================================
    # Noise Parameters (base values before noise_multiplier is applied)
    # ========================================================================
    base_error_std: float = 0.025
    """Base standard deviation of noise added to evaluation scores"""

    base_noise_std_model: float = 0.2
    """Base standard deviation of noise added to model capabilities"""

    base_noise_std_bench: float = 0.2
    """Base standard deviation of noise added to benchmark difficulties"""

    # ========================================================================
    # Detection Parameters
    # ========================================================================
    detection_threshold: float = 2.0
    """Minimum slope ratio to declare detection (e.g., 2.0 = detect if ≥2x acceleration)"""

    min_r2: float = 0.6
    """Minimum R² for piecewise linear fit"""

    scan_resolution: int = 50
    """Number of candidate breakpoints to test"""

    min_gap_years: float = 0.0
    """Minimum time after cutoff to start scanning for detection"""

    min_points_after: int = 3
    """Minimum data points required after breakpoint"""

    # ========================================================================
    # False Positive Testing Parameters
    # ========================================================================
    fpr_num_models: int = 100
    """Number of models for false positive rate testing"""

    fpr_num_benchmarks: int = 20
    """Number of benchmarks for false positive rate testing"""

    fpr_time_offset: float = 3.0
    """Years before cutoff to start false positive test data (cutoff - offset)"""


# Global configuration instance
CONFIG = SimulationConfig()

# Default output directory
OUTPUT_DIR = Path("outputs/software_singularity")

# Global storage for run data (used for visualizations)
RUN_DATA_STORAGE = {}


# ============================================================================
# DATA GENERATION
# ============================================================================


def generate_data(
    num_models,
    num_benchmarks,
    true_acceleration,
    time_range_start,
    time_range_end,
    cutoff_year,
    noise_std_model,
    noise_std_bench,
    error_std,
    elo_change=CONFIG.elo_change,
    frac_eval=CONFIG.frac_eval,
    frac_accelerate_models=1.0,
    base_model=0,
    base_bench=0.5,
    saturation_level=0.05,
    min_alpha=3,
    max_alpha=10,
    random_seed=None,
):
    """
    Generate synthetic benchmark evaluation data with optional acceleration.

    Models released after cutoff_year can have accelerated capability growth.
    Benchmarks track the frontier, with difficulty scaling over time.

    Required Parameters:
    -------------------
    num_models : int
        Number of models to generate
    num_benchmarks : int
        Number of benchmarks to generate
    true_acceleration : float
        Multiplier for capability growth rate after cutoff (e.g., 2 = 2x faster)
    time_range_start : float
        Start year for simulation
    time_range_end : float
        End year for simulation
    cutoff_year : float
        Year when acceleration begins
    noise_std_model : float
        Standard deviation of noise added to model capabilities
    noise_std_bench : float
        Standard deviation of noise added to benchmark difficulties
    error_std : float
        Standard deviation of noise added to evaluation scores

    Optional Parameters (with CONFIG defaults):
    ------------------------------------------
    elo_change : float
        Total capability change over time range (default: CONFIG.elo_change)
    frac_eval : float
        Fraction of (model, benchmark) pairs evaluated (default: CONFIG.frac_eval)
    frac_accelerate_models : float
        Fraction of post-cutoff models that accelerate (default: 1.0)
    base_model : float
        Starting capability level (default: 0)
    base_bench : float
        Starting difficulty level (default: 0.5)
    saturation_level : float
        Exclude scores near 0 or 1 (default: 0.05)
    min_alpha, max_alpha : float
        Range for benchmark slope parameters (default: 3, 10)
    random_seed : int or None
        Random seed for reproducibility (default: None)

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
            + true_acceleration * slope_model * (model_times - cutoff_year),
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
    detection_threshold=CONFIG.detection_threshold,
    min_r2=CONFIG.min_r2,
    min_gap_years=CONFIG.min_gap_years,
    scan_resolution=CONFIG.scan_resolution,
    min_points_after=CONFIG.min_points_after,
    verbose=False,
):
    """
    Sequentially scan for the FIRST point where we can detect acceleration.

    Scans from cutoff_year onwards and returns the first breakpoint where:
    1. The piecewise fit has R² >= min_r2
    2. The slope ratio (slope2/slope1) >= detection_threshold
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
        if result["r2"] >= min_r2 and ratio >= detection_threshold:
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
    detection_threshold=CONFIG.detection_threshold,
    min_r2=CONFIG.min_r2,
    min_gap_years=CONFIG.min_gap_years,
    scan_resolution=CONFIG.scan_resolution,
    min_points_after=CONFIG.min_points_after,
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
        print(f"  Looking for {detection_threshold}x acceleration\n")

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
            detection_threshold=detection_threshold,
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


def _run_single_simulation(args):
    """
    Helper function to run a single simulation.

    This is designed to be called in parallel via ProcessPoolExecutor.

    Parameters:
    -----------
    args : tuple
        (models_per_year, benchmarks_per_year, true_accel, noise_mult,
         frac_accelerate_models, sim_idx, seed, time_range_start, horizon_years,
         cutoff_year, detection_threshold, base_error_std, base_noise_std_model,
         base_noise_std_bench, min_r2, min_gap_years, scan_resolution,
         min_points_after, store_runs, run_idx)

    Returns:
    --------
    dict with keys: detected, years_to_detect, params, run_data (if store_runs)
    """
    (models_per_year, benchmarks_per_year, true_accel, noise_mult,
     frac_accelerate_models, sim_idx, seed, time_range_start, horizon_years,
     cutoff_year, detection_threshold, base_error_std, base_noise_std_model,
     base_noise_std_bench, min_r2, min_gap_years, scan_resolution,
     min_points_after, store_runs, run_idx) = args

    # Generate synthetic data
    num_models = int(models_per_year * horizon_years)
    num_benchmarks = int(benchmarks_per_year * horizon_years)

    models_df, benchmarks_df, scores_df = generate_data(
        num_models=num_models,
        num_benchmarks=num_benchmarks,
        true_acceleration=true_accel,
        time_range_start=time_range_start,
        time_range_end=time_range_start + horizon_years,
        cutoff_year=cutoff_year,
        error_std=base_error_std * noise_mult,
        noise_std_model=base_noise_std_model * noise_mult,
        noise_std_bench=base_noise_std_bench * noise_mult,
        frac_accelerate_models=frac_accelerate_models,
        random_seed=seed,
    )

    # Estimate capabilities
    df_est = estimated_capabilities(models_df, benchmarks_df, scores_df)

    # Estimate detection
    result = estimate_detection_for_single_trajectory(
        models_df,
        benchmarks_df,
        scores_df,
        cutoff_year=cutoff_year,
        detection_threshold=detection_threshold,
        min_r2=min_r2,
        min_gap_years=min_gap_years,
        scan_resolution=scan_resolution,
        min_points_after=min_points_after,
        verbose=False,
    )

    # Prepare return data
    ret = {
        "detected": result["detected"],
        "years_to_detect": result["years_to_detect"],
        "params": {
            "models_per_year": models_per_year,
            "benchmarks_per_year": benchmarks_per_year,
            "true_accel": true_accel,
            "noise_multiplier": noise_mult,
            "frac_accelerate_models": frac_accelerate_models,
            "sim_idx": sim_idx,
            "seed": seed,
            "cutoff_year": cutoff_year,
        },
        "run_idx": run_idx,
    }

    if store_runs:
        ret["run_data"] = {
            "models_df": models_df.copy(),
            "benchmarks_df": benchmarks_df.copy(),
            "scores_df": scores_df.copy(),
            "df_est": df_est.copy(),
            "result": result.copy(),
        }

    return ret


def run_detection_sweep(
    models_per_year_list,
    benchmarks_per_year_list,
    true_accelerations,
    noise_multipliers,
    frac_accelerate_models_list,
    n_simulations=5,
    time_range_start=CONFIG.time_range_start,
    horizon_years=CONFIG.horizon_years,
    cutoff_year=CONFIG.cutoff_year,
    detection_threshold=CONFIG.detection_threshold,
    random_seed_base=42,
    store_runs=True,
    n_jobs=1,
):
    """
    Run detection analysis across a grid of parameters.

    Tests different combinations of:
    - Model release rates (models per year)
    - Benchmark release rates (benchmarks per year)
    - True acceleration factors (2x, 3x, etc.)
    - Noise multipliers (1x, 2x, etc.)
    - Acceleration model fractions (fraction of post-cutoff models that accelerate)

    For each combination, runs multiple simulations and computes:
    - Detection success rate
    - Average time to detection

    Parameters:
    -----------
    n_jobs : int
        Number of parallel jobs to run. If 1, runs sequentially (default).
        If -1, uses all available CPU cores. If > 1, uses that many cores.

    Returns:
    --------
    DataFrame with columns: models_per_year, benchmarks_per_year, true_accel,
                           noise_multiplier, frac_accelerate_models, detected_fraction, mean_years_to_detect, ...
    """
    global RUN_DATA_STORAGE
    if store_runs:
        RUN_DATA_STORAGE = {}

    results = []
    total_runs = (
        len(models_per_year_list)
        * len(benchmarks_per_year_list)
        * len(true_accelerations)
        * len(noise_multipliers)
        * len(frac_accelerate_models_list)
        * n_simulations
    )

    print(f"\nRunning detection sweep: {total_runs} total simulations")
    if n_jobs != 1:
        import os
        max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
        print(f"Using {max_workers} parallel workers")
    print(f"{'='*60}\n")

    # Build list of all simulation tasks
    simulation_tasks = []
    run_idx = 0
    for models_per_year in models_per_year_list:
        for benchmarks_per_year in benchmarks_per_year_list:
            for true_accel in true_accelerations:
                for noise_mult in noise_multipliers:
                    for frac_accelerate_models in frac_accelerate_models_list:
                        for sim_idx in range(n_simulations):
                            run_idx += 1
                            seed = random_seed_base + run_idx

                            task = (
                                models_per_year, benchmarks_per_year, true_accel, noise_mult,
                                frac_accelerate_models, sim_idx, seed, time_range_start, horizon_years,
                                cutoff_year, detection_threshold, CONFIG.base_error_std,
                                CONFIG.base_noise_std_model, CONFIG.base_noise_std_bench,
                                CONFIG.min_r2, CONFIG.min_gap_years, CONFIG.scan_resolution,
                                CONFIG.min_points_after, store_runs, run_idx
                            )
                            simulation_tasks.append(task)

    # Run simulations (in parallel if n_jobs > 1)
    if n_jobs == 1:
        # Sequential execution
        simulation_results = []
        for i, task in enumerate(simulation_tasks, 1):
            result = _run_single_simulation(task)
            simulation_results.append(result)
            if i % 10 == 0 or i == len(simulation_tasks):
                print(f"Progress: {i}/{len(simulation_tasks)} runs complete")
    else:
        # Parallel execution
        import os
        max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            simulation_results = list(executor.map(_run_single_simulation, simulation_tasks))
        print(f"All {len(simulation_results)} simulations complete!")

    # Store run data if requested
    if store_runs:
        for sim_result in simulation_results:
            if "run_data" in sim_result:
                RUN_DATA_STORAGE[sim_result["run_idx"]] = {
                    **sim_result["run_data"],
                    "params": sim_result["params"],
                }

    # Aggregate results by parameter combination
    param_combos = {}
    for sim_result in simulation_results:
        params = sim_result["params"]
        key = (
            params["models_per_year"],
            params["benchmarks_per_year"],
            params["true_accel"],
            params["noise_multiplier"],
            params["frac_accelerate_models"],
        )

        if key not in param_combos:
            param_combos[key] = {
                "detections": [],
                "detection_times": [],
            }

        param_combos[key]["detections"].append(sim_result["detected"])
        if sim_result["detected"] and sim_result["years_to_detect"] is not None:
            param_combos[key]["detection_times"].append(sim_result["years_to_detect"])

    # Build final results dataframe
    for key, data in param_combos.items():
        models_per_year, benchmarks_per_year, true_accel, noise_mult, frac_accelerate_models = key
        detected_fraction = np.mean(data["detections"])
        mean_detection_time = np.mean(data["detection_times"]) if data["detection_times"] else None

        results.append(
            {
                "models_per_year": models_per_year,
                "benchmarks_per_year": benchmarks_per_year,
                "true_accel": true_accel,
                "noise_multiplier": noise_mult,
                "frac_accelerate_models": frac_accelerate_models,
                "detected_fraction": detected_fraction,
                "mean_years_to_detect": mean_detection_time,
                "n_detected": sum(data["detections"]),
                "n_total": len(data["detections"]),
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
    num_models=CONFIG.fpr_num_models,
    num_benchmarks=CONFIG.fpr_num_benchmarks,
    cutoff_year=CONFIG.cutoff_year,
    observation_years=3.0,
    detection_threshold=CONFIG.detection_threshold,
    min_r2=CONFIG.min_r2,
    noise_multiplier=1.0,
    **kwargs,
):
    """
    Test false positive rate for a single configuration.

    This is a helper function used by test_false_positive_rate_sweep.
    """
    # Apply noise multiplier to base noise parameters from CONFIG
    error_std = CONFIG.base_error_std * noise_multiplier
    noise_std_model = CONFIG.base_noise_std_model * noise_multiplier
    noise_std_bench = CONFIG.base_noise_std_bench * noise_multiplier

    false_positives = 0
    detection_times = []

    # Calculate time range based on observation window
    time_range_start = cutoff_year - CONFIG.fpr_time_offset
    time_range_end = cutoff_year + observation_years

    for trial in range(n_trials):
        # Generate data with NO acceleration (true_acceleration=1)
        models_df, benchmarks_df, scores_df = generate_data(
            num_models=num_models,
            num_benchmarks=num_benchmarks,
            true_acceleration=1.0,  # NO ACCELERATION
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            cutoff_year=cutoff_year,
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
                detection_threshold=detection_threshold,
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
    num_models=CONFIG.fpr_num_models,
    num_benchmarks=CONFIG.fpr_num_benchmarks,
    cutoff_year=CONFIG.cutoff_year,
    detection_threshold=CONFIG.detection_threshold,
    min_r2=CONFIG.min_r2,
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
                detection_threshold=detection_threshold,
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
    num_models=CONFIG.fpr_num_models,
    num_benchmarks=CONFIG.fpr_num_benchmarks,
    cutoff_year=CONFIG.cutoff_year,
    observation_years=3.0,
    detection_threshold=CONFIG.detection_threshold,
    min_r2=CONFIG.min_r2,
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
    time_range_start = cutoff_year - CONFIG.fpr_time_offset
    time_range_end = cutoff_year + observation_years

    print(f"Testing false positive rate ({n_trials} trials, NO acceleration)...")
    print(
        f"  Observation window: {observation_years} years after cutoff ({cutoff_year:.1f} to {time_range_end:.1f})"
    )

    for trial in range(n_trials):
        # Generate data with NO acceleration (true_acceleration=1)
        models_df, benchmarks_df, scores_df = generate_data(
            num_models=num_models,
            num_benchmarks=num_benchmarks,
            true_acceleration=1.0,  # NO ACCELERATION
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            cutoff_year=cutoff_year,
            error_std=kwargs.get("error_std", CONFIG.base_error_std),
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
                detection_threshold=detection_threshold,
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
# SINGLE EXAMPLE RUN
# ============================================================================


def run_single_example(
    output_dir,
    models_per_year=30,
    benchmarks_per_year=10,
    true_acceleration=4.0,
    noise_multiplier=1.0,
    horizon_years=CONFIG.horizon_years,
    time_range_start=CONFIG.time_range_start,
    cutoff_year=CONFIG.cutoff_year,
    detection_threshold=CONFIG.detection_threshold,
    random_seed=42,
):
    """
    Run a single example configuration with detailed output.

    This demonstrates the acceleration detection process for a typical
    "normal" parameter configuration from the sweep.

    Parameters:
    -----------
    models_per_year : int
        Model release rate (default: 30 = middle of sweep range)
    benchmarks_per_year : int
        Benchmark release rate (default: 10 = middle of sweep range)
    true_acceleration : float
        True acceleration factor (default: 4.0 = middle of sweep range)
    noise_multiplier : float
        Noise level multiplier (default: 1.0 = standard noise)
    """
    print("\n" + "=" * 80)
    print("SINGLE EXAMPLE RUN: Acceleration Detection")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Models per year: {models_per_year}")
    print(f"  Benchmarks per year: {benchmarks_per_year}")
    print(f"  True acceleration factor: {true_acceleration}x")
    print(f"  Noise multiplier: {noise_multiplier}x")
    print(f"  Time range: {time_range_start} to {time_range_start + horizon_years}")
    print(f"  Cutoff year (acceleration starts): {cutoff_year}")
    print(f"  Detection threshold: {detection_threshold}x")
    print()

    # Calculate data generation parameters
    num_models = int(models_per_year * horizon_years)
    num_benchmarks = int(benchmarks_per_year * horizon_years)

    # Generate synthetic data (using CONFIG for base noise parameters)
    print("Generating synthetic data...")
    models_df, benchmarks_df, scores_df = generate_data(
        num_models=num_models,
        num_benchmarks=num_benchmarks,
        true_acceleration=true_acceleration,
        time_range_start=time_range_start,
        time_range_end=time_range_start + horizon_years,
        cutoff_year=cutoff_year,
        error_std=CONFIG.base_error_std * noise_multiplier,
        noise_std_model=CONFIG.base_noise_std_model * noise_multiplier,
        noise_std_bench=CONFIG.base_noise_std_bench * noise_multiplier,
        random_seed=random_seed,
    )

    print(f"  Generated {len(models_df)} models")
    print(f"  Generated {len(benchmarks_df)} benchmarks")
    print(f"  Generated {len(scores_df)} evaluation scores")
    print(f"  Models with acceleration: {models_df['accelerated'].sum()}")
    print()

    # Estimate capabilities
    print("Estimating model capabilities from benchmark scores...")
    df_est = estimated_capabilities(models_df, benchmarks_df, scores_df, verbose=True)
    print(f"  Estimated capabilities for {len(df_est)} models")
    print()

    # Run temporal detection
    print("Running temporal acceleration detection...")
    result = estimate_detection_for_single_trajectory(
        models_df,
        benchmarks_df,
        scores_df,
        cutoff_year=cutoff_year,
        detection_threshold=detection_threshold,
        min_r2=0.6,
        min_gap_years=0.0,
        scan_resolution=50,
        min_points_after=3,
        verbose=True,
    )

    # Display results
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)

    if result["detected"]:
        print(f"✓ ACCELERATION DETECTED")
        print(f"  Years to detection: {result['years_to_detect']:.2f}")
        print(f"  Detection time: {result['detection_time']:.2f}")
        print(f"  Breakpoint: {result['breakpoint']:.2f}")
        print(f"  Slope before breakpoint: {result['slope_before']:.4f}")
        print(f"  Slope after breakpoint: {result['slope_after']:.4f}")
        print(f"  Slope ratio: {result['ratio']:.2f}x")
        print(f"  R²: {result['r2']:.4f}")
    else:
        print("✗ NO ACCELERATION DETECTED")
        print(f"  True acceleration factor was {true_acceleration}x")
        print(f"  Detection threshold was {detection_threshold}x")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Create example subdirectory
    example_dir = output_dir / "example"
    example_dir.mkdir(parents=True, exist_ok=True)

    output_path = example_dir / "models.csv"
    models_df.to_csv(output_path, index=False)
    print(f"  Saved models: {output_path}")

    output_path = example_dir / "benchmarks.csv"
    benchmarks_df.to_csv(output_path, index=False)
    print(f"  Saved benchmarks: {output_path}")

    output_path = example_dir / "scores.csv"
    scores_df.to_csv(output_path, index=False)
    print(f"  Saved scores: {output_path}")

    output_path = example_dir / "estimated_capabilities.csv"
    df_est.to_csv(output_path, index=False)
    print(f"  Saved estimated capabilities: {output_path}")

    output_path = example_dir / "detection_result.csv"
    pd.DataFrame([result]).to_csv(output_path, index=False)
    print(f"  Saved detection result: {output_path}")

    print("\n" + "=" * 80)
    print("EXAMPLE RUN COMPLETE")
    print("=" * 80)
    print("\nYou can use the saved CSV files to create visualizations.")
    print(f"All outputs saved to: {example_dir}")
    print("=" * 80)

    return result


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
        "--example-only",
        action="store_true",
        help="Only run a single example configuration with detailed output",
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

    # ========================================================================
    # EXAMPLE MODE: Run single configuration with detailed output
    # ========================================================================

    if args.example_only:
        run_single_example(output_dir)
        return

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

        print("\nRunning parameter sweep...")
        models_per_year_list = [30]
        benchmarks_per_year_list = [10]
        true_accelerations = [2, 4, 8]
        noise_multipliers = [0.5, 1.0, 4.0]
        frac_accelerate_models_list = [0.25, 1.0]
        n_simulations = 5

        results_df = run_detection_sweep(
            models_per_year_list=models_per_year_list,
            benchmarks_per_year_list=benchmarks_per_year_list,
            true_accelerations=true_accelerations,
            noise_multipliers=noise_multipliers,
            frac_accelerate_models_list=frac_accelerate_models_list,
            n_simulations=n_simulations,
            # Using CONFIG defaults for horizon_years, cutoff_year, detection_threshold
        )

        # Save results
        sweep_dir = output_dir / "parameter_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        results_path = sweep_dir / "detection_sweep_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved results: {results_path}")

        # Display summary
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP RESULTS")
        print("=" * 80)
        print(
            results_df.sort_values(
                [
                    "frac_accelerate_models",
                    "true_accel",
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
    n_trials = 50

    fpr_results_df = test_false_positive_rate_sweep(
        noise_multipliers=noise_multipliers,
        observation_years_list=observation_years_list,
        n_trials=n_trials,
        # Using CONFIG defaults for num_models, num_benchmarks, cutoff_year, detection_threshold, min_r2
    )

    # Save results
    fpr_dir = output_dir / "false_positive"
    fpr_dir.mkdir(parents=True, exist_ok=True)
    fpr_results_path = fpr_dir / "false_positive_rate_sweep_results.csv"
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
        print(
            "  - Check parameter_sweep/detection_sweep_results.csv for detailed results"
        )
    print(
        "  - Check false_positive/false_positive_rate_sweep_results.csv for false positive analysis"
    )
    print("  - Adjust parameters based on false positive rate tolerance")
    print("=" * 80)


if __name__ == "__main__":
    main()
