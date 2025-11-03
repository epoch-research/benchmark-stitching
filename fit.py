from typing import Optional

import pandas as pd
import numpy as np
from data_loader import df_model
from scipy.optimize import least_squares

def fit_statistical_model(df, 
                         anchor_mode="benchmark",  # "benchmark", "model", or "global_constraints"
                         # Benchmark anchoring parameters
                         anchor_benchmark=None, 
                         anchor_difficulty=None, 
                         anchor_slope=1.0,
                         # Model anchoring parameters
                         anchor_model1=None,
                         anchor_model1_capability=None,
                         anchor_model2=None,
                         anchor_model2_capability=None,
                         # Other parameters
                         slope_init=1.0,
                         regularization_strength=0.1,  # NEW: Add L2 regularization
                         df_model=df_model,
                         # Standard error / CI options
                         compute_standard_errors: bool = True,
                         ci_level: float = 0.90,
                         cov_regularization: float = 1e-6,
                         performance_clip_eps: float = 1e-3,
                         bootstrap_samples: int = 100,
                         bootstrap_seed: Optional[int] = 12345):
    """
    Fit a statistical model with multiple anchoring modes and L2 regularization.
    
    1. 'benchmark' mode: Anchor on a specific benchmark's difficulty and slope
       - anchor_benchmark: name of the benchmark to anchor
       - anchor_difficulty: fixed difficulty value for the anchor benchmark
       - anchor_slope: fixed slope value for the anchor benchmark
    
    2. 'model' mode: Anchor on two specific models' capabilities
       - anchor_model1: name of the first model to anchor
       - anchor_model1_capability: fixed capability value for the first model
       - anchor_model2: name of the second model to anchor
       - anchor_model2_capability: fixed capability value for the second model
    
    regularization_strength: L2 regularization strength (0 = no regularization, typical: 0.01-0.5)
    performance_clip_eps: clamp raw benchmark performances to [eps, 1-eps] to avoid 0/1 extremes
    compute_standard_errors: if True, estimate uncertainty via bootstrap resampling
    bootstrap_samples: number of bootstrap resamples to draw for confidence intervals (default 200)
    bootstrap_seed: random seed controlling bootstrap resampling
    cov_regularization: retained for backwards compatibility; ignored by bootstrap workflow

    Additional anchoring mode:
    3. 'global_constraints' mode: Enforce that benchmark difficulties sum to 0 and
       the geometric mean of benchmark slopes equals 1. No explicit anchor benchmark/model.

    Bootstrap-derived uncertainty metrics are attached to the returned DataFrames
    under columns prefixed with ``bootstrap_`` for easier interpretation.
    """
    # Defensive copy so we can clip without mutating caller data
    df = df.copy()

    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    bootstrap_samples = int(bootstrap_samples)

    # ------------------------------------------------------------
    # 1)  Mappings & data arrays
    # ------------------------------------------------------------
    # Check for invalid data
    if df["performance"].isna().any():
        raise ValueError("Performance data contains NaN values")
    if not np.all(np.isfinite(df["performance"].values)):
        raise ValueError("Performance data contains infinite values")
    if (df["performance"] < 0).any() or (df["performance"] > 1).any():
        print(f"Warning: Performance scores outside [0,1] range. Min: {df['performance'].min()}, Max: {df['performance'].max()}")
    # Clip to avoid exactly 0/1 which cause degenerate slopes / difficulties
    if performance_clip_eps > 0:
        eps = performance_clip_eps
        df["performance"] = df["performance"].clip(eps, 1 - eps)
    
    valid_model_ids   = df["model_id"].unique()
    benchmark_ids     = df["benchmark_id"].unique()
    
    model_id_to_idx   = {m_id: i for i, m_id in enumerate(valid_model_ids)}
    bench_id_to_idx   = {b_id: i for i, b_id in enumerate(benchmark_ids)}
    
    num_models        = len(valid_model_ids)
    num_benchmarks    = len(benchmark_ids)
    
    model_idx_data    = np.array([model_id_to_idx[m] for m in df["model_id"]])
    bench_idx_data    = np.array([bench_id_to_idx[b] for b in df["benchmark_id"]])
    observed_scores   = df["performance"].values
    
    # ------------------------------------------------------------
    # 2)  Set up anchoring based on mode
    # ------------------------------------------------------------
    if anchor_mode == "benchmark":
        # Original benchmark anchoring logic
        if anchor_benchmark is None or anchor_difficulty is None:
            raise ValueError("For benchmark mode, must provide anchor_benchmark and anchor_difficulty")
            
        try:
            anchor_bench_id = df.loc[
                df["benchmark"] == anchor_benchmark, "benchmark_id"
            ].iloc[0]
        except IndexError:
            raise ValueError(f"Benchmark named '{anchor_benchmark}' not found in df")
        
        anchor_bench_idx = bench_id_to_idx[anchor_bench_id]
        anchor_model_indices = None
        
    elif anchor_mode == "model":
        # New model anchoring logic
        if any(x is None for x in [anchor_model1, anchor_model1_capability, 
                                   anchor_model2, anchor_model2_capability]):
            raise ValueError("For model mode, must provide all model anchoring parameters")
        
        # Find model IDs for the anchor models
        try:
            anchor_model1_id = df.loc[df["model"] == anchor_model1, "model_id"].iloc[0]
            anchor_model2_id = df.loc[df["model"] == anchor_model2, "model_id"].iloc[0]
        except IndexError as e:
            raise ValueError(f"One of the anchor models not found in df") from e
        
        anchor_model1_idx = model_id_to_idx[anchor_model1_id]
        anchor_model2_idx = model_id_to_idx[anchor_model2_id]
        
        if anchor_model1_idx == anchor_model2_idx:
            raise ValueError("Must specify two different models for anchoring")
        
        # Store indices in order (smaller first) for consistent parameter arrangement
        anchor_model_indices = tuple(sorted([anchor_model1_idx, anchor_model2_idx]))
        anchor_bench_idx = None
        
    elif anchor_mode == "global_constraints":
        anchor_bench_idx = None
        anchor_model_indices = None
    else:
        raise ValueError(f"anchor_mode must be 'benchmark', 'model', or 'global_constraints', got '{anchor_mode}'")
    
    # ------------------------------------------------------------
    # 3)  Helpers
    # ------------------------------------------------------------
    def logistic(x: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))
    
    def split_params(params: np.ndarray):
        """
        Break the flat parameter vector into C, D and α based on anchor mode.
        """
        if anchor_mode == "benchmark":
            # Original logic: all C and D are free, one α is fixed
            C = params[:num_models]
            D = params[num_models : num_models + num_benchmarks]
            alpha_free = params[num_models + num_benchmarks :]
            alpha = np.insert(alpha_free, anchor_bench_idx, anchor_slope)
            
        elif anchor_mode == "model":
            # New logic: two C values are fixed, all D and α are free
            C_free = params[:num_models - 2]
            D = params[num_models - 2 : num_models - 2 + num_benchmarks]
            alpha = params[num_models - 2 + num_benchmarks :]
            
            # Reconstruct full C vector with fixed values
            C = np.zeros(num_models)
            free_idx = 0
            for i in range(num_models):
                if i == anchor_model_indices[0]:
                    # Use the appropriate capability based on original order
                    if anchor_model1_idx < anchor_model2_idx:
                        C[i] = anchor_model1_capability
                    else:
                        C[i] = anchor_model2_capability
                elif i == anchor_model_indices[1]:
                    # Use the appropriate capability based on original order
                    if anchor_model1_idx < anchor_model2_idx:
                        C[i] = anchor_model2_capability
                    else:
                        C[i] = anchor_model1_capability
                else:
                    C[i] = C_free[free_idx]
                    free_idx += 1
        elif anchor_mode == "global_constraints":
            C = params[:num_models]
            D_raw = params[num_models : num_models + num_benchmarks]
            log_alpha_raw = params[num_models + num_benchmarks :]

            # Enforce sum(D) = 0 via centering
            D_centered = D_raw - np.mean(D_raw)
            # Enforce geometric mean(alpha) = 1 via centered log-slopes
            log_alpha_centered = log_alpha_raw - np.mean(log_alpha_raw)
            alpha = np.exp(log_alpha_centered)
            return C, D_centered, alpha

        return C, D, alpha

    def finalize_parameters(theta: np.ndarray):
        """Convert optimizer output into anchored parameter vectors."""
        C_hat, D_hat, alpha_hat = split_params(theta)
        if anchor_mode == "benchmark":
            shift = D_hat[anchor_bench_idx] - anchor_difficulty
            C_hat = C_hat - shift
            D_hat = D_hat - shift
        return C_hat, D_hat, alpha_hat
    
    def residuals(params, model_idx, bench_idx, y):
        C, D, alpha = split_params(params)
        preds = logistic(alpha[bench_idx] * (C[model_idx] - D[bench_idx]))
        residuals = preds - y
        
        # Add L2 regularization
        if regularization_strength > 0:
            # Calculate regularization penalty on free parameters only
            if anchor_mode == "benchmark":
                # All C and D are free, one α is fixed
                # Add small epsilon to prevent division by zero
                reg_term = regularization_strength * (
                    np.sum(C**2) + 
                    np.sum(D**2) + 
                    np.sum(alpha[alpha != anchor_slope]**2)
                ) / (num_models + num_benchmarks + num_benchmarks - 1)
            elif anchor_mode == "model":
                # Two C values are fixed, all D and α are free
                free_C_mask = np.ones(num_models, dtype=bool)
                free_C_mask[list(anchor_model_indices)] = False
                reg_term = regularization_strength * (
                    np.sum(C[free_C_mask]**2) + 
                    np.sum(D**2) + 
                    np.sum(alpha**2)
                ) / (num_models - 2 + num_benchmarks + num_benchmarks)
            else:  # global_constraints mode
                # Regularize centered parameters to discourage drift
                reg_term = regularization_strength * (
                    np.sum(C**2) +
                    np.sum(D**2) +
                    np.sum(np.log(alpha)**2)
                ) / (num_models + num_benchmarks + num_benchmarks)
            
            # Add regularization as additional residuals (simpler approach)
            # Just append a single regularization term scaled appropriately
            reg_penalty = np.sqrt(reg_term) if reg_term > 0 else 0
            return np.append(residuals, reg_penalty)
        
        return residuals
    
    # ------------------------------------------------------------
    # 4)  Initial guesses (small random values to avoid numerical issues)
    # ------------------------------------------------------------
    np.random.seed(42)  # For reproducibility
    
    if anchor_mode == "benchmark":
        # Original: C and D free, one α fixed
        initial_C     = np.random.randn(num_models) * 0.1
        initial_D     = np.random.randn(num_benchmarks) * 0.1
        initial_alpha = np.full(num_benchmarks - 1, slope_init)
        initial_theta = np.concatenate([initial_C, initial_D, initial_alpha])
        
    elif anchor_mode == "model":
        # New: two C fixed, all D and α free
        initial_C_free = np.random.randn(num_models - 2) * 0.1
        initial_D      = np.random.randn(num_benchmarks) * 0.1
        initial_alpha  = np.full(num_benchmarks, slope_init)
        initial_theta  = np.concatenate([initial_C_free, initial_D, initial_alpha])
    elif anchor_mode == "global_constraints":
        initial_C = np.random.randn(num_models) * 0.1
        initial_D_raw = np.random.randn(num_benchmarks) * 0.1
        # Work in log-space for slopes to guarantee positivity after exponentiation
        initial_log_alpha_raw = np.full(num_benchmarks, np.log(slope_init))
        initial_theta = np.concatenate([initial_C, initial_D_raw, initial_log_alpha_raw])
    
    # ------------------------------------------------------------
    # 5)  Set bounds to prevent extreme values
    # ------------------------------------------------------------
    if anchor_mode == "benchmark":
        # Bounds: C in [-10, 10], D in [-10, 10], alpha in [0.1, 10]
        lower_bounds = np.concatenate([
            np.full(num_models, -10),
            np.full(num_benchmarks, -10),
            np.full(num_benchmarks - 1, 0.1)
        ])
        upper_bounds = np.concatenate([
            np.full(num_models, 10),
            np.full(num_benchmarks, 10),
            np.full(num_benchmarks - 1, 10)
        ])
    elif anchor_mode == "model":
        lower_bounds = np.concatenate([
            np.full(num_models - 2, -10),
            np.full(num_benchmarks, -10),
            np.full(num_benchmarks, 0.1)
        ])
        upper_bounds = np.concatenate([
            np.full(num_models - 2, 10),
            np.full(num_benchmarks, 10),
            np.full(num_benchmarks, 10)
        ])
    else:  # global_constraints mode
        lower_bounds = np.concatenate([
            np.full(num_models, -10),
            np.full(num_benchmarks, -10),
            np.full(num_benchmarks, -5)
        ])
        upper_bounds = np.concatenate([
            np.full(num_models, 10),
            np.full(num_benchmarks, 10),
            np.full(num_benchmarks, 5)
        ])
    
    # ------------------------------------------------------------
    # 6)  Fit with bounds
    # ------------------------------------------------------------
    result = least_squares(
        residuals,
        initial_theta,
        args=(model_idx_data, bench_idx_data, observed_scores),
        bounds=(lower_bounds, upper_bounds),  # Add bounds to prevent extreme values
        method="trf",
        verbose=1
    )
    
    # ------------------------------------------------------------
    # 7)  Recover full parameter vectors
    # ------------------------------------------------------------
    theta_hat = result.x
    C_hat, D_hat, alpha_hat = finalize_parameters(theta_hat)

    # ------------------------------------------------------------
    # 7.5)  Optional: bootstrap confidence intervals
    # ------------------------------------------------------------
    se_C = se_D = se_alpha = None
    ci_C_low = ci_C_high = None
    ci_D_low = ci_D_high = None
    ci_alpha_low = ci_alpha_high = None
    bootstrap_successes = 0

    if compute_standard_errors and bootstrap_samples and bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        C_samples = []
        D_samples = []
        alpha_samples = []

        for _ in range(bootstrap_samples):
            sample_idx = rng.integers(0, observed_scores.shape[0], size=observed_scores.shape[0])
            boot_scores = observed_scores[sample_idx]
            boot_model_idx = model_idx_data[sample_idx]
            boot_bench_idx = bench_idx_data[sample_idx]

            try:
                boot_res = least_squares(
                    residuals,
                    theta_hat.copy(),
                    args=(boot_model_idx, boot_bench_idx, boot_scores),
                    bounds=(lower_bounds, upper_bounds),
                    method="trf",
                    verbose=0,
                )
            except Exception:
                continue

            if not boot_res.success:
                continue

            boot_C, boot_D, boot_alpha = finalize_parameters(boot_res.x)
            C_samples.append(boot_C)
            D_samples.append(boot_D)
            alpha_samples.append(boot_alpha)

        if C_samples:
            C_samples_arr = np.vstack(C_samples)
            D_samples_arr = np.vstack(D_samples)
            alpha_samples_arr = np.vstack(alpha_samples)
            bootstrap_successes = C_samples_arr.shape[0]

            if bootstrap_successes < bootstrap_samples:
                print(
                    f"Warning: Bootstrap convergence in {bootstrap_successes}/{bootstrap_samples} resamples; using converged fits only."
                )

            if bootstrap_successes > 1:
                se_C = np.std(C_samples_arr, axis=0, ddof=1)
                se_D = np.std(D_samples_arr, axis=0, ddof=1)
                se_alpha = np.std(alpha_samples_arr, axis=0, ddof=1)
            else:
                se_C = np.full(num_models, np.nan)
                se_D = np.full(num_benchmarks, np.nan)
                se_alpha = np.full(num_benchmarks, np.nan)

            if ci_level is not None and 0 < ci_level < 1:
                tail = (1.0 - ci_level) / 2.0
                ci_C_low = np.quantile(C_samples_arr, tail, axis=0)
                ci_C_high = np.quantile(C_samples_arr, 1.0 - tail, axis=0)
                ci_D_low = np.quantile(D_samples_arr, tail, axis=0)
                ci_D_high = np.quantile(D_samples_arr, 1.0 - tail, axis=0)
                ci_alpha_low = np.quantile(alpha_samples_arr, tail, axis=0)
                ci_alpha_high = np.quantile(alpha_samples_arr, 1.0 - tail, axis=0)
        else:
            print("Warning: All bootstrap fits failed; skipping confidence interval computation.")
    
    # ------------------------------------------------------------
    # 8)  Pack tidy DataFrames for inspection / downstream use
    # ------------------------------------------------------------
    # ---- Model capabilities ----
    id_to_name = df.drop_duplicates("model_id").set_index("model_id")["model"].to_dict()
    
    model_cap_df = (
        pd.DataFrame(
            {
                "model_id": valid_model_ids,
                "estimated_capability": C_hat,
            }
        )
        .assign(model=lambda d: d["model_id"].map(id_to_name))
    )
    
    # Add anchoring information
    if anchor_mode == "model":
        model_cap_df["is_anchor"] = model_cap_df["model"].isin([anchor_model1, anchor_model2])
    else:
        model_cap_df["is_anchor"] = False
    
    if df_model is not None:
        model_cap_df = model_cap_df.merge(df_model, on="model", how="left")

    # Attach bootstrap summaries if computed
    if se_C is not None:
        model_cap_df["bootstrap_std_capability"] = se_C
    if ci_C_low is not None and ci_C_high is not None:
        model_cap_df["bootstrap_ci_lower_capability"] = ci_C_low
        model_cap_df["bootstrap_ci_upper_capability"] = ci_C_high
        if ci_level is not None:
            model_cap_df["bootstrap_ci_level"] = ci_level

    model_capabilities_df = model_cap_df.sort_values(
        "estimated_capability", ascending=False
    )
    
    # ---- Benchmark parameters ----
    benchmark_params_df = (
        pd.DataFrame(
            {
                "benchmark_id": benchmark_ids,
                "estimated_difficulty": D_hat,
                "estimated_slope": alpha_hat,
            }
        )
        .assign(benchmark_name=lambda d: d["benchmark_id"].map(dict(zip(
            df["benchmark_id"], df["benchmark"]
        ))))
        .sort_values("estimated_difficulty")
    )
    
    # Add anchoring information
    if anchor_mode == "benchmark":
        benchmark_params_df["is_anchor"] = benchmark_params_df["benchmark_name"] == anchor_benchmark
    else:
        benchmark_params_df["is_anchor"] = False
    
    bench_dates = (
        df
        .drop_duplicates("benchmark_id")[["benchmark_id", "benchmark_release_date"]]
    )
    benchmark_params_df = (
        benchmark_params_df
        .merge(bench_dates, on="benchmark_id", how="left")
    )

    if se_D is not None:
        benchmark_params_df["bootstrap_std_difficulty"] = se_D
    if se_alpha is not None:
        benchmark_params_df["bootstrap_std_slope"] = se_alpha
    if ci_D_low is not None and ci_D_high is not None:
        benchmark_params_df["bootstrap_ci_lower_difficulty"] = ci_D_low
        benchmark_params_df["bootstrap_ci_upper_difficulty"] = ci_D_high
        if ci_level is not None:
            benchmark_params_df["bootstrap_ci_level"] = ci_level
    if ci_alpha_low is not None and ci_alpha_high is not None:
        benchmark_params_df["bootstrap_ci_lower_slope"] = ci_alpha_low
        benchmark_params_df["bootstrap_ci_upper_slope"] = ci_alpha_high
    
    return df, model_capabilities_df, benchmark_params_df

# def fit_statistical_model(df, 
#                          anchor_mode="benchmark",  # "benchmark" or "model"
#                          # Benchmark anchoring parameters
#                          anchor_benchmark=None, 
#                          anchor_difficulty=None, 
#                          anchor_slope=1.0,
#                          # Model anchoring parameters
#                          anchor_model1=None,
#                          anchor_model1_capability=None,
#                          anchor_model2=None,
#                          anchor_model2_capability=None,
#                          # Other parameters
#                          slope_init=1.0, # MODIFIED: Changed default from 0.05 to 1.0
#                          df_model=df_model): # Assuming df_model is provided or defined elsewhere
#     """
#     Fit a statistical model with two anchoring modes:
    
#     1. 'benchmark' mode: Anchor on a specific benchmark's difficulty and slope
#        - anchor_benchmark: name of the benchmark to anchor
#        - anchor_difficulty: fixed difficulty value for the anchor benchmark
#        - anchor_slope: fixed slope value for the anchor benchmark
    
#     2. 'model' mode: Anchor on two specific models' capabilities
#        - anchor_model1: name of the first model to anchor
#        - anchor_model1_capability: fixed capability value for the first model
#        - anchor_model2: name of the second model to anchor
#        - anchor_model2_capability: fixed capability value for the second model
#     """
    
#     # ------------------------------------------------------------
#     # 1)  Mappings & data arrays
#     # ------------------------------------------------------------
#     valid_model_ids   = df["model_id"].unique()
#     benchmark_ids     = df["benchmark_id"].unique()
    
#     model_id_to_idx   = {m_id: i for i, m_id in enumerate(valid_model_ids)}
#     bench_id_to_idx   = {b_id: i for i, b_id in enumerate(benchmark_ids)}
    
#     num_models        = len(valid_model_ids)
#     num_benchmarks    = len(benchmark_ids)
    
#     model_idx_data    = np.array([model_id_to_idx[m] for m in df["model_id"]])
#     bench_idx_data    = np.array([bench_id_to_idx[b] for b in df["benchmark_id"]])
#     observed_scores   = df["performance"].values
    
#     # ------------------------------------------------------------
#     # 2)  Set up anchoring based on mode
#     # ------------------------------------------------------------
#     if anchor_mode == "benchmark":
#         # Original benchmark anchoring logic
#         if anchor_benchmark is None or anchor_difficulty is None:
#             raise ValueError("For benchmark mode, must provide anchor_benchmark and anchor_difficulty")
            
#         try:
#             anchor_bench_id = df.loc[
#                 df["benchmark"] == anchor_benchmark, "benchmark_id"
#             ].iloc[0]
#         except IndexError:
#             raise ValueError(f"Benchmark named '{anchor_benchmark}' not found in df")
        
#         anchor_bench_idx = bench_id_to_idx[anchor_bench_id]
#         anchor_model_indices = None
        
#     elif anchor_mode == "model":
#         # New model anchoring logic
#         if any(x is None for x in [anchor_model1, anchor_model1_capability, 
#                                    anchor_model2, anchor_model2_capability]):
#             raise ValueError("For model mode, must provide all model anchoring parameters")
        
#         # Find model IDs for the anchor models
#         try:
#             anchor_model1_id = df.loc[df["model"] == anchor_model1, "model_id"].iloc[0]
#             anchor_model2_id = df.loc[df["model"] == anchor_model2, "model_id"].iloc[0]
#         except IndexError as e:
#             raise ValueError(f"One of the anchor models not found in df") from e
        
#         anchor_model1_idx = model_id_to_idx[anchor_model1_id]
#         anchor_model2_idx = model_id_to_idx[anchor_model2_id]
        
#         if anchor_model1_idx == anchor_model2_idx:
#             raise ValueError("Must specify two different models for anchoring")
        
#         # Store indices in order (smaller first) for consistent parameter arrangement
#         anchor_model_indices = tuple(sorted([anchor_model1_idx, anchor_model2_idx]))
#         anchor_bench_idx = None
        
#     else:
#         raise ValueError(f"anchor_mode must be 'benchmark' or 'model', got '{anchor_mode}'")
    
#     # ------------------------------------------------------------
#     # 3)  Helpers
#     # ------------------------------------------------------------
#     def logistic(x: np.ndarray) -> np.ndarray:
#         return 1.0 / (1.0 + np.exp(-x))
    
#     def split_params(params: np.ndarray):
#         """
#         Break the flat parameter vector into C, D and α based on anchor mode.
#         """
#         if anchor_mode == "benchmark":
#             # Original logic: all C and D are free, one α is fixed
#             C = params[:num_models]
#             D = params[num_models : num_models + num_benchmarks]
#             alpha_free = params[num_models + num_benchmarks :]
#             alpha = np.insert(alpha_free, anchor_bench_idx, anchor_slope)
            
#         elif anchor_mode == "model":
#             # New logic: two C values are fixed, all D and α are free
#             C_free = params[:num_models - 2]
#             D = params[num_models - 2 : num_models - 2 + num_benchmarks]
#             alpha = params[num_models - 2 + num_benchmarks :]
            
#             # Reconstruct full C vector with fixed values
#             C = np.zeros(num_models)
#             free_idx = 0
#             for i in range(num_models):
#                 if i == anchor_model_indices[0]:
#                     # Use the appropriate capability based on original order
#                     if anchor_model1_idx < anchor_model2_idx:
#                         C[i] = anchor_model1_capability
#                     else:
#                         C[i] = anchor_model2_capability
#                 elif i == anchor_model_indices[1]:
#                     # Use the appropriate capability based on original order
#                     if anchor_model1_idx < anchor_model2_idx:
#                         C[i] = anchor_model2_capability
#                     else:
#                         C[i] = anchor_model1_capability
#                 else:
#                     C[i] = C_free[free_idx]
#                     free_idx += 1
        
#         return C, D, alpha
    
#     def residuals(params, model_idx, bench_idx, y):
#         C, D, alpha = split_params(params)
#         preds = logistic(alpha[bench_idx] * (C[model_idx] - D[bench_idx]))
#         return preds - y
    
#     # ------------------------------------------------------------
#     # 4)  Initial guesses
#     # ------------------------------------------------------------
#     if anchor_mode == "benchmark":
#         # Original: C and D free, one α fixed
#         initial_C     = np.zeros(num_models)
#         initial_D     = np.zeros(num_benchmarks)
#         initial_alpha = np.full(num_benchmarks - 1, slope_init) # Will now use 1.0 by default
#         initial_theta = np.concatenate([initial_C, initial_D, initial_alpha])
        
#     elif anchor_mode == "model":
#         # New: two C fixed, all D and α free
#         initial_C_free = np.zeros(num_models - 2)
#         initial_D      = np.zeros(num_benchmarks)
#         initial_alpha  = np.full(num_benchmarks, slope_init) # Will now use 1.0 by default
#         initial_theta  = np.concatenate([initial_C_free, initial_D, initial_alpha])
    
#     # ------------------------------------------------------------
#     # 5)  Fit
#     # ------------------------------------------------------------
#     result = least_squares(
#         residuals,
#         initial_theta,
#         args=(model_idx_data, bench_idx_data, observed_scores),
#         method="trf",
#         verbose=1
#     )
    
#     # ------------------------------------------------------------
#     # 6)  Recover full parameter vectors
#     # ------------------------------------------------------------
#     theta_hat = result.x
    
#     if anchor_mode == "benchmark":
#         # Original extraction logic
#         C_hat          = theta_hat[:num_models]
#         D_hat          = theta_hat[num_models : num_models + num_benchmarks]
#         alpha_free_hat = theta_hat[num_models + num_benchmarks :]
#         alpha_hat      = np.insert(alpha_free_hat, anchor_bench_idx, anchor_slope)
        
#         # Shift to match anchor difficulty
#         shift = D_hat[anchor_bench_idx] - anchor_difficulty
#         C_hat -= shift
#         D_hat -= shift
        
#     elif anchor_mode == "model":
#         # New extraction logic
#         C_hat, D_hat, alpha_hat = split_params(theta_hat)
#         # No shifting needed - model capabilities are already anchored
    
#     # ------------------------------------------------------------
#     # 7)  Pack tidy DataFrames for inspection / downstream use
#     # ------------------------------------------------------------
#     # ---- Model capabilities ----
#     id_to_name = df.drop_duplicates("model_id").set_index("model_id")["model"].to_dict()
    
#     model_cap_df = (
#         pd.DataFrame(
#             {
#                 "model_id": valid_model_ids,
#                 "estimated_capability": C_hat,
#             }
#         )
#         .assign(model=lambda d: d["model_id"].map(id_to_name))
#     )
    
#     # Add anchoring information
#     if anchor_mode == "model":
#         model_cap_df["is_anchor"] = model_cap_df["model"].isin([anchor_model1, anchor_model2])
#     else:
#         model_cap_df["is_anchor"] = False
    
#     if df_model is not None:
#         model_cap_df = model_cap_df.merge(df_model, on="model", how="left")

#     model_capabilities_df = model_cap_df.sort_values(
#         "estimated_capability", ascending=False
#     )
    
#     # ---- Benchmark parameters ----
#     benchmark_params_df = (
#         pd.DataFrame(
#             {
#                 "benchmark_id": benchmark_ids,
#                 "estimated_difficulty": D_hat,
#                 "estimated_slope": alpha_hat,
#             }
#         )
#         .assign(benchmark_name=lambda d: d["benchmark_id"].map(dict(zip(
#             df["benchmark_id"], df["benchmark"]
#         ))))
#         .sort_values("estimated_difficulty")
#     )
    
#     # Add anchoring information
#     if anchor_mode == "benchmark":
#         benchmark_params_df["is_anchor"] = benchmark_params_df["benchmark_name"] == anchor_benchmark
#     else:
#         benchmark_params_df["is_anchor"] = False
    
#     bench_dates = (
#         df
#         .drop_duplicates("benchmark_id")[["benchmark_id", "benchmark_release_date"]]
#     )
#     benchmark_params_df = (
#         benchmark_params_df
#         .merge(bench_dates, on="benchmark_id", how="left")
#     )
    
#     return df, model_capabilities_df, benchmark_params_df

# def fit_statistical_model(df, anchor_benchmark, anchor_difficulty, anchor_slope, slope_init=0.05, df_model=df_model):
#   # ------------------------------------------------------------
#   # 1)  Mappings & data arrays
#   # ------------------------------------------------------------
# #   print("hello world")
#   valid_model_ids   = df["model_id"].unique()
#   benchmark_ids     = df["benchmark_id"].unique()

#   model_id_to_idx   = {m_id: i for i, m_id in enumerate(valid_model_ids)}
#   bench_id_to_idx   = {b_id: i for i, b_id in enumerate(benchmark_ids)}

#   num_models        = len(valid_model_ids)
#   num_benchmarks    = len(benchmark_ids)

#   model_idx_data    = np.array([model_id_to_idx[m] for m in df["model_id"]])
#   bench_idx_data    = np.array([bench_id_to_idx[b] for b in df["benchmark_id"]])
#   observed_scores   = df["performance"].values

#   # ------------------------------------------------------------
#   # 2)  Anchor benchmark
#   # ------------------------------------------------------------
#   try:
#       anchor_bench_id = df.loc[
#           df["benchmark"] == anchor_benchmark, "benchmark_id"
#       ].iloc[0]
#   except IndexError:
#       raise ValueError(f"Benchmark named “{anchor_benchmark}” not found in df")

#   anchor_idx = bench_id_to_idx[anchor_bench_id]      # 0-based position

#   # ------------------------------------------------------------
#   # 3)  Helpers
#   # ------------------------------------------------------------
#   def logistic(x: np.ndarray) -> np.ndarray:
#       return 1.0 / (1.0 + np.exp(-x))


#   def split_params(params: np.ndarray):
#       """
#       Break the flat parameter vector into C, D and full-length α,
#       with α_anchor hard-set to 1.
#       """
#       C = params[:num_models]
#       D = params[num_models : num_models + num_benchmarks]
#       alpha_free = params[num_models + num_benchmarks :]          # length = num_benchmarks − 1
#     #   alpha = np.insert(alpha_free, anchor_idx, 1.0)              # put the fixed 1 back in
#       alpha = np.insert(alpha_free, anchor_idx, anchor_slope)
#       return C, D, alpha


#   def residuals(params, model_idx, bench_idx, y):
#       C, D, alpha = split_params(params)
#       preds = logistic(alpha[bench_idx] * (C[model_idx] - D[bench_idx]))
#       return preds - y

#   # ------------------------------------------------------------
#   # 4)  Initial guesses  (note α vector is *one element shorter*)
#   # ------------------------------------------------------------
#   initial_C     = np.zeros(num_models)
#   initial_D     = np.zeros(num_benchmarks)
#   initial_alpha = np.full(num_benchmarks - 1, slope_init)
#   initial_theta = np.concatenate([initial_C, initial_D, initial_alpha])

#   # ------------------------------------------------------------
#   # 5)  Fit
#   # ------------------------------------------------------------
#   result = least_squares(
#       residuals,
#       initial_theta,
#       args=(model_idx_data, bench_idx_data, observed_scores),
#       method="trf",      # default; listed here for clarity
#       verbose=1
#   )

#   # ------------------------------------------------------------
#   # 6)  Recover full parameter vectors
#   # ------------------------------------------------------------
#   theta_hat      = result.x
#   C_hat          = theta_hat[:num_models]
#   D_hat          = theta_hat[num_models : num_models + num_benchmarks]
#   alpha_free_hat = theta_hat[num_models + num_benchmarks :]
#   alpha_hat      = np.insert(alpha_free_hat, anchor_idx, 1.0)
#   shift = D_hat[anchor_idx] - anchor_difficulty
#   C_hat -= shift
#   D_hat -= shift

#   # ------------------------------------------------------------
#   # 7)  Pack tidy DataFrames for inspection / downstream use
#   # ------------------------------------------------------------
#   # ---- Model capabilities ----
#   id_to_name = df.drop_duplicates("model_id").set_index("model_id")["model"].to_dict()

#   model_cap_df = (
#       pd.DataFrame(
#           {
#               "model_id": valid_model_ids,
#               "estimated_capability": C_hat,
#           }
#       )
#       .assign(model=lambda d: d["model_id"].map(id_to_name))
#   )
#   model_cap_df = model_cap_df.merge(df_model, on="model", how="left")
#   model_capabilities_df = model_cap_df.sort_values(
#       "estimated_capability", ascending=False
#   )

#   # ---- Benchmark parameters ----
#   benchmark_params_df = (
#       pd.DataFrame(
#           {
#               "benchmark_id": benchmark_ids,
#               "estimated_difficulty": D_hat,
#               "estimated_slope": alpha_hat,
#           }
#       )
#       .assign(benchmark_name=lambda d: d["benchmark_id"].map(dict(zip(
#           df["benchmark_id"], df["benchmark"]
#       ))))
#       .sort_values("estimated_difficulty")
#   )
#   bench_dates = (
#     df
#     .drop_duplicates("benchmark_id")[["benchmark_id", "benchmark_release_date"]]
#   )
#   benchmark_params_df = (
#     benchmark_params_df
#     .merge(bench_dates, on="benchmark_id", how="left")
#   )

# #   print(benchmark_params_df)
# #   print(bench_dates)

#   return df, model_capabilities_df, benchmark_params_df
