import pandas as pd
import numpy as np
from data_loader import df_model
from scipy.optimize import least_squares

def fit_statistical_model(df, 
                         anchor_mode="benchmark",  # "benchmark" or "model"
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
                         ci_level: float = 0.90):
    """
    Fit a statistical model with two anchoring modes and L2 regularization.
    
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
    """
    
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
        # Clip to valid range
        df = df.copy()
        df["performance"] = df["performance"].clip(0.001, 0.999)
    
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
        
    else:
        raise ValueError(f"anchor_mode must be 'benchmark' or 'model', got '{anchor_mode}'")
    
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
        
        return C, D, alpha
    
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
            else:  # model mode
                # Two C values are fixed, all D and α are free
                free_C_mask = np.ones(num_models, dtype=bool)
                free_C_mask[list(anchor_model_indices)] = False
                reg_term = regularization_strength * (
                    np.sum(C[free_C_mask]**2) + 
                    np.sum(D**2) + 
                    np.sum(alpha**2)
                ) / (num_models - 2 + num_benchmarks + num_benchmarks)
            
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
    else:  # model mode
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
    
    if anchor_mode == "benchmark":
        # Original extraction logic
        C_hat          = theta_hat[:num_models]
        D_hat          = theta_hat[num_models : num_models + num_benchmarks]
        alpha_free_hat = theta_hat[num_models + num_benchmarks :]
        alpha_hat      = np.insert(alpha_free_hat, anchor_bench_idx, anchor_slope)
        
        # Shift to match anchor difficulty
        shift = D_hat[anchor_bench_idx] - anchor_difficulty
        C_hat -= shift
        D_hat -= shift
        
    elif anchor_mode == "model":
        # New extraction logic
        C_hat, D_hat, alpha_hat = split_params(theta_hat)
        # No shifting needed - model capabilities are already anchored

    # ------------------------------------------------------------
    # 7.5)  Optional: standard errors and CIs from Jacobian
    # ------------------------------------------------------------
    se_C = se_D = se_alpha = None
    if compute_standard_errors and result.jac is not None:
        try:
            # Use only the data residuals (exclude the single appended regularization residual)
            n_obs = observed_scores.shape[0]
            J_full = result.jac
            J_data = J_full[:n_obs, :]
            # Residual sum-of-squares from data residuals
            res_data = result.fun[:n_obs]
            sse = float(np.sum(res_data**2))
            p = theta_hat.size
            dof = max(n_obs - p, 1)
            sigma2 = sse / dof
            JTJ = J_data.T @ J_data
            try:
                cov_theta = sigma2 * np.linalg.inv(JTJ)
            except np.linalg.LinAlgError:
                cov_theta = sigma2 * np.linalg.pinv(JTJ)

            # Helper accessors
            def var_at(idx):
                return float(np.clip(cov_theta[idx, idx], 0, np.inf))

            def cov_at(i, j):
                return float(cov_theta[i, j])

            if anchor_mode == "benchmark":
                # Parameterization in theta: [C (M), D (B), alpha_free (B-1)]
                idx_C_start = 0
                idx_D_start = num_models
                # idx_alpha_free_start = num_models + num_benchmarks  # not needed directly here

                # After shifting by D_anchor to match anchor_difficulty, propagate variance:
                d_anchor_idx_in_theta = idx_D_start + anchor_bench_idx
                var_D_anchor = var_at(d_anchor_idx_in_theta)

                se_C = np.empty(num_models)
                for i in range(num_models):
                    var_ci = var_at(idx_C_start + i)
                    cov_ci_danchor = cov_at(idx_C_start + i, d_anchor_idx_in_theta)
                    var_new = var_ci + var_D_anchor - 2.0 * cov_ci_danchor
                    se_C[i] = np.sqrt(max(var_new, 0.0))

                se_D = np.empty(num_benchmarks)
                for j in range(num_benchmarks):
                    if j == anchor_bench_idx:
                        se_D[j] = 0.0  # anchored exactly after shift
                    else:
                        var_dj = var_at(idx_D_start + j)
                        cov_dj_danchor = cov_at(idx_D_start + j, d_anchor_idx_in_theta)
                        var_new = var_dj + var_D_anchor - 2.0 * cov_dj_danchor
                        se_D[j] = np.sqrt(max(var_new, 0.0))

                # Slopes: anchored slope fixed, others read from cov directly
                se_alpha_free = np.sqrt(np.clip(np.diag(cov_theta)[num_models + num_benchmarks :], 0, np.inf))
                se_alpha = np.insert(se_alpha_free, anchor_bench_idx, 0.0)

            else:  # anchor_mode == "model"
                # Parameterization in theta: [C_free (M-2), D (B), alpha (B)]
                idx_C_free_start = 0
                idx_D_start = num_models - 2
                idx_alpha_start = num_models - 2 + num_benchmarks

                se_C_free = np.sqrt(np.clip(np.diag(cov_theta)[idx_C_free_start : idx_C_free_start + (num_models - 2)], 0, np.inf))
                se_D = np.sqrt(np.clip(np.diag(cov_theta)[idx_D_start : idx_D_start + num_benchmarks], 0, np.inf))
                se_alpha = np.sqrt(np.clip(np.diag(cov_theta)[idx_alpha_start : idx_alpha_start + num_benchmarks], 0, np.inf))

                se_C = np.zeros(num_models)
                free_idx = 0
                for i in range(num_models):
                    if i in anchor_model_indices:
                        se_C[i] = 0.0
                    else:
                        se_C[i] = se_C_free[free_idx]
                        free_idx += 1

            # CI multiplier (normal approx)
            if ci_level is not None:
                if abs(ci_level - 0.90) < 1e-9:
                    z = 1.6448536269514722
                else:
                    # Simple approximation for other levels (Abramowitz-Stegun)
                    from math import sqrt, log
                    p = 0.5 + ci_level / 2.0
                    t = sqrt(-2.0 * log(1.0 - p))
                    z = t - ((2.515517 + 0.802853*t + 0.010328*t*t) / (1 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t))
            else:
                z = None
        except Exception:
            se_C = se_D = se_alpha = None
    
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

    # Attach SEs/CIs if computed
    if se_C is not None:
        model_cap_df["se_capability"] = se_C
        if ci_level is not None:
            z_mult = 1.6448536269514722 if abs(ci_level - 0.90) < 1e-9 else None
            if z_mult is None:
                from math import sqrt, log
                p = 0.5 + ci_level / 2.0
                t = sqrt(-2.0 * log(1.0 - p))
                z_mult = t - ((2.515517 + 0.802853*t + 0.010328*t*t) / (1 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t))
            model_cap_df["ci90_low"] = model_cap_df["estimated_capability"] - z_mult * model_cap_df["se_capability"]
            model_cap_df["ci90_high"] = model_cap_df["estimated_capability"] + z_mult * model_cap_df["se_capability"]

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

    if se_D is not None and se_alpha is not None:
        benchmark_params_df["se_difficulty"] = se_D
        benchmark_params_df["se_slope"] = se_alpha
        if ci_level is not None:
            z_mult = 1.6448536269514722 if abs(ci_level - 0.90) < 1e-9 else None
            if z_mult is None:
                from math import sqrt, log
                p = 0.5 + ci_level / 2.0
                t = sqrt(-2.0 * log(1.0 - p))
                z_mult = t - ((2.515517 + 0.802853*t + 0.010328*t*t) / (1 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t))
            benchmark_params_df["ci90_low_difficulty"] = benchmark_params_df["estimated_difficulty"] - z_mult * benchmark_params_df["se_difficulty"]
            benchmark_params_df["ci90_high_difficulty"] = benchmark_params_df["estimated_difficulty"] + z_mult * benchmark_params_df["se_difficulty"]
            benchmark_params_df["ci90_low_slope"] = benchmark_params_df["estimated_slope"] - z_mult * benchmark_params_df["se_slope"]
            benchmark_params_df["ci90_high_slope"] = benchmark_params_df["estimated_slope"] + z_mult * benchmark_params_df["se_slope"]
    
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
