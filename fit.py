import pandas as pd
import numpy as np
from data_loader import df_model
from scipy.optimize import least_squares
from typing import Optional

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
                         random_state: Optional[object] = None):
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
    random_state: Optional seed (int) or numpy.random.Generator for reproducible initialization
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
    # Use a local RNG to avoid mutating global numpy RNG state.
    # To preserve legacy behavior for notebooks that don't pass a seed,
    # replicate the old np.random.seed(42); np.random.randn(...) draws
    # by using a dedicated RandomState(42).
    legacy_rng = None
    rng = None
    if random_state is None:
        legacy_rng = np.random.RandomState(42)
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(int(random_state))
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng()
    
    if anchor_mode == "benchmark":
        # Original: C and D free, one α fixed
        if legacy_rng is not None:
            initial_C = legacy_rng.randn(num_models) * 0.1
            initial_D = legacy_rng.randn(num_benchmarks) * 0.1
        else:
            initial_C = rng.normal(0.0, 0.1, size=num_models)
            initial_D = rng.normal(0.0, 0.1, size=num_benchmarks)
        initial_alpha = np.full(num_benchmarks - 1, slope_init)
        initial_theta = np.concatenate([initial_C, initial_D, initial_alpha])
        
    elif anchor_mode == "model":
        # New: two C fixed, all D and α free
        if legacy_rng is not None:
            initial_C_free = legacy_rng.randn(num_models - 2) * 0.1
            initial_D      = legacy_rng.randn(num_benchmarks) * 0.1
        else:
            initial_C_free = rng.normal(0.0, 0.1, size=num_models - 2)
            initial_D      = rng.normal(0.0, 0.1, size=num_benchmarks)
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