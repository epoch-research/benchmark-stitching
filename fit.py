import pandas as pd
import numpy as np
from data_loader import df_model
from scipy.optimize import least_squares

def fit_statistical_model(df, anchor_benchmark, anchor_difficulty, anchor_slope, slope_init=0.05, df_model=df_model):
  # ------------------------------------------------------------
  # 1)  Mappings & data arrays
  # ------------------------------------------------------------
#   print("hello world")
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
  # 2)  Anchor benchmark
  # ------------------------------------------------------------
  try:
      anchor_bench_id = df.loc[
          df["benchmark"] == anchor_benchmark, "benchmark_id"
      ].iloc[0]
  except IndexError:
      raise ValueError(f"Benchmark named “{anchor_benchmark}” not found in df")

  anchor_idx = bench_id_to_idx[anchor_bench_id]      # 0-based position

  # ------------------------------------------------------------
  # 3)  Helpers
  # ------------------------------------------------------------
  def logistic(x: np.ndarray) -> np.ndarray:
      return 1.0 / (1.0 + np.exp(-x))


  def split_params(params: np.ndarray):
      """
      Break the flat parameter vector into C, D and full-length α,
      with α_anchor hard-set to 1.
      """
      C = params[:num_models]
      D = params[num_models : num_models + num_benchmarks]
      alpha_free = params[num_models + num_benchmarks :]          # length = num_benchmarks − 1
    #   alpha = np.insert(alpha_free, anchor_idx, 1.0)              # put the fixed 1 back in
      alpha = np.insert(alpha_free, anchor_idx, anchor_slope)
      return C, D, alpha


  def residuals(params, model_idx, bench_idx, y):
      C, D, alpha = split_params(params)
      preds = logistic(alpha[bench_idx] * (C[model_idx] - D[bench_idx]))
      return preds - y

  # ------------------------------------------------------------
  # 4)  Initial guesses  (note α vector is *one element shorter*)
  # ------------------------------------------------------------
  initial_C     = np.zeros(num_models)
  initial_D     = np.zeros(num_benchmarks)
  initial_alpha = np.full(num_benchmarks - 1, slope_init)
  initial_theta = np.concatenate([initial_C, initial_D, initial_alpha])

  # ------------------------------------------------------------
  # 5)  Fit
  # ------------------------------------------------------------
  result = least_squares(
      residuals,
      initial_theta,
      args=(model_idx_data, bench_idx_data, observed_scores),
      method="trf",      # default; listed here for clarity
      verbose=1
  )

  # ------------------------------------------------------------
  # 6)  Recover full parameter vectors
  # ------------------------------------------------------------
  theta_hat      = result.x
  C_hat          = theta_hat[:num_models]
  D_hat          = theta_hat[num_models : num_models + num_benchmarks]
  alpha_free_hat = theta_hat[num_models + num_benchmarks :]
  alpha_hat      = np.insert(alpha_free_hat, anchor_idx, 1.0)
  shift = D_hat[anchor_idx] - anchor_difficulty
  C_hat -= shift
  D_hat -= shift

  # ------------------------------------------------------------
  # 7)  Pack tidy DataFrames for inspection / downstream use
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
  bench_dates = (
    df
    .drop_duplicates("benchmark_id")[["benchmark_id", "benchmark_release_date"]]
  )
  benchmark_params_df = (
    benchmark_params_df
    .merge(bench_dates, on="benchmark_id", how="left")
  )

#   print(benchmark_params_df)
#   print(bench_dates)

  return df, model_capabilities_df, benchmark_params_df