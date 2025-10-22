#!/usr/bin/env python3
"""
Benchmark anchor approach for optimization pressure analysis.

This script implements the benchmark anchor approach to analyze optimization pressure
by comparing model capabilities between optimized and unoptimized data partitions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr
from pathlib import Path
from datetime import datetime
from data_loader import scores_df
from fit import fit_statistical_model


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


def ensure_anchor_present(part_df: pd.DataFrame, anchor_name: str) -> pd.DataFrame:
    """Include the anchor benchmark rows if missing from the partition."""
    if anchor_name in part_df["benchmark"].unique():
        return part_df
    return pd.concat(
        [part_df, scores_df[scores_df["benchmark"] == anchor_name]], ignore_index=True
    )


def _simple_avg(series_list: list[pd.Series]) -> pd.Series:
    """Simple average of a list of pandas Series."""
    if not series_list:
        return pd.Series(dtype=float)
    return pd.concat(series_list, axis=1).mean(axis=1)


def save_results(results: dict, output_dir: Path = None):
    """
    Save analysis results to files.

    Args:
        results: Dictionary containing analysis results
        output_dir: Output directory path (defaults to outputs/benchmark_anchor_diff/)
    """
    if output_dir is None:
        output_dir = Path("outputs/benchmark_anchor_diff")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save approach 1 results
    comp1 = results["comp1"]
    comp1.to_csv(output_dir / "approach1_model_capabilities.csv", index=False)

    # Save approach 1 summary
    summary1 = results["summary1"]
    with open(output_dir / "approach1_summary.txt", "w") as f:
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

    # Save approach 2 results if available
    if "approach2_results" in results:
        approach2 = results["approach2_results"]
        with open(output_dir / "approach2_summary.txt", "w") as f:
            f.write("Benchmark Anchor Approach 2 Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total benchmarks: {approach2['n_benchmarks_total']}\n")
            f.write(f"Optimized benchmarks: {approach2['n_opt_benchmarks']}\n")
            f.write(f"Unoptimized benchmarks: {approach2['n_unopt_benchmarks']}\n")
            f.write(f"Number of models: {approach2['n_models_overlap']}\n")
            f.write(
                f"Mean difference (weighted): {approach2['delta_mean_weighted']:.6f}\n"
            )
            f.write(
                f"Median difference (weighted): {approach2['delta_median_weighted']:.6f}\n"
            )
            f.write(
                f"Standard deviation (weighted): {approach2['delta_std_weighted']:.6f}\n"
            )

    # Save execution summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"execution_summary_{timestamp}.txt", "w") as f:
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
        if "approach2_results" in results:
            f.write("Approach 2 - All benchmarks as anchors\n")
            f.write(
                f"  - Total benchmarks: {results['approach2_results']['n_benchmarks_total']}\n"
            )
            f.write(
                f"  - Models analyzed: {results['approach2_results']['n_models_overlap']}\n"
            )
            f.write(
                f"  - Mean delta (weighted): {results['approach2_results']['delta_mean_weighted']:.6f}\n"
            )

    print(f"\nResults saved to: {output_dir}")
    print("  - approach1_model_capabilities.csv: Model capability comparisons")
    print("  - approach1_summary.txt: Statistical summary for approach 1")
    if "approach2_results" in results:
        print("  - approach2_summary.txt: Statistical summary for approach 2")
    print(f"  - execution_summary_{timestamp}.txt: Execution details")


def run_benchmark_anchor_approach():
    """
    Run the benchmark anchor approach analysis.

    This function implements two approaches:
    1. Top-10 anchors by coverage within each partition
    2. All benchmarks as anchors with weighted averaging
    """
    print("Running Benchmark anchor approach...")

    # Reference fit anchored on Winogrande to cache per-benchmark difficulty (D) and slope (α)
    print("Creating reference fit...")
    ref_df, ref_cap, ref_bench = fit_statistical_model(
        scores_df,
        anchor_mode="benchmark",
        anchor_benchmark="Winogrande",
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
    opt_df = scores_df[scores_df["optimized"]].copy()
    unopt_df = scores_df[~scores_df["optimized"]].copy()

    # Approach 1: Top-10 by coverage within each partition
    print("\n=== Approach 1: Top-10 anchors by coverage ===")
    cov_opt = (
        opt_df.groupby("benchmark")["model"].nunique().sort_values(ascending=False)
    )
    cov_unopt = (
        unopt_df.groupby("benchmark")["model"].nunique().sort_values(ascending=False)
    )
    opt_anchors = cov_opt.head(10).index.tolist()
    unopt_anchors = cov_unopt.head(10).index.tolist()
    anchors20 = opt_anchors + unopt_anchors

    # Run fits for each anchor on both partitions (anchor enforced if missing)
    opt_caps_by_anchor = []
    unopt_caps_by_anchor = []
    for anc in anchors20:
        mcap_opt, _ = fit_subset_with_benchmark_anchor_ref_init(
            ensure_anchor_present(opt_df, anc), anc
        )
        mcap_unopt, _ = fit_subset_with_benchmark_anchor_ref_init(
            ensure_anchor_present(unopt_df, anc), anc
        )
        opt_caps_by_anchor.append(mcap_opt.set_index("model")["estimated_capability"])
        unopt_caps_by_anchor.append(
            mcap_unopt.set_index("model")["estimated_capability"]
        )

    # Average across anchors in each partition
    opt_avg20 = pd.concat(opt_caps_by_anchor, axis=1).mean(axis=1)
    unopt_avg20 = pd.concat(unopt_caps_by_anchor, axis=1).mean(axis=1)

    # Difference over overlapping models
    common20 = opt_avg20.index.intersection(unopt_avg20.index)
    delta20 = opt_avg20.loc[common20] - unopt_avg20.loc[common20]

    approach1_results = {
        "approach": 1,
        "n_opt_anchors": len(opt_anchors),
        "n_unopt_anchors": len(unopt_anchors),
        "n_models_overlap": int(len(common20)),
        "delta_mean": float(delta20.mean()),
        "delta_median": float(delta20.median()),
        "delta_std": float(delta20.std(ddof=0)),
    }
    print(approach1_results)

    # Approach 2: All benchmarks as anchors with weighted averaging
    print("\n=== Approach 2: All benchmarks as anchors with weighted averaging ===")

    # Count optimized vs unoptimized benchmarks
    bench_meta = scores_df.drop_duplicates("benchmark")[["benchmark", "optimized"]]
    opt_bench_set = set(bench_meta[bench_meta["optimized"]]["benchmark"])
    num_opt_total = int(len(opt_bench_set))
    num_unopt_total = int(len(bench_meta) - num_opt_total)
    num_total = num_opt_total + num_unopt_total

    # Per spec: weights = 11/31 for optimized-anchors and 20/31 for unoptimized-anchors (generalized)
    w_optAnch = num_unopt_total / num_total
    w_unoptAnch = num_opt_total / num_total

    all_benchmarks = bench_meta["benchmark"].tolist()

    opt_caps_optAnch = []
    opt_caps_unoptAnch = []
    unopt_caps_optAnch = []
    unopt_caps_unoptAnch = []

    for anc in all_benchmarks:
        is_opt_anchor = anc in opt_bench_set

        # Optimized partition
        mcap_opt, _ = fit_subset_with_benchmark_anchor_ref_init(
            ensure_anchor_present(opt_df, anc), anc
        )
        s_opt = mcap_opt.set_index("model")["estimated_capability"]
        (opt_caps_optAnch if is_opt_anchor else opt_caps_unoptAnch).append(s_opt)

        # Unoptimized partition
        mcap_unopt, _ = fit_subset_with_benchmark_anchor_ref_init(
            ensure_anchor_present(unopt_df, anc), anc
        )
        s_unopt = mcap_unopt.set_index("model")["estimated_capability"]
        (unopt_caps_optAnch if is_opt_anchor else unopt_caps_unoptAnch).append(s_unopt)

    # Averages within anchor categories
    opt_avg_over_optAnch = _simple_avg(opt_caps_optAnch)
    opt_avg_over_unoptAnch = _simple_avg(opt_caps_unoptAnch)
    unopt_avg_over_optAnch = _simple_avg(unopt_caps_optAnch)
    unopt_avg_over_unoptAnch = _simple_avg(unopt_caps_unoptAnch)

    # Weighted averages per set
    opt_weighted = (opt_avg_over_optAnch * w_optAnch).add(
        opt_avg_over_unoptAnch * w_unoptAnch, fill_value=0
    )
    unopt_weighted = (unopt_avg_over_optAnch * w_optAnch).add(
        unopt_avg_over_unoptAnch * w_unoptAnch, fill_value=0
    )

    # Difference over overlapping models
    common_w = opt_weighted.index.intersection(unopt_weighted.index)
    delta_w = opt_weighted.loc[common_w] - unopt_weighted.loc[common_w]

    approach2_results = {
        "approach": 2,
        "n_benchmarks_total": num_total,
        "n_opt_benchmarks": num_opt_total,
        "n_unopt_benchmarks": num_unopt_total,
        "n_models_overlap": int(len(common_w)),
        "delta_mean_weighted": float(delta_w.mean()),
        "delta_median_weighted": float(delta_w.median()),
        "delta_std_weighted": float(delta_w.std(ddof=0)),
    }
    print(approach2_results)

    # Statistical tests for Approach 1
    print("\n=== Statistical Analysis for Approach 1 ===")
    comp1 = (
        pd.DataFrame({"cap_opt20": opt_avg20, "cap_unopt20": unopt_avg20})
        .dropna()
        .assign(diff=lambda d: d["cap_opt20"] - d["cap_unopt20"])
        .reset_index()
        .rename(columns={"index": "model"})
    )

    # Summary stats (paired)
    t_stat1, t_p1 = ttest_rel(comp1["cap_opt20"], comp1["cap_unopt20"])
    w_stat1, w_p1 = wilcoxon(comp1["cap_opt20"], comp1["cap_unopt20"])
    r_pearson1, p_pearson1 = pearsonr(comp1["cap_opt20"], comp1["cap_unopt20"])
    r_spear1, p_spear1 = spearmanr(comp1["cap_opt20"], comp1["cap_unopt20"])
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
    top10 = comp1.nlargest(10, "diff")[["model", "cap_opt20", "cap_unopt20", "diff"]]
    print(top10.to_string(index=False))

    print("\nBottom 10 (opt - unopt):")
    bottom10 = comp1.nsmallest(10, "diff")[
        ["model", "cap_opt20", "cap_unopt20", "diff"]
    ]
    print(bottom10.to_string(index=False))

    results = {
        "approach1_results": approach1_results,
        "approach2_results": approach2_results,
        "summary1": summary1,
        "comp1": comp1,
        "opt_avg20": opt_avg20,
        "unopt_avg20": unopt_avg20,
        "delta20": delta20,
    }

    # Save results to files
    save_results(results)

    return results


if __name__ == "__main__":
    results = run_benchmark_anchor_approach()
    print("\nBenchmark anchor approach completed successfully!")
