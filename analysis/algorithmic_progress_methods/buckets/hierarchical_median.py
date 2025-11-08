#!/usr/bin/env python3
"""Hierarchical model for compute reduction median estimates.

This script sweeps over multiple ECI bucket sizes, re-runs the compute
reduction analysis for each sweep, and fits a Normal-Normal hierarchical
model to the resulting per-bucket slopes. The model treats each bucket
estimate as having measurement error (from bootstrap variability) and
partially pools them toward a global location parameter whose posterior
is reported as the median compute reduction rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import invgamma

# Change to project root so shared modules/data paths resolve correctly
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
os.chdir(PROJECT_ROOT)

# Ensure the shared analysis package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import analyze_compute_reduction  # noqa: E402
from shared.cli_utils import (  # noqa: E402
    add_data_source_args,
    add_distilled_filter_args,
    create_output_directory,
    generate_output_suffix,
    validate_distilled_args,
)
from shared.data_loading import load_model_capabilities_and_compute  # noqa: E402


@dataclass
class Observation:
    """Container for a single bucket-level slope observation."""

    bucket_size: float
    bucket_center: float
    compute_reduction: float  # Positive OOMs/year (−slope)
    measurement_sd: float
    n_models_total: int
    n_models_sota: int


def auto_bucket_sizes(df: pd.DataFrame, n_bucket_sizes: int) -> np.ndarray:
    """Generate evenly spaced bucket sizes between 5% and 25% of ECI range."""
    eci_range = df["estimated_capability"].max() - df["estimated_capability"].min()
    bucket_sizes = np.linspace(0.05 * eci_range, 0.25 * eci_range, n_bucket_sizes)
    return bucket_sizes


def collect_bucket_observations(
    df: pd.DataFrame,
    bucket_sizes: Iterable[float],
    min_models_per_bucket: int,
    n_bootstrap: int,
) -> Tuple[pd.DataFrame, List[Observation]]:
    """Run compute-reduction analysis for each bucket size and collect slopes."""
    records: List[Observation] = []

    for bucket_size in bucket_sizes:
        print("\n" + "-" * 50)
        print(f"Running compute reduction analysis for ECI bucket size={bucket_size:.3f}")
        results_df, bucket_data = analyze_compute_reduction(
            df,
            bucket_size_oom=bucket_size,
            min_models_per_bucket=min_models_per_bucket,
            n_bootstrap=n_bootstrap,
        )

        if len(results_df) == 0:
            print("  No valid buckets for this size; skipping.")
            continue

        # Build lookup from bucket center to bootstrap info for quick access
        bootstrap_lookup = {
            info["bucket_center"]: info["bootstrap_results"] for info in bucket_data
        }

        for _, row in results_df.iterrows():
            center = row["bucket_center"]
            bootstrap_results = bootstrap_lookup.get(center)
            if bootstrap_results is None:
                # Fallback: skip if we somehow lack bootstrap draws
                print(f"Warning: Missing bootstrap results for bucket {center:.3f}")
                continue

            slope_std = bootstrap_results.get("slope_std", np.nan)
            if np.isnan(slope_std) or slope_std <= 0:
                slopes = np.asarray(bootstrap_results.get("slopes", []))
                if slopes.size > 1:
                    slope_std = slopes.std(ddof=1)
                else:
                    ci = bootstrap_results.get("slope_ci")
                    if ci is not None:
                        slope_std = max((ci[1] - ci[0]) / (2 * 1.96), 1e-3)
                    else:
                        slope_std = max(row.get("std_err", 0.1), 1e-3)

            slope_std = max(abs(float(slope_std)), 1e-3)

            observation = Observation(
                bucket_size=float(bucket_size),
                bucket_center=float(center),
                compute_reduction=float(-row["slope_oom_per_year"]),
                measurement_sd=float(slope_std),
                n_models_total=int(row["n_models_total"]),
                n_models_sota=int(row["n_models_sota"]),
            )
            records.append(observation)

    if not records:
        return pd.DataFrame(), []

    data = pd.DataFrame(
        [
            {
                "bucket_size": obs.bucket_size,
                "bucket_center": obs.bucket_center,
                "compute_reduction_oom_per_year": obs.compute_reduction,
                "measurement_sd": obs.measurement_sd,
                "n_models_total": obs.n_models_total,
                "n_models_sota": obs.n_models_sota,
            }
            for obs in records
        ]
    )

    return data, records


def gibbs_hierarchical_normal(
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    n_iter: int = 20000,
    burn_in: int = 5000,
    thin: int = 10,
    mu_prior_mean: float = 1.0,
    mu_prior_sd: float = 3.0,
    tau_prior_alpha: float = 2.0,
    tau_prior_beta: float = 0.5,
    random_seed: int = 0,
) -> dict:
    """Run Gibbs sampling for Normal-Normal hierarchy with known SEs."""
    rng = np.random.default_rng(random_seed)
    n = y.size

    sigma = np.maximum(sigma, 1e-4)

    theta = y.copy()
    mu = np.median(y)
    tau2 = max(np.var(y), 0.05)

    kept_mu: List[float] = []
    kept_tau2: List[float] = []
    kept_theta: List[np.ndarray] = []

    for it in range(n_iter):
        # Sample theta_i | y_i, mu, tau2
        precision = 1.0 / sigma**2 + 1.0 / tau2
        mean_theta = (y / sigma**2 + mu / tau2) / precision
        std_theta = np.sqrt(1.0 / precision)
        theta = rng.normal(mean_theta, std_theta)

        # Sample mu | theta, tau2
        precision_mu = n / tau2 + 1.0 / (mu_prior_sd**2)
        mean_mu = (theta.sum() / tau2 + mu_prior_mean / (mu_prior_sd**2)) / precision_mu
        std_mu = np.sqrt(1.0 / precision_mu)
        mu = rng.normal(mean_mu, std_mu)

        # Sample tau2 | theta, mu
        alpha = tau_prior_alpha + n / 2.0
        beta = tau_prior_beta + 0.5 * np.sum((theta - mu) ** 2)
        tau2 = invgamma.rvs(alpha, scale=beta, random_state=rng)

        if it >= burn_in and (it - burn_in) % thin == 0:
            kept_mu.append(float(mu))
            kept_tau2.append(float(tau2))
            kept_theta.append(theta.copy())

    if not kept_mu:
        raise RuntimeError("No posterior samples retained; check sampler settings.")

    return {
        "mu": np.array(kept_mu),
        "tau2": np.array(kept_tau2),
        "theta": np.stack(kept_theta, axis=0),
    }


def summarize_posterior(samples: np.ndarray) -> dict:
    """Return median, mean, and 95% interval for an array of draws."""
    return {
        "median": float(np.median(samples)),
        "mean": float(samples.mean()),
        "ci_lower": float(np.percentile(samples, 2.5)),
        "ci_upper": float(np.percentile(samples, 97.5)),
    }


def run(args: argparse.Namespace) -> None:
    """Execute the hierarchical analysis pipeline."""
    validate_distilled_args(args)

    df = load_model_capabilities_and_compute(
        use_website_data=args.use_website_data,
        exclude_distilled=args.exclude_distilled,
        exclude_med_high_distilled=args.exclude_med_high_distilled,
    )

    if df is None or len(df) == 0:
        print("No data available; aborting.")
        return

    bucket_sizes = (
        [float(x) for x in args.bucket_sizes.split(",")]
        if args.bucket_sizes
        else auto_bucket_sizes(df, args.n_bucket_sizes)
    )

    print("\nBucket sizes to evaluate:", ", ".join(f"{bs:.3f}" for bs in bucket_sizes))

    observations_df, records = collect_bucket_observations(
        df,
        bucket_sizes=bucket_sizes,
        min_models_per_bucket=args.min_models,
        n_bootstrap=args.n_bootstrap,
    )

    if observations_df.empty:
        print("No bucket observations collected; nothing to model.")
        return

    y = observations_df["compute_reduction_oom_per_year"].to_numpy()
    sigma = observations_df["measurement_sd"].to_numpy()

    samples = gibbs_hierarchical_normal(
        y,
        sigma,
        n_iter=args.n_iter,
        burn_in=args.burn_in,
        thin=args.thin,
        mu_prior_mean=args.mu_prior_mean,
        mu_prior_sd=args.mu_prior_sd,
        tau_prior_alpha=args.tau_prior_alpha,
        tau_prior_beta=args.tau_prior_beta,
        random_seed=args.seed,
    )

    mu_samples = samples["mu"]
    tau_samples = np.sqrt(samples["tau2"])
    theta_samples = samples["theta"]

    summary_mu = summarize_posterior(mu_samples)
    multiplier_samples = 10 ** mu_samples
    summary_multiplier = summarize_posterior(multiplier_samples)

    per_bucket_summaries = []
    for idx, obs in observations_df.reset_index(drop=True).iterrows():
        bucket_draws = theta_samples[:, idx]
        stats = summarize_posterior(bucket_draws)
        stats.update(
            {
                "bucket_size": float(obs["bucket_size"]),
                "bucket_center": float(obs["bucket_center"]),
                "n_models_total": int(obs["n_models_total"]),
                "n_models_sota": int(obs["n_models_sota"]),
            }
        )
        per_bucket_summaries.append(stats)

    bucket_size_summary = (
        observations_df.groupby("bucket_size")["compute_reduction_oom_per_year"]
        .median()
        .reset_index()
        .rename(columns={"compute_reduction_oom_per_year": "observed_median"})
    )

    # Create output directory and file paths
    output_dir = create_output_directory(
        "buckets",
        exclude_distilled=args.exclude_distilled,
        exclude_med_high_distilled=args.exclude_med_high_distilled,
        use_website_data=args.use_website_data,
    )
    suffix = generate_output_suffix(
        exclude_distilled=args.exclude_distilled,
        exclude_med_high_distilled=args.exclude_med_high_distilled,
        use_website_data=args.use_website_data,
    )

    observations_path = (
        output_dir / f"hierarchical_median_observations{suffix}.csv"
    )
    summary_path = output_dir / f"hierarchical_median_summary{suffix}.json"

    observations_df.to_csv(observations_path, index=False)

    summary_payload = {
        "settings": {
            "bucket_sizes": [float(x) for x in bucket_sizes],
            "min_models_per_bucket": args.min_models,
            "n_bootstrap": args.n_bootstrap,
            "n_iter": args.n_iter,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "seed": args.seed,
        },
        "posterior_mu_oom_per_year": summary_mu,
        "posterior_multiplier_per_year": summary_multiplier,
        "posterior_tau_oom_per_year": summarize_posterior(tau_samples),
        "posterior_mu_samples": [float(x) for x in mu_samples],
        "bucket_size_observed_medians": bucket_size_summary.to_dict(
            orient="records"
        ),
        "bucket_posteriors": per_bucket_summaries,
    }

    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    print("\nHierarchical median summary:")
    print(json.dumps(summary_payload["posterior_mu_oom_per_year"], indent=2))
    print("\nEquivalent compute multiplier per year:")
    print(json.dumps(summary_payload["posterior_multiplier_per_year"], indent=2))
    print(f"\nSaved observation table to: {observations_path}")
    print(f"Saved summary JSON to: {summary_path}")

    return {
        "summary_path": str(summary_path),
        "observations_path": str(observations_path),
        "summary_payload": summary_payload,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Configure argument parser for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Hierarchical median estimate for compute reduction rates."
    )
    parser.add_argument(
        "--bucket-sizes",
        type=str,
        default=None,
        help="Comma-separated ECI bucket sizes to sweep "
        "(default: auto range from 5% to 50% of ECI span).",
    )
    parser.add_argument(
        "--n-bucket-sizes",
        type=int,
        default=5,
        help="Number of bucket sizes to auto-generate when --bucket-sizes is not provided.",
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=3,
        help="Minimum SOTA models per bucket.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for slope uncertainty.",
    )
    parser.add_argument("--n-iter", type=int, default=20000, help="Gibbs iterations.")
    parser.add_argument("--burn-in", type=int, default=5000, help="Burn-in iterations.")
    parser.add_argument("--thin", type=int, default=10, help="Keep one sample every THIN steps.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--mu-prior-mean",
        type=float,
        default=1.0,
        help="Prior mean for global compute reduction (OOMs/year).",
    )
    parser.add_argument(
        "--mu-prior-sd",
        type=float,
        default=3.0,
        help="Prior standard deviation for global compute reduction.",
    )
    parser.add_argument(
        "--tau-prior-alpha",
        type=float,
        default=2.0,
        help="Shape parameter for tau² inverse-gamma prior.",
    )
    parser.add_argument(
        "--tau-prior-beta",
        type=float,
        default=0.5,
        help="Scale parameter for tau² inverse-gamma prior.",
    )

    add_distilled_filter_args(parser)
    add_data_source_args(parser)
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    run(parser.parse_args())
