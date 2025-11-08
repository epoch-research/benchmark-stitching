"""Pure analysis code for linear model method.

This module fits a linear model to predict ECI from log(compute) and release date,
then analyzes the compute-year tradeoff for algorithmic progress.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from shared.data_loading import load_model_capabilities_and_compute, prepare_for_analysis
from shared.bootstrap import bootstrap_linear_regression, compute_tradeoff_from_coefficients


def compute_pareto_frontier_at_release(df):
    """Identify which models were on the Pareto frontier at their release date.

    For each model, check if any model released on or before that date had both:
    - Lower or equal compute AND
    - Higher or equal ECI

    Args:
        df: DataFrame with 'date_obj', 'compute', 'estimated_capability' columns

    Returns:
        Series of boolean values indicating frontier status
    """
    df_sorted = df.sort_values('date_obj').copy()
    is_frontier = []

    for idx, row in df_sorted.iterrows():
        earlier_or_same = df_sorted[df_sorted['date_obj'] <= row['date_obj']]

        dominated = False
        for _, other_row in earlier_or_same.iterrows():
            if other_row.name == idx:
                continue

            if (other_row['compute'] <= row['compute'] and
                other_row['estimated_capability'] >= row['estimated_capability']):
                if (other_row['compute'] < row['compute'] or
                    other_row['estimated_capability'] > row['estimated_capability']):
                    dominated = True
                    break

        is_frontier.append(not dominated)

    result = pd.Series(is_frontier, index=df_sorted.index)
    return result


def load_and_filter_data(exclude_distilled=False,
                         include_low_confidence=False,
                         frontier_only=False,
                         use_website_data=False,
                         min_release_date=None):
    """Load ECI scores and merge with compute data.

    Args:
        exclude_distilled: If True, exclude distilled models
        include_low_confidence: If True, also exclude low-confidence distilled models
        frontier_only: If True, only include models on Pareto frontier at release
        use_website_data: If True, load from website data
        min_release_date: If provided, only include models released on or after this date

    Returns:
        DataFrame with complete data for plotting, or None on error
    """
    # Load data
    df = load_model_capabilities_and_compute(
        use_website_data=use_website_data,
        exclude_distilled=exclude_distilled,
        include_low_confidence=include_low_confidence,
        filter_complete=True,
        min_release_date=min_release_date
    )

    if df is None:
        return None

    # Optionally filter to only frontier models
    if frontier_only:
        print("\nFiltering to only models on Pareto frontier at release date...")
        is_frontier = compute_pareto_frontier_at_release(df)
        df = df[is_frontier].copy()
        print(f"Filtered to {len(df)} frontier models")

        if len(df) > 0:
            print("\nSample frontier models:")
            sample_models = df.nlargest(5, 'estimated_capability')[
                ['Model', 'date', 'compute', 'estimated_capability']]
            for _, row in sample_models.iterrows():
                print(f"  {row['Model']}: ECI={row['estimated_capability']:.2f}, "
                      f"Compute={row['compute']:.2e}, Date={row['date']}")

    return df


def fit_linear_predictor(df_plot, n_bootstrap=1000):
    """Fit a linear model to predict ECI from log(compute) and date with bootstrap uncertainty.

    Args:
        df_plot: DataFrame with date_numeric, log_compute, estimated_capability
        n_bootstrap: Number of bootstrap iterations

    Returns:
        model: Fitted LinearRegression model
        df_plot: DataFrame (with date_numeric and log_compute columns added if needed)
        bootstrap_results: Dict with bootstrap statistics
    """
    # Prepare data - add log_compute and date_numeric if not present
    if 'log_compute' not in df_plot.columns or 'date_numeric' not in df_plot.columns:
        df_plot = prepare_for_analysis(df_plot)

    # Prepare data
    X = df_plot[['log_compute', 'date_numeric']].values
    y = df_plot['estimated_capability'].values

    # Fit main model
    model = LinearRegression()
    model.fit(X, y)

    score = model.score(X, y)
    print(f"\nLinear predictor R² = {score:.4f}")
    print(f"Coefficients: log(compute)={model.coef_[0]:.4f}, "
          f"date={model.coef_[1]:.4f}, intercept={model.intercept_:.4f}")

    # Bootstrap for uncertainty estimation
    print(f"\nRunning {n_bootstrap} bootstrap iterations for uncertainty estimation...")
    bootstrap_results = bootstrap_linear_regression(X, y, n_bootstrap=n_bootstrap)

    print(f"\n95% Confidence Intervals:")
    print(f"  log(compute): {model.coef_[0]:.4f} [{bootstrap_results['slope_ci'][0, 0]:.4f}, {bootstrap_results['slope_ci'][1, 0]:.4f}]")
    print(f"  date: {model.coef_[1]:.4f} [{bootstrap_results['slope_ci'][0, 1]:.4f}, {bootstrap_results['slope_ci'][1, 1]:.4f}]")
    print(f"  intercept: {model.intercept_:.4f} [{bootstrap_results['intercept_ci'][0]:.4f}, {bootstrap_results['intercept_ci'][1]:.4f}]")

    # Calculate compute-year tradeoff
    tradeoff_results = compute_tradeoff_from_coefficients(
        model.coef_[0], model.coef_[1],
        bootstrap_coefs=bootstrap_results['slopes']
    )

    print(f"\n{'='*70}")
    print(f"COMPUTE-YEAR TRADEOFF ANALYSIS")
    print(f"{'='*70}")
    print(f"Question: How many OOMs of compute = 1 year of algorithmic progress?")
    print(f"\nMain estimate: {tradeoff_results['tradeoff']:.4f} OOMs per year")
    print(f"Bootstrap mean: {tradeoff_results['tradeoff_mean']:.4f} OOMs per year")
    print(f"Bootstrap median: {tradeoff_results['tradeoff_median']:.4f} OOMs per year")
    print(f"95% CI: [{tradeoff_results['tradeoff_ci'][0]:.4f}, {tradeoff_results['tradeoff_ci'][1]:.4f}] OOMs per year")
    print(f"\nInterpretation: Waiting 1 year for algorithmic improvements")
    print(f"is equivalent to ~{10**tradeoff_results['tradeoff_mean']:.1f}× more training compute")
    print(f"(range: {10**tradeoff_results['tradeoff_ci'][0]:.1f}× to {10**tradeoff_results['tradeoff_ci'][1]:.1f}×)")
    print(f"{'='*70}")

    # Add tradeoff results to bootstrap_results
    bootstrap_results['compute_year_tradeoff'] = tradeoff_results['bootstrap_tradeoffs']

    return model, df_plot, bootstrap_results
