#!/usr/bin/env python3
"""
Create a plot showing:
- X-axis: Release date
- Y-axis: Training compute (log scale)
- Each data point is a colored circle with its ECI (Epoch Capabilities Index)
- Contour lines showing constant ECI values
- Linear predictor for ECI based on log(compute) and date

Uses caching to speed up iterations on the visualization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.linear_model import LinearRegression
import pickle


def compute_pareto_frontier_at_release(df):
    """
    Identify which models were on the Pareto frontier at their release date.

    For each model, check if any model released on or before that date had both:
    - Lower or equal compute AND
    - Higher or equal ECI

    If such a model exists, the current model is dominated and NOT on the frontier.

    Args:
        df: DataFrame with 'date_obj', 'compute', 'estimated_capability' columns

    Returns:
        Series of boolean values indicating frontier status
    """
    df_sorted = df.sort_values('date_obj').copy()
    is_frontier = []

    for idx, row in df_sorted.iterrows():
        # Get all models released on or before this date
        earlier_or_same = df_sorted[df_sorted['date_obj'] <= row['date_obj']]

        # Check if this model is dominated by any earlier model
        # A model is dominated if there exists another model with:
        # - compute <= this model's compute AND
        # - ECI >= this model's ECI
        # AND at least one inequality is strict
        dominated = False
        for _, other_row in earlier_or_same.iterrows():
            if other_row.name == idx:
                continue  # Don't compare to self

            # Check if other model dominates this one
            if (other_row['compute'] <= row['compute'] and
                other_row['estimated_capability'] >= row['estimated_capability']):
                # At least one must be strictly better
                if (other_row['compute'] < row['compute'] or
                    other_row['estimated_capability'] > row['estimated_capability']):
                    dominated = True
                    break

        is_frontier.append(not dominated)

    # Create a series with original index
    result = pd.Series(is_frontier, index=df_sorted.index)
    return result


def load_or_fit_model(cache_file="outputs/algorithmic_progress_methods/linear_model/fitted_capabilities_cache.pkl",
                      force_refit=False,
                      exclude_distilled=False,
                      include_low_confidence=False,
                      frontier_only=False):
    """Load ECI scores from outputs/model_fit/model_capabilities.csv and merge with compute data

    Args:
        cache_file: Path to cache file
        force_refit: If True, ignore cache and refit
        exclude_distilled: If True, exclude distilled models from analysis
        include_low_confidence: If True, also exclude low-confidence distilled models
        frontier_only: If True, only include models on Pareto frontier at release
    """
    # Use different cache file if excluding distilled models or frontier-only
    suffix_parts = []
    if exclude_distilled:
        suffix_parts.append("no_distilled_all" if include_low_confidence else "no_distilled")
    if frontier_only:
        suffix_parts.append("frontier_only")

    if suffix_parts:
        suffix = "_" + "_".join(suffix_parts)
        cache_file = cache_file.replace(".pkl", f"{suffix}.pkl")

    cache_path = Path(cache_file)

    if cache_path.exists() and not force_refit:
        print(f"Loading cached model from {cache_path}...")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Loaded cached data for {len(cached_data['df_plot'])} models")
        return cached_data['df_plot']

    print("Loading ECI scores from outputs/model_fit/model_capabilities.csv...")

    # Load ECI scores (already has 'estimated_capability' column)
    eci_df = pd.read_csv("outputs/model_fit/model_capabilities.csv")
    # Rename 'model' column if it exists, otherwise the data already has the right column names
    print(f"Loaded ECI scores for {len(eci_df)} models")

    # Filter out distilled models if requested
    if exclude_distilled:
        print("\nFiltering out distilled models...")
        distilled_df = pd.read_csv("data/distilled_models.csv")

        # Determine which confidence levels to exclude
        if include_low_confidence:
            confidence_levels = ['high', 'medium', 'low']
            print("  Excluding: high, medium, AND low confidence distilled models")
        else:
            confidence_levels = ['high', 'medium']
            print("  Excluding: high and medium confidence distilled models only")

        distilled_models = distilled_df[
            (distilled_df['distilled'] == True) &
            (distilled_df['confidence'].isin(confidence_levels))
        ]['model'].tolist()

        before_count = len(eci_df)
        eci_df = eci_df[~eci_df['model'].isin(distilled_models)]
        after_count = len(eci_df)

        print(f"Excluded {before_count - after_count} distilled models "
              f"({before_count - after_count} / {before_count} = "
              f"{100 * (before_count - after_count) / before_count:.1f}%)")
        print(f"Remaining models: {after_count}")

    # Prepare capability data
    # The date_obj column already exists in model_capabilities.csv
    if 'date_obj' not in eci_df.columns:
        eci_df['date_obj'] = pd.to_datetime(eci_df['date'])
    else:
        # Ensure it's datetime type
        eci_df['date_obj'] = pd.to_datetime(eci_df['date_obj'])
    df_cap = eci_df.copy(deep=True)

    # Verify anchor models
    anchor_models = df_cap[df_cap['model'].isin([
        'claude-3-5-sonnet-20240620', 'gpt-5-2025-08-07_medium'])]
    if len(anchor_models) > 0:
        print("\nAnchor models in ECI data:")
        for _, row in anchor_models.iterrows():
            print(f"  {row['model']}: ECI={row['estimated_capability']:.1f}")

    # Load compute data and merge
    try:
        pcd_dataset = pd.read_csv("data/all_ai_models.csv")[
            ["Model", "Training compute (FLOP)", "Parameters",
             "Training dataset size (datapoints)"]
        ]
        columns = {
            "Training compute (FLOP)": "compute",
            "Parameters": "parameters",
            "Training dataset size (datapoints)": "data"
        }
        pcd_dataset = pcd_dataset.rename(columns=columns)

        df_cap = df_cap.merge(pcd_dataset, on="Model", how="left")
        print(f"Merged compute data: {df_cap['compute'].notna().sum()} models have compute info")
    except Exception as e:
        print(f"Error: Could not load compute data: {e}")
        return None

    # Filter to only models with complete data FOR PLOTTING
    df_plot = df_cap.dropna(subset=['date_obj', 'compute', 'estimated_capability'])
    print(f"Prepared {len(df_plot)} models with complete data for plotting")

    # Optionally filter to only frontier models
    if frontier_only:
        print("\nFiltering to only models on Pareto frontier at release date...")
        is_frontier = compute_pareto_frontier_at_release(df_plot)
        df_plot = df_plot[is_frontier].copy()
        print(f"Filtered to {len(df_plot)} frontier models (out of {len(is_frontier)} total)")

        # Show some examples
        if len(df_plot) > 0:
            print("\nSample frontier models:")
            sample_models = df_plot.nlargest(5, 'estimated_capability')[['Model', 'date', 'compute', 'estimated_capability']]
            for _, row in sample_models.iterrows():
                print(f"  {row['Model']}: ECI={row['estimated_capability']:.2f}, "
                      f"Compute={row['compute']:.2e}, Date={row['date']}")

    # Cache the results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'df_plot': df_plot}, f)
    print(f"Cached data saved to {cache_path}")

    return df_plot


def fit_linear_predictor(df_plot, n_bootstrap=1000):
    """Fit a linear model to predict ECI from log(compute) and date with bootstrap uncertainty"""
    # Convert date to days since epoch for numerical regression
    df_plot['date_numeric'] = (df_plot['date_obj'] -
                                pd.Timestamp('2020-01-01')).dt.total_seconds() / (365.25 * 24 * 3600)
    df_plot['log_compute'] = np.log10(df_plot['compute'])

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
    np.random.seed(42)
    bootstrap_coefs = []
    bootstrap_intercepts = []
    bootstrap_predictions = []

    n_samples = len(df_plot)
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"  Bootstrap iteration {i + 1}/{n_bootstrap}")

        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Fit model on bootstrap sample
        model_boot = LinearRegression()
        model_boot.fit(X_boot, y_boot)

        bootstrap_coefs.append(model_boot.coef_)
        bootstrap_intercepts.append(model_boot.intercept_)

        # Store predictions on original data
        y_pred_boot = model_boot.predict(X)
        bootstrap_predictions.append(y_pred_boot)

    bootstrap_coefs = np.array(bootstrap_coefs)
    bootstrap_intercepts = np.array(bootstrap_intercepts)
    bootstrap_predictions = np.array(bootstrap_predictions)

    # Calculate confidence intervals for coefficients
    coef_ci = np.percentile(bootstrap_coefs, [2.5, 97.5], axis=0)
    intercept_ci = np.percentile(bootstrap_intercepts, [2.5, 97.5])

    print(f"\n95% Confidence Intervals:")
    print(f"  log(compute): {model.coef_[0]:.4f} [{coef_ci[0, 0]:.4f}, {coef_ci[1, 0]:.4f}]")
    print(f"  date: {model.coef_[1]:.4f} [{coef_ci[0, 1]:.4f}, {coef_ci[1, 1]:.4f}]")
    print(f"  intercept: {model.intercept_:.4f} [{intercept_ci[0]:.4f}, {intercept_ci[1]:.4f}]")

    # Calculate compute-year tradeoff: OOMs of compute equivalent to 1 year of progress
    # From the model: ECI = β_compute * log10(compute) + β_date * date + intercept
    # For constant ECI: Δ(log10(compute)) = -(β_date / β_compute) * Δ(date)
    # For 1 year: OOMs_per_year = β_date / β_compute
    compute_year_tradeoff = bootstrap_coefs[:, 1] / bootstrap_coefs[:, 0]  # date / log(compute)
    tradeoff_mean = compute_year_tradeoff.mean()
    tradeoff_median = np.median(compute_year_tradeoff)
    tradeoff_ci = np.percentile(compute_year_tradeoff, [2.5, 97.5])

    print(f"\n{'='*70}")
    print(f"COMPUTE-YEAR TRADEOFF ANALYSIS")
    print(f"{'='*70}")
    print(f"Question: How many OOMs of compute = 1 year of algorithmic progress?")
    print(f"(i.e., for constant capability, how much can you reduce compute by")
    print(f" waiting 1 year for better algorithms?)")
    print(f"\nMain estimate: {model.coef_[1] / model.coef_[0]:.4f} OOMs per year")
    print(f"Bootstrap mean: {tradeoff_mean:.4f} OOMs per year")
    print(f"Bootstrap median: {tradeoff_median:.4f} OOMs per year")
    print(f"95% CI: [{tradeoff_ci[0]:.4f}, {tradeoff_ci[1]:.4f}] OOMs per year")
    print(f"\nInterpretation: Waiting 1 year for algorithmic improvements")
    print(f"is equivalent to ~{10**tradeoff_mean:.1f}× more training compute")
    print(f"(range: {10**tradeoff_ci[0]:.1f}× to {10**tradeoff_ci[1]:.1f}×)")
    print(f"{'='*70}")

    # Calculate prediction uncertainty
    pred_mean = bootstrap_predictions.mean(axis=0)
    pred_std = bootstrap_predictions.std(axis=0)
    pred_ci_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
    pred_ci_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)

    bootstrap_results = {
        'coefs': bootstrap_coefs,
        'intercepts': bootstrap_intercepts,
        'predictions': bootstrap_predictions,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_ci_lower': pred_ci_lower,
        'pred_ci_upper': pred_ci_upper,
        'compute_year_tradeoff': compute_year_tradeoff
    }

    return model, df_plot, bootstrap_results


def add_eci_contours(ax, df_plot, model, eci_levels=None, bootstrap_results=None):
    """Add contour lines for constant ECI values with optional uncertainty bands"""
    # Get date range
    date_min = df_plot['date_obj'].min()
    date_max = df_plot['date_obj'].max()

    # Convert to numeric for grid
    date_min_numeric = (date_min - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)
    date_max_numeric = (date_max - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)

    # Get compute range
    compute_min = df_plot['compute'].min()
    compute_max = df_plot['compute'].max()
    log_compute_min = np.log10(compute_min)
    log_compute_max = np.log10(compute_max)

    # Create grid
    date_numeric_grid = np.linspace(date_min_numeric, date_max_numeric, 100)
    log_compute_grid = np.linspace(log_compute_min, log_compute_max, 100)
    Date_numeric_mesh, LogCompute_mesh = np.meshgrid(date_numeric_grid, log_compute_grid)

    # Predict ECI on grid
    X_grid = np.column_stack([LogCompute_mesh.ravel(), Date_numeric_mesh.ravel()])
    ECI_grid = model.predict(X_grid).reshape(Date_numeric_mesh.shape)

    # Convert date numeric back to datetime for plotting (create meshgrid)
    date_obj_grid = pd.Timestamp('2020-01-01') + pd.to_timedelta(
        date_numeric_grid * 365.25, unit='D')
    Date_obj_mesh, _ = np.meshgrid(date_obj_grid, log_compute_grid)

    # Convert log compute back to linear (create meshgrid)
    Compute_mesh = 10 ** LogCompute_mesh

    # Define ECI levels if not provided
    if eci_levels is None:
        eci_min = df_plot['estimated_capability'].min()
        eci_max = df_plot['estimated_capability'].max()
        eci_range = eci_max - eci_min

        # Choose spacing based on range
        if eci_range > 10:
            spacing = 2.0
        elif eci_range > 5:
            spacing = 1.0
        elif eci_range > 2:
            spacing = 0.5
        else:
            spacing = 0.2

        # Create nice round numbers with appropriate spacing
        eci_levels = np.arange(np.floor(eci_min / spacing) * spacing,
                               np.ceil(eci_max / spacing) * spacing + spacing, spacing)
        print(f"Creating {len(eci_levels)} contour levels from {eci_levels.min():.2f} to {eci_levels.max():.2f} with spacing {spacing}")

    # If bootstrap results provided, add uncertainty bands for contours
    if bootstrap_results is not None:
        # Calculate prediction std on grid for uncertainty visualization
        bootstrap_preds_grid = []
        for i in range(len(bootstrap_results['coefs'])):
            coef = bootstrap_results['coefs'][i]
            intercept = bootstrap_results['intercepts'][i]
            pred = (X_grid @ coef + intercept).reshape(Date_numeric_mesh.shape)
            bootstrap_preds_grid.append(pred)

        bootstrap_preds_grid = np.array(bootstrap_preds_grid)
        ECI_grid_std = bootstrap_preds_grid.std(axis=0)

        # Add shaded uncertainty regions (every 10th to keep it manageable)
        for eci_level in eci_levels[::2]:  # Every other level
            # Upper and lower uncertainty bands
            contours_upper = ax.contour(Date_obj_mesh, Compute_mesh,
                                       ECI_grid + 1.96 * ECI_grid_std,
                                       levels=[eci_level], colors='gray',
                                       alpha=0.2, linewidths=1, linestyles='--', zorder=1)
            contours_lower = ax.contour(Date_obj_mesh, Compute_mesh,
                                       ECI_grid - 1.96 * ECI_grid_std,
                                       levels=[eci_level], colors='gray',
                                       alpha=0.2, linewidths=1, linestyles='--', zorder=1)

    # Plot main contours with much higher visibility
    print(f"Plotting contours with {len(eci_levels)} levels")
    print(f"Date mesh shape: {Date_obj_mesh.shape}, range: {Date_obj_mesh.min()} to {Date_obj_mesh.max()}")
    print(f"Compute mesh shape: {Compute_mesh.shape}, range: {Compute_mesh.min():.2e} to {Compute_mesh.max():.2e}")
    print(f"ECI grid shape: {ECI_grid.shape}, range: {ECI_grid.min():.2f} to {ECI_grid.max():.2f}")

    contours = ax.contour(Date_obj_mesh, Compute_mesh, ECI_grid,
                          levels=eci_levels, colors='black', alpha=0.8,
                          linewidths=2.5, linestyles='-', zorder=2)

    # Add labels with background for readability
    ax.clabel(contours, inline=True, fontsize=10, fmt='%.2f',
              colors='black', inline_spacing=10,
              use_clabeltext=True)

    print(f"Contours plotted successfully")
    return contours


def compute_predicted_frontier(model, date_obj, compute_range, df_plot):
    """
    Compute the Pareto frontier predicted by the linear model for a given date.

    The Pareto frontier is the set of (compute, ECI) pairs where no other point
    has both higher ECI and lower compute.

    For a linear model ECI = β_compute * log10(compute) + β_date * date + intercept,
    at fixed date, the frontier is simply the maximum ECI for each compute level.
    Since ECI increases monotonically with compute, the entire curve is the frontier.

    Args:
        model: Fitted LinearRegression model
        date_obj: datetime object for the date
        compute_range: array of compute values to evaluate
        df_plot: DataFrame with 'date_obj' for conversion

    Returns:
        compute_range: array of compute values
        predicted_eci: array of predicted ECI values
    """
    # Convert date to numeric
    date_numeric = (date_obj - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)

    # Create input array
    log_compute = np.log10(compute_range)
    X = np.column_stack([log_compute, np.full_like(log_compute, date_numeric)])

    # Predict ECI
    predicted_eci = model.predict(X)

    return compute_range, predicted_eci


def add_predicted_frontiers(ax, df_plot, model, frequency='MS'):
    """
    Add predicted Pareto frontiers for multiple dates

    Args:
        ax: matplotlib axis
        df_plot: DataFrame with data
        model: Fitted linear model
        frequency: pandas date frequency ('MS' for month start, 'QS' for quarter start)
    """
    # Get date range
    min_date = df_plot['date_obj'].min()
    max_date = df_plot['date_obj'].max()

    # Create monthly or quarterly time points
    time_points = pd.date_range(start=min_date, end=max_date, freq=frequency)
    if time_points[-1] < max_date:
        time_points = time_points.append(pd.DatetimeIndex([max_date]))

    # Get compute range (extend slightly for better visualization)
    compute_min = df_plot['compute'].min() * 0.5
    compute_max = df_plot['compute'].max() * 2.0
    compute_range = np.logspace(np.log10(compute_min), np.log10(compute_max), 200)

    # Convert dates to numeric for coloring
    date_numeric = mdates.date2num(df_plot['date_obj'])
    min_date_num = min(date_numeric)
    max_date_num = max(date_numeric)

    print(f"\nPlotting {len(time_points)} predicted frontiers...")

    # Plot frontiers in reverse order (oldest on top for visibility)
    for i, date_point in enumerate(reversed(time_points)):
        compute_vals, eci_vals = compute_predicted_frontier(
            model, date_point, compute_range, df_plot
        )

        # Get color based on date
        date_num = mdates.date2num(date_point)
        color_value = (date_num - min_date_num) / (max_date_num - min_date_num)
        color = plt.cm.plasma(color_value)  # Use different colormap to distinguish from data

        # Plot the frontier curve
        ax.plot(
            [date_point] * len(compute_vals),
            compute_vals,
            color=color,
            linewidth=2.0,
            alpha=0.6,
            zorder=2.5,
            linestyle='--'
        )

    # Add a legend entry for predicted frontiers
    ax.plot([], [], color='purple', linewidth=2.0, alpha=0.6,
            linestyle='--', label='Predicted Pareto frontiers (linear model)')

    return time_points


def plot_uncertainty_diagnostics(df_plot, bootstrap_results, output_dir,
                                exclude_distilled=False, include_low_confidence=False,
                                frontier_only=False):
    """Create diagnostic plots for bootstrap uncertainty"""
    suffix_parts = []
    if exclude_distilled:
        suffix_parts.append("no_distilled_all" if include_low_confidence else "no_distilled")
    if frontier_only:
        suffix_parts.append("frontier_only")

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Coefficient distributions
    ax = axes[0, 0]
    ax.hist(bootstrap_results['coefs'][:, 0], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(bootstrap_results['coefs'][:, 0].mean(), color='red',
               linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('log(compute) coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Bootstrap Distribution: log(compute) Coefficient')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(bootstrap_results['coefs'][:, 1], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(bootstrap_results['coefs'][:, 1].mean(), color='red',
               linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('date coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Bootstrap Distribution: Date Coefficient')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Prediction uncertainty vs actual values
    ax = axes[1, 0]
    pred_std = bootstrap_results['pred_std']
    actual_eci = df_plot['estimated_capability'].values

    scatter = ax.scatter(actual_eci, pred_std, c=df_plot['log_compute'],
                        cmap='viridis', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Actual ECI')
    ax.set_ylabel('Prediction Std Dev (bootstrap)')
    ax.set_title('Prediction Uncertainty vs Actual ECI')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='log(compute)')

    # 3. Residuals with uncertainty bands
    ax = axes[1, 1]
    residuals = actual_eci - bootstrap_results['pred_mean']
    ax.scatter(bootstrap_results['pred_mean'], residuals, alpha=0.6,
              s=80, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    # Add 95% CI bands
    sorted_idx = np.argsort(bootstrap_results['pred_mean'])
    pred_sorted = bootstrap_results['pred_mean'][sorted_idx]
    std_sorted = bootstrap_results['pred_std'][sorted_idx]

    ax.fill_between(pred_sorted, -1.96 * std_sorted, 1.96 * std_sorted,
                    alpha=0.2, color='gray', label='95% CI')
    ax.set_xlabel('Predicted ECI (mean)')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals with 95% Confidence Interval')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"bootstrap_uncertainty_diagnostics{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Uncertainty diagnostics saved to: {output_path}")
    plt.close()

    # Additional plot: Coefficient correlation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(bootstrap_results['coefs'][:, 0], bootstrap_results['coefs'][:, 1],
              alpha=0.3, s=20)
    ax.set_xlabel('log(compute) coefficient')
    ax.set_ylabel('date coefficient')
    ax.set_title('Bootstrap Coefficient Correlation')
    ax.grid(alpha=0.3)

    # Add correlation coefficient
    corr = np.corrcoef(bootstrap_results['coefs'][:, 0],
                       bootstrap_results['coefs'][:, 1])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.4f}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    output_path = output_dir / f"coefficient_correlation{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Coefficient correlation plot saved to: {output_path}")
    plt.close()

    # Compute-year tradeoff distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Distribution in OOMs
    ax = axes[0]
    tradeoff_oom = bootstrap_results['compute_year_tradeoff']
    ax.hist(tradeoff_oom, bins=50, alpha=0.7, edgecolor='black', color='steelblue')

    mean_val = tradeoff_oom.mean()
    median_val = np.median(tradeoff_oom)
    ci_vals = np.percentile(tradeoff_oom, [2.5, 97.5])

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    ax.axvline(ci_vals[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(ci_vals[1], color='gray', linestyle=':', linewidth=2, alpha=0.7, label=f'95% CI: [{ci_vals[0]:.3f}, {ci_vals[1]:.3f}]')

    ax.set_xlabel('OOMs per year', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Bootstrap Distribution: Compute-Year Tradeoff\n(OOMs of compute = 1 year of progress)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Right: Distribution as multiplier
    ax = axes[1]
    tradeoff_multiplier = 10 ** tradeoff_oom
    ax.hist(tradeoff_multiplier, bins=50, alpha=0.7, edgecolor='black', color='forestgreen')

    mean_mult = tradeoff_multiplier.mean()
    median_mult = np.median(tradeoff_multiplier)
    ci_mult = 10 ** ci_vals

    ax.axvline(mean_mult, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_mult:.1f}×')
    ax.axvline(median_mult, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_mult:.1f}×')
    ax.axvline(ci_mult[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(ci_mult[1], color='gray', linestyle=':', linewidth=2, alpha=0.7, label=f'95% CI: [{ci_mult[0]:.1f}×, {ci_mult[1]:.1f}×]')

    ax.set_xlabel('Compute multiplier (×)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Bootstrap Distribution: Compute-Year Tradeoff\n(Compute multiplier = 1 year of progress)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Use log scale if range is large
    if ci_mult[1] / ci_mult[0] > 100:
        ax.set_xscale('log')
        ax.set_xlabel('Compute multiplier (× , log scale)', fontsize=12)

    plt.tight_layout()

    output_path = output_dir / f"compute_year_tradeoff_distribution{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Compute-year tradeoff distribution saved to: {output_path}")
    plt.close()


def main():
    """Main plotting function"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot AI model compute vs date with ECI contours')
    parser.add_argument('--exclude-distilled', action='store_true',
                       help='Exclude distilled models (high/medium confidence) from analysis')
    parser.add_argument('--include-low-confidence', action='store_true',
                       help='When excluding distilled models, also exclude low-confidence ones (requires --exclude-distilled)')
    parser.add_argument('--force-refit', action='store_true',
                       help='Force refitting (ignore cache)')
    parser.add_argument('--show-predicted-frontier', action='store_true',
                       help='Show the Pareto frontier predicted by the linear model for each month')
    parser.add_argument('--frontier-only', action='store_true',
                       help='Only include models that were on the Pareto frontier at their release date')
    args = parser.parse_args()

    # Validate arguments
    if args.include_low_confidence and not args.exclude_distilled:
        parser.error('--include-low-confidence requires --exclude-distilled')

    # Load or fit model (use cache by default)
    df_plot = load_or_fit_model(
        force_refit=args.force_refit,
        exclude_distilled=args.exclude_distilled,
        include_low_confidence=args.include_low_confidence,
        frontier_only=args.frontier_only
    )

    if df_plot is None:
        print("Failed to load data. Exiting.")
        return

    # Fit linear predictor with bootstrap
    model, df_plot, bootstrap_results = fit_linear_predictor(df_plot, n_bootstrap=1000)

    # Create output directory (following project structure from LINEAR_NOTES.md)
    output_dir = Path("outputs/algorithmic_progress_methods/linear_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create uncertainty diagnostic plots
    print("\nCreating uncertainty diagnostic plots...")
    plot_uncertainty_diagnostics(df_plot, bootstrap_results, output_dir,
                                args.exclude_distilled, args.include_low_confidence,
                                args.frontier_only)

    # Create the main plot
    fig, ax = plt.subplots(figsize=(14, 9))

    # Normalize ECI for color mapping
    eci_values = df_plot['estimated_capability'].values
    eci_min, eci_max = eci_values.min(), eci_values.max()

    # Plot each point
    scatter = ax.scatter(
        df_plot['date_obj'],
        df_plot['compute'],
        c=df_plot['estimated_capability'],
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        zorder=3
    )

    # Add colorbar for ECI
    cbar = plt.colorbar(scatter, ax=ax,
                        label='ECI (Epoch Capabilities Index / Capability)')

    # Add contour lines for constant ECI (without uncertainty bands on main plot)
    add_eci_contours(ax, df_plot, model, bootstrap_results=None)

    # Optionally add predicted Pareto frontiers
    if args.show_predicted_frontier:
        add_predicted_frontiers(ax, df_plot, model, frequency='MS')

    # Add ECI labels to each point
    for _, row in df_plot.iterrows():
        ax.annotate(
            f"{row['estimated_capability']:.2f}",
            xy=(row['date_obj'], row['compute']),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=7,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='none', alpha=0.7),
            zorder=4
        )

    # Format y-axis as log scale
    ax.set_yscale('log')

    # Format x-axis for dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))

    # Labels and title
    ax.set_xlabel('Release Date', fontsize=12)
    ax.set_ylabel('Training Compute (FLOP)', fontsize=12)

    # Add suffix to title if excluding distilled models or frontier-only
    title_suffix_parts = []
    if args.exclude_distilled:
        if args.include_low_confidence:
            title_suffix_parts.append('excluding all distilled models')
        else:
            title_suffix_parts.append('excluding distilled models')
    if args.frontier_only:
        title_suffix_parts.append('frontier models only')

    title_suffix = ' (' + ', '.join(title_suffix_parts) + ')' if title_suffix_parts else ''

    title = (f'AI Models: Release Date vs Training Compute{title_suffix}\n'
             'Colored by ECI (black lines: constant ECI)')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)

    # Add legend if predicted frontier shown
    if args.show_predicted_frontier:
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Rotate date labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Add suffix to filename if excluding distilled models, showing predicted frontier, or frontier-only
    suffix_parts = []
    if args.exclude_distilled:
        suffix_parts.append("no_distilled_all" if args.include_low_confidence else "no_distilled")
    if args.show_predicted_frontier:
        suffix_parts.append("predicted_frontier")
    if args.frontier_only:
        suffix_parts.append("frontier_only")

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    output_path = output_dir / f"compute_vs_date_with_eci{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as SVG for higher quality
    output_path_svg = output_dir / f"compute_vs_date_with_eci{suffix}.svg"
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
    print(f"SVG version saved to: {output_path_svg}")

    plt.close()

    # Print some summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Number of models plotted: {len(df_plot)}")
    print(f"Date range: {df_plot['date_obj'].min().date()} to "
          f"{df_plot['date_obj'].max().date()}")
    print(f"Compute range: {df_plot['compute'].min():.2e} to "
          f"{df_plot['compute'].max():.2e} FLOP")
    print(f"ECI range: {eci_min:.4f} to {eci_max:.4f}")

    # Print anchor models for verification
    anchor_models = df_plot[df_plot['model'].isin([
        'claude-3-5-sonnet-20240620', 'gpt-5-2025-08-07_medium'])]
    if len(anchor_models) > 0:
        print("\nAnchor models:")
        for _, row in anchor_models.iterrows():
            print(f"  {row['model']}: ECI={row['estimated_capability']:.4f}")

    print("\nTop 5 models by ECI:")
    top_models = df_plot.nlargest(5, 'estimated_capability')[
        ['Model', 'date', 'compute', 'estimated_capability']]
    for _, row in top_models.iterrows():
        print(f"  {row['Model']}: ECI={row['estimated_capability']:.4f}, "
              f"Compute={row['compute']:.2e}, Date={row['date']}")


if __name__ == "__main__":
    main()
