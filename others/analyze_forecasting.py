#!/usr/bin/env python3
"""
Forecasting analysis - converts predicting_capabilities.ipynb to a Python script

This script focuses on forecasting frontier AI capabilities by tracking the top-performing
models over time rather than averaging across all models. This provides insights into 
when next-generation capabilities will emerge.

This script:
1. Creates predictive models for future AI capabilities using top N models
2. Validates forecasts using historical frontier model data
3. Generates capability forecasts with uncertainty bounds for frontier models
4. Compares pre-cutoff predictions to actual post-cutoff results

Usage: python analyze_forecasting.py [--cutoff-date YYYY-MM-DD] [--top-n-models N]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import argparse
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# Local imports
from data_loader import scores_df, df_model
from fit import fit_statistical_model
from analysis_utils import (
    setup_analysis_environment, setup_plotting_style,
    prepare_model_data, bootstrap_slope_analysis,
    filter_by_date, save_results_summary
)


def validate_forecast_accuracy(df_capabilities: pd.DataFrame, 
                             cutoff_date: str,
                             top_n_models: int,
                             output_dir: Path,
                             label_frontier: bool = False):
    """Validate forecast accuracy by comparing pre-cutoff predictions to post-cutoff reality"""
    print(f"Validating forecast accuracy with cutoff date: {cutoff_date} using models that were top {top_n_models} at release")
    
    df = prepare_model_data(df_capabilities)
    cutoff_dt = pd.to_datetime(cutoff_date)
    
    # Get all data for context plots
    pre_cutoff_all = filter_by_date(df, 'date_obj', cutoff_date, before=True)
    post_cutoff_all = filter_by_date(df, 'date_obj', cutoff_date, before=False)
    
    # Identify frontier models using the same logic as create_future_forecast
    # But only use pre-cutoff data for training
    frontier_models_pre = []
    frontier_models_post = []
    
    for _, model in df.iterrows():
        model_release_date = model['date_obj']
        
        # Find all models that existed at or before this model's release date
        available_models = df[df['date_obj'] <= model_release_date]
        
        # Check if this model was among the top N at its release time
        top_models_at_release = available_models.nlargest(top_n_models, 'estimated_capability')
        
        if model['model'] in top_models_at_release['model'].values:
            if model_release_date <= cutoff_dt:
                frontier_models_pre.append(model)
            else:
                frontier_models_post.append(model)
    
    pre_cutoff = pd.DataFrame(frontier_models_pre)
    post_cutoff = pd.DataFrame(frontier_models_post)
    
    if len(pre_cutoff) == 0 or len(post_cutoff) == 0:
        print("Insufficient data for validation - skipping accuracy analysis")
        return {}
    
    # Fit model on pre-cutoff data
    X_pre = (pre_cutoff['date_obj'] - pre_cutoff['date_obj'].min()).dt.days.values.reshape(-1, 1)
    y_pre = pre_cutoff['estimated_capability'].values
    
    model = LinearRegression()
    model.fit(X_pre, y_pre)
    
    # Make predictions for post-cutoff dates
    X_post = (post_cutoff['date_obj'] - pre_cutoff['date_obj'].min()).dt.days.values.reshape(-1, 1)
    y_pred = model.predict(X_post)
    y_actual = post_cutoff['estimated_capability'].values
    
    # Calculate accuracy metrics
    mae = np.mean(np.abs(y_pred - y_actual))
    rmse = np.sqrt(np.mean((y_pred - y_actual)**2))
    r2 = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - np.mean(y_actual))**2)
    
    # Create validation plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all models (faded) for context
    ax.scatter(pre_cutoff_all['date_obj'], pre_cutoff_all['estimated_capability'], 
              alpha=0.2, s=20, label='All models (pre-cutoff)', color='lightblue')
    ax.scatter(post_cutoff_all['date_obj'], post_cutoff_all['estimated_capability'], 
              alpha=0.2, s=20, label='All models (post-cutoff)', color='lightgreen')
    
    # Plot frontier models
    ax.scatter(pre_cutoff['date_obj'], pre_cutoff['estimated_capability'], 
              alpha=0.8, s=50, label=f'Frontier models (pre-cutoff)', color='blue')
    
    ax.scatter(post_cutoff['date_obj'], post_cutoff['estimated_capability'], 
              alpha=0.8, s=50, label=f'Frontier models (actual)', color='green')

    # Optional labels for frontier models
    if label_frontier:
        for _, r in pre_cutoff.dropna(subset=['date_obj', 'estimated_capability']).iterrows():
            ax.annotate(
                r['model'],
                xy=(r['date_obj'], r['estimated_capability']),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
                color='blue',
                alpha=0.9,
            )
        for _, r in post_cutoff.dropna(subset=['date_obj', 'estimated_capability']).iterrows():
            ax.annotate(
                r['model'],
                xy=(r['date_obj'], r['estimated_capability']),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
                color='green',
                alpha=0.9,
            )
    
    # Plot predictions
    ax.scatter(post_cutoff['date_obj'], y_pred, 
              alpha=0.7, s=50, marker='x', label='Predicted', color='red')
    
    # Plot trend line
    all_dates = pd.concat([pre_cutoff['date_obj'], post_cutoff['date_obj']])
    X_all = (all_dates - pre_cutoff['date_obj'].min()).dt.days.values.reshape(-1, 1)
    y_trend = model.predict(X_all)
    ax.plot(all_dates, y_trend, 'r--', alpha=0.7, label='Trend line')
    
    # Add cutoff line
    ax.axvline(x=cutoff_dt, color='black', linestyle=':', alpha=0.7, label='Cutoff date')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated Capability')
    ax.set_title(f'Frontier Forecast Validation (Models that were Top {top_n_models} at Release, Cutoff: {cutoff_date})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"forecast_validation_{cutoff_date}.png", dpi=300, bbox_inches='tight')
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_training': len(pre_cutoff),
        'n_validation': len(post_cutoff),
        'slope': model.coef_[0] * 365.25,  # Convert to per-year
        'intercept': model.intercept_
    }


def create_future_forecast(df_capabilities: pd.DataFrame,
                         forecast_years: int = 3,
                         top_n_models: int = 1,
                         output_dir: Path = None,
                         label_frontier: bool = False):
    """Create forecast for future capabilities based on models that were frontier at release"""
    print(f"Creating {forecast_years}-year capability forecast using models that were top {top_n_models} at release...")
    
    df = prepare_model_data(df_capabilities)
    
    # Identify frontier models: those that were among top N when they were released
    frontier_models = []
    
    for _, model in df.iterrows():
        model_release_date = model['date_obj']
        model_capability = model['estimated_capability']
        
        # Find all models that existed at or before this model's release date
        available_models = df[df['date_obj'] <= model_release_date]
        
        # Check if this model was among the top N at its release time
        top_models_at_release = available_models.nlargest(top_n_models, 'estimated_capability')
        
        if model['model'] in top_models_at_release['model'].values:
            frontier_models.append(model)
    
    # Convert to DataFrame
    df_frontier = pd.DataFrame(frontier_models)
    
    print(f"Using {len(df_frontier)} frontier model data points (from {len(df)} total)")
    print(f"Frontier models: {', '.join(df_frontier.nsmallest(10, 'date_obj')['model'].tolist())}")
    
    # Prepare data for modeling using only frontier models
    X = (df_frontier['date_obj'] - df_frontier['date_obj'].min()).dt.days.values.reshape(-1, 1)
    y = df_frontier['estimated_capability'].values
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Also fit with statsmodels for confidence intervals
    X_sm = sm.add_constant(X.flatten())
    model_sm = sm.OLS(y, X_sm).fit()
    
    # Create forecast dates
    last_date = df['date_obj'].max()
    forecast_end = last_date + timedelta(days=365.25 * forecast_years)
    forecast_dates = pd.date_range(start=last_date, end=forecast_end, freq='ME')
    
    # Convert forecast dates to numeric
    X_forecast = (forecast_dates - df['date_obj'].min()).days.values.reshape(-1, 1)
    X_forecast_sm = sm.add_constant(X_forecast.flatten())
    
    # Make predictions
    y_forecast = model.predict(X_forecast)
    
    # Get confidence and prediction intervals from statsmodels
    forecast_sm = model_sm.get_prediction(X_forecast_sm)
    ci_lower = forecast_sm.conf_int()[:, 0]
    ci_upper = forecast_sm.conf_int()[:, 1]
    
    # Get prediction intervals - handle different statsmodels versions
    try:
        pi_lower = forecast_sm.prediction_interval[:, 0]
        pi_upper = forecast_sm.prediction_interval[:, 1]
    except AttributeError:
        # Fallback for older statsmodels versions or manual calculation
        prediction_std_err = forecast_sm.se_mean
        from scipy import stats
        t_val = stats.t.ppf(0.975, model_sm.df_resid)  # 95% prediction interval
        pi_lower = y_forecast - t_val * prediction_std_err
        pi_upper = y_forecast + t_val * prediction_std_err
    
    # Create comprehensive forecast plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot all historical data (faded)
    ax.scatter(df['date_obj'], df['estimated_capability'], 
              alpha=0.3, s=20, label='All models', color='lightblue')
    
    # Plot frontier models data (highlighted)
    ax.scatter(df_frontier['date_obj'], df_frontier['estimated_capability'], 
              alpha=0.8, s=40, label=f'Models that were top {top_n_models} at release', color='blue')

    # Optional labels for frontier models
    if label_frontier:
        for _, r in df_frontier.dropna(subset=['date_obj', 'estimated_capability']).iterrows():
            ax.annotate(
                r['model'],
                xy=(r['date_obj'], r['estimated_capability']),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
                color='blue',
                alpha=0.9,
            )
    
    # Plot historical trend (based on frontier models)
    y_hist_trend = model.predict(X)
    ax.plot(df_frontier['date_obj'], y_hist_trend, 'b--', alpha=0.7, label='Frontier trend')
    
    # Plot forecast
    ax.plot(forecast_dates, y_forecast, 'r-', linewidth=2, label='Forecast')
    
    # Plot confidence intervals
    ax.fill_between(forecast_dates, ci_lower, ci_upper, 
                   alpha=0.3, color='red', label='95% Confidence interval')
    
    # Plot prediction intervals
    ax.fill_between(forecast_dates, pi_lower, pi_upper, 
                   alpha=0.2, color='red', label='95% Prediction interval')
    
    # Add vertical line for present
    ax.axvline(x=last_date, color='black', linestyle=':', alpha=0.7, label='Present')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated Capability')
    ax.set_title(f'{forecast_years}-Year AI Frontier Capability Forecast (Models that were Top {top_n_models} at Release)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f"capability_forecast_{forecast_years}yr.png", 
                   dpi=300, bbox_inches='tight')
    
    # Create forecast table
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'predicted_capability': y_forecast,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'pi_lower': pi_lower,
        'pi_upper': pi_upper
    })
    
    if output_dir:
        forecast_df.to_csv(output_dir / f"forecast_table_{forecast_years}yr.csv", index=False)
    
    return {
        'model': model,
        'forecast_df': forecast_df,
        'slope_per_year': model.coef_[0] * 365.25,
        'r2': model.score(X, y),
        'model_summary': model_sm.summary()
    }


def create_post_cutoff_frontier_forecast(
    df_capabilities: pd.DataFrame,
    cutoff_date: str,
    forecast_years: int = 3,
    top_n_models: int = 1,
    output_dir: Path | None = None,
    label_frontier: bool = False,
):
    """Create a post-cutoff forecast using only models that were frontier at release AFTER the cutoff.

    Fits a linear trend to post-cutoff frontier (top-N-at-release) points and extrapolates
    forecast_years into the future.
    """
    print(
        f"Creating post-cutoff ({cutoff_date}) {forecast_years}-year forecast using models that were top {top_n_models} at release..."
    )

    df = prepare_model_data(df_capabilities)
    cutoff_dt = pd.to_datetime(cutoff_date)

    # Identify post-cutoff frontier models: those that were among top N at their release, and released after cutoff
    post_cutoff_frontier_models: list[pd.Series] = []

    for _, model_row in df.iterrows():
        model_release_date = model_row['date_obj']
        if pd.isna(model_release_date) or pd.isna(model_row['estimated_capability']):
            continue

        if model_release_date <= cutoff_dt:
            continue

        available_models = df[df['date_obj'] <= model_release_date]
        top_models_at_release = available_models.nlargest(top_n_models, 'estimated_capability')

        if model_row['model'] in top_models_at_release['model'].values:
            post_cutoff_frontier_models.append(model_row)

    df_frontier_post = pd.DataFrame(post_cutoff_frontier_models)

    if len(df_frontier_post) < 2:
        print("Insufficient post-cutoff frontier data points to fit a trend. Skipping post-cutoff forecast plot.")
        return {}

    # Prepare numeric features from post-cutoff data only
    x0 = df_frontier_post['date_obj'].min()
    X = (df_frontier_post['date_obj'] - x0).dt.days.values.reshape(-1, 1)
    y = df_frontier_post['estimated_capability'].values

    # Fit models
    lr_model = LinearRegression()
    lr_model.fit(X, y)

    X_sm = sm.add_constant(X.flatten())
    ols_model = sm.OLS(y, X_sm).fit()

    # Forecast horizon based on latest available date in the full dataset
    last_date = df['date_obj'].max()
    forecast_end = last_date + timedelta(days=365.25 * forecast_years)
    forecast_dates = pd.date_range(start=last_date, end=forecast_end, freq='ME')

    # Use the same baseline (x0) for conversion to numeric
    X_forecast = (forecast_dates - x0).days.values.reshape(-1, 1)
    X_forecast_sm = sm.add_constant(X_forecast.flatten())

    # Predictions
    y_hist_trend = lr_model.predict(X)
    y_forecast = lr_model.predict(X_forecast)

    # Confidence and prediction intervals via statsmodels
    forecast_sm = ols_model.get_prediction(X_forecast_sm)
    ci_lower = forecast_sm.conf_int()[:, 0]
    ci_upper = forecast_sm.conf_int()[:, 1]

    try:
        pi_lower = forecast_sm.prediction_interval[:, 0]
        pi_upper = forecast_sm.prediction_interval[:, 1]
    except AttributeError:
        prediction_std_err = forecast_sm.se_mean
        t_val = stats.t.ppf(0.975, ols_model.df_resid)
        pi_lower = y_forecast - t_val * prediction_std_err
        pi_upper = y_forecast + t_val * prediction_std_err

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Context: all models
    ax.scatter(
        df['date_obj'],
        df['estimated_capability'],
        alpha=0.25,
        s=20,
        label='All models',
        color='lightblue',
    )

    # Highlight: post-cutoff frontier models used for fit
    ax.scatter(
        df_frontier_post['date_obj'],
        df_frontier_post['estimated_capability'],
        alpha=0.9,
        s=50,
        label=f'Post-cutoff models that were top {top_n_models} at release',
        color='blue',
    )

    # Optional labels for frontier models (post-cutoff)
    if label_frontier:
        for _, r in df_frontier_post.dropna(subset=['date_obj', 'estimated_capability']).iterrows():
            ax.annotate(
                r['model'],
                xy=(r['date_obj'], r['estimated_capability']),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
                color='blue',
                alpha=0.9,
            )

    # Trend on the post-cutoff segment
    ax.plot(
        df_frontier_post['date_obj'], y_hist_trend, 'b--', alpha=0.8, label='Post-cutoff frontier trend'
    )

    # Forecast and intervals
    ax.plot(forecast_dates, y_forecast, 'r-', linewidth=2, label='Post-cutoff forecast')
    ax.fill_between(
        forecast_dates, ci_lower, ci_upper, alpha=0.3, color='red', label='95% Confidence interval'
    )
    ax.fill_between(
        forecast_dates, pi_lower, pi_upper, alpha=0.2, color='red', label='95% Prediction interval'
    )

    # Lines for cutoff and present
    ax.axvline(x=cutoff_dt, color='black', linestyle=':', alpha=0.7, label='Cutoff date')
    ax.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, label='Present')

    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated Capability')
    ax.set_title(
        f'Post-cutoff Frontier Forecast (Top {top_n_models} at Release after {cutoff_date}, +{forecast_years}y)'
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save
    forecast_df = pd.DataFrame(
        {
            'date': forecast_dates,
            'predicted_capability': y_forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pi_lower': pi_lower,
            'pi_upper': pi_upper,
        }
    )

    if output_dir is not None:
        plt.savefig(
            output_dir / f"post_cutoff_frontier_forecast_{forecast_years}yr.png",
            dpi=300,
            bbox_inches='tight',
        )
        forecast_df.to_csv(
            output_dir / f"post_cutoff_forecast_table_{forecast_years}yr.csv", index=False
        )

    return {
        'model': lr_model,
        'forecast_df': forecast_df,
        'slope_per_year': lr_model.coef_[0] * 365.25,
        'r2': lr_model.score(X, y),
        'model_summary': ols_model.summary(),
        'n_points': len(df_frontier_post),
    }


def analyze_benchmark_saturation_forecasts(df_capabilities: pd.DataFrame,
                                         df_benchmarks: pd.DataFrame,
                                         output_dir: Path):
    """Forecast when models will reach 50% performance on different benchmarks"""
    print("Analyzing benchmark saturation forecasts...")
    
    # Get capability growth rate
    df_cap = prepare_model_data(df_capabilities)
    # Remove rows with NaN values in date or estimated_capability
    df_cap_clean = df_cap.dropna(subset=['date_obj', 'estimated_capability'])
    growth_stats = bootstrap_slope_analysis(df_cap_clean, 'date_obj', 'estimated_capability')
    annual_growth = growth_stats['mean_slope']
    
    # Current maximum capability
    current_max_capability = df_cap_clean['estimated_capability'].max()
    current_date = df_cap_clean['date_obj'].max()
    
    # For each benchmark, estimate when 50% performance will be reached
    saturation_forecasts = []
    
    for _, benchmark in df_benchmarks.iterrows():
        difficulty = benchmark['estimated_difficulty']
        
        # When C_m - D_b = 0, sigmoid gives 50% performance
        capability_needed = difficulty
        
        if capability_needed > current_max_capability:
            # Calculate time needed
            capability_gap = capability_needed - current_max_capability
            years_needed = capability_gap / annual_growth
            saturation_date = current_date + timedelta(days=365.25 * years_needed)
        else:
            # Already achievable
            saturation_date = current_date
            years_needed = 0
        
        saturation_forecasts.append({
            'benchmark': benchmark['benchmark_name'],
            'difficulty': difficulty,
            'capability_needed': capability_needed,
            'years_to_saturation': years_needed,
            'predicted_saturation_date': saturation_date
        })
    
    # Convert to DataFrame and sort by years needed
    saturation_df = pd.DataFrame(saturation_forecasts)
    saturation_df = saturation_df.sort_values('years_to_saturation')
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot for benchmarks that will be saturated within 5 years
    near_term = saturation_df[saturation_df['years_to_saturation'] <= 5]
    
    if len(near_term) > 0:
        y_pos = np.arange(len(near_term))
        bars = ax.barh(y_pos, near_term['years_to_saturation'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(near_term['benchmark'])
        ax.set_xlabel('Years to 50% Saturation')
        ax.set_title('Predicted Benchmark Saturation Timeline (Next 5 Years)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add current date reference
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Present')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_saturation_forecast.png", dpi=300, bbox_inches='tight')
    
    # Save full table
    saturation_df.to_csv(output_dir / "benchmark_saturation_forecasts.csv", index=False)
    
    return saturation_df


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze AI capability forecasting')
    parser.add_argument('--cutoff-date', default='2024-07-01', 
                       help='Cutoff date for forecast validation (YYYY-MM-DD)')
    parser.add_argument('--forecast-years', type=int, default=3,
                       help='Number of years to forecast ahead')
    parser.add_argument('--top-n-models', type=int, default=1,
                       help='Number of top models at release to use for frontier forecasting (default: 1 for pure frontier)')
    parser.add_argument('--label-frontier', action='store_true',
                       help='Annotate frontier model points with their model names')
    
    args = parser.parse_args()
    
    print("Starting forecasting analysis...")
    
    # Setup environment
    setup_analysis_environment()
    setup_plotting_style()
    
    # Create output directory
    output_dir = Path("outputs/forecasting")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fit the statistical model
    print("Fitting statistical model...")
    df_filtered, df_capabilities, df_benchmarks = fit_statistical_model(
        scores_df,
        anchor_mode="benchmark",
        anchor_benchmark="Winogrande",
        anchor_difficulty=0,
        anchor_slope=1
    )
    
    # Validate forecast accuracy
    validation_results = validate_forecast_accuracy(
        df_capabilities, args.cutoff_date, args.top_n_models, output_dir,
        label_frontier=args.label_frontier,
    )
    
    # Create future forecast
    forecast_results = create_future_forecast(
        df_capabilities, args.forecast_years, args.top_n_models, output_dir,
        label_frontier=args.label_frontier,
    )

    # Create post-cutoff frontier-only forecast (top-N-at-release after cutoff)
    post_cutoff_results = create_post_cutoff_frontier_forecast(
        df_capabilities,
        cutoff_date=args.cutoff_date,
        forecast_years=args.forecast_years,
        top_n_models=args.top_n_models,
        output_dir=output_dir,
        label_frontier=args.label_frontier,
    )
    
    # Analyze benchmark saturation forecasts
    saturation_forecasts = analyze_benchmark_saturation_forecasts(
        df_capabilities, df_benchmarks, output_dir
    )
    
    # Compile results
    all_results = {
        "Forecast Validation": validation_results,
        "Future Forecast": {
            "slope_per_year": forecast_results['slope_per_year'],
            "r_squared": forecast_results['r2'],
            "forecast_years": args.forecast_years
        },
        "Post-cutoff Forecast": {
            "slope_per_year": post_cutoff_results.get('slope_per_year') if post_cutoff_results else None,
            "r_squared": post_cutoff_results.get('r2') if post_cutoff_results else None,
            "n_points": post_cutoff_results.get('n_points') if post_cutoff_results else 0,
            "forecast_years": args.forecast_years,
        },
        "Benchmark Saturation": {
            "benchmarks_analyzed": len(saturation_forecasts),
            "benchmarks_within_5_years": len(saturation_forecasts[saturation_forecasts['years_to_saturation'] <= 5]),
            "next_to_saturate": saturation_forecasts.iloc[0]['benchmark'] if len(saturation_forecasts) > 0 else "None"
        }
    }
    
    save_results_summary(all_results, output_dir / "forecasting_summary.txt")
    
    # Print summary
    print("\n" + "="*60)
    print("FORECASTING ANALYSIS SUMMARY")
    print("="*60)
    
    if validation_results:
        print(f"Validation Results (cutoff: {args.cutoff_date}):")
        print(f"  MAE: {validation_results['mae']:.3f}")
        print(f"  RMSE: {validation_results['rmse']:.3f}")
        print(f"  R²: {validation_results['r2']:.3f}")
    
    print(f"\nFrontier Forecast ({args.forecast_years} years, models that were top {args.top_n_models} at release):")
    print(f"  Annual growth rate: {forecast_results['slope_per_year']:.3f}")
    print(f"  Model R²: {forecast_results['r2']:.3f}")

    if post_cutoff_results:
        print(
            f"\nPost-cutoff Frontier Forecast (after {args.cutoff_date}, top {args.top_n_models} at release, {args.forecast_years} years):"
        )
        print(f"  Points used: {post_cutoff_results['n_points']}")
        print(f"  Annual growth rate: {post_cutoff_results['slope_per_year']:.3f}")
        print(f"  Model R²: {post_cutoff_results['r2']:.3f}")
    
    if len(saturation_forecasts) > 0:
        next_benchmark = saturation_forecasts.iloc[0]
        print(f"\nNext benchmark to saturate:")
        print(f"  {next_benchmark['benchmark']} in {next_benchmark['years_to_saturation']:.1f} years")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()