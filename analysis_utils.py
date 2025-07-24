"""
Shared utilities for benchmark stitching analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional
import os


def setup_analysis_environment():
    """Set up the analysis environment by changing to project root"""
    current_dir = Path.cwd()
    if current_dir.name == "notebooks":
        os.chdir(current_dir.parent)
    return Path.cwd()


def prepare_model_data(df_capabilities: pd.DataFrame) -> pd.DataFrame:
    """Prepare model capabilities data with datetime conversion"""
    df = df_capabilities.copy()
    df['date_obj'] = pd.to_datetime(df['date'])
    return df.sort_values('date_obj')


def prepare_benchmark_data(df_benchmarks: pd.DataFrame) -> pd.DataFrame:
    """Prepare benchmark data with datetime conversion"""
    df = df_benchmarks.copy()
    df['benchmark_release_date'] = pd.to_datetime(df['benchmark_release_date'])
    return df.sort_values('benchmark_release_date')


def bootstrap_slope_analysis(df: pd.DataFrame, 
                           x_col: str, 
                           y_col: str, 
                           n_bootstrap: int = 10000,
                           cutoff_date: Optional[str] = None) -> dict:
    """
    Perform bootstrap analysis to estimate slope with confidence intervals
    
    Args:
        df: DataFrame with data
        x_col: Column name for x variable (typically date)
        y_col: Column name for y variable (typically capability)
        n_bootstrap: Number of bootstrap samples
        cutoff_date: Optional cutoff date for filtering data
        
    Returns:
        Dictionary with slope statistics
    """
    from sklearn.linear_model import LinearRegression
    
    # Filter data if cutoff provided
    if cutoff_date:
        df = df[df[x_col] <= cutoff_date].copy()
    
    # Convert dates to numeric if needed
    if df[x_col].dtype == 'datetime64[ns]':
        x_numeric = (df[x_col] - df[x_col].min()).dt.days
    else:
        x_numeric = df[x_col]
    
    # Bootstrap sampling
    slopes = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(df), size=len(df), replace=True)
        x_sample = x_numeric.iloc[sample_idx].values.reshape(-1, 1)
        y_sample = df[y_col].iloc[sample_idx].values
        
        model = LinearRegression()
        model.fit(x_sample, y_sample)
        
        # Convert slope to per-year if using days
        slope = model.coef_[0]
        if df[x_col].dtype == 'datetime64[ns]':
            slope *= 365.25  # Convert from per-day to per-year
            
        slopes.append(slope)
    
    slopes = np.array(slopes)
    
    return {
        'mean_slope': np.mean(slopes),
        'std_slope': np.std(slopes),
        'ci_2_5': np.percentile(slopes, 2.5),
        'ci_97_5': np.percentile(slopes, 97.5),
        'ci_5': np.percentile(slopes, 5),
        'ci_95': np.percentile(slopes, 95),
        'slopes': slopes
    }


def plot_capabilities_over_time(df_capabilities: pd.DataFrame, 
                              title: str = "Model Capabilities Over Time",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)):
    """Plot model capabilities over time with trend line"""
    df = prepare_model_data(df_capabilities)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    scatter = ax.scatter(df['date_obj'], df['estimated_capability'], 
                        alpha=0.7, s=50)
    
    # Add model labels for recent/important models
    important_models = df.nlargest(10, 'estimated_capability')
    for _, row in important_models.iterrows():
        ax.annotate(row['model'], 
                   (row['date_obj'], row['estimated_capability']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    # Trend line
    from sklearn.linear_model import LinearRegression
    x_numeric = (df['date_obj'] - df['date_obj'].min()).dt.days.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_numeric, df['estimated_capability'])
    
    trend_line = model.predict(x_numeric)
    ax.plot(df['date_obj'], trend_line, 'r--', alpha=0.7, label='Trend')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated Capability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_benchmark_difficulties(df_benchmarks: pd.DataFrame,
                              top_n: int = 20,
                              title: str = "Benchmark Difficulties",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)):
    """Plot benchmark difficulties ranking"""
    df = df_benchmarks.nlargest(top_n, 'estimated_difficulty')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['estimated_difficulty'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['benchmark_name'])
    ax.set_xlabel('Estimated Difficulty')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def filter_by_date(df: pd.DataFrame, 
                  date_col: str, 
                  cutoff_date: str,
                  before: bool = True) -> pd.DataFrame:
    """Filter dataframe by date"""
    cutoff = pd.to_datetime(cutoff_date)
    if before:
        return df[df[date_col] <= cutoff].copy()
    else:
        return df[df[date_col] > cutoff].copy()


def calculate_capability_growth_rate(df_capabilities: pd.DataFrame,
                                   cutoff_date: Optional[str] = None) -> dict:
    """Calculate capability growth rate with bootstrap confidence intervals"""
    df = prepare_model_data(df_capabilities)
    
    if cutoff_date:
        df = filter_by_date(df, 'date_obj', cutoff_date, before=True)
    
    return bootstrap_slope_analysis(df, 'date_obj', 'estimated_capability')


def save_results_summary(results: dict, output_path: str):
    """Save analysis results to a text file"""
    with open(output_path, 'w') as f:
        f.write("Benchmark Stitching Analysis Results\n")
        f.write("=" * 40 + "\n\n")
        
        for section, data in results.items():
            f.write(f"{section}:\n")
            f.write("-" * len(section) + "\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {data}\n")
            f.write("\n")


def setup_plotting_style():
    """Set up consistent plotting style"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3