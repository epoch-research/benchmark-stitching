#!/usr/bin/env python3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
Data exploration analysis - converts data_exploration.ipynb to a Python script

This script:
1. Analyzes the temporal distribution of benchmark entries
2. Creates overlap matrices showing model coverage across benchmarks
3. Provides dataset overview statistics
4. Generates visualizations of data structure

Usage: python analyze_data_exploration.py
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
import seaborn as sns
from datetime import datetime

# Local imports
from data_loader import scores_df, df_model
from analysis_utils import setup_analysis_environment, setup_plotting_style


def analyze_temporal_distribution(scores_df: pd.DataFrame, output_dir: Path):
    """Analyze temporal distribution of benchmark entries"""
    print("Analyzing temporal distribution...")
    
    # Convert date column
    df = scores_df.copy()
    df['date_obj'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date_obj'].dt.to_period('M')
    
    # Count entries per month
    monthly_counts = df.groupby('year_month').size().reset_index(name='count')
    monthly_counts['date'] = monthly_counts['year_month'].dt.to_timestamp()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(monthly_counts['date'], monthly_counts['count'], 
                  width=20, alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Benchmark Entries')
    ax.set_title('Temporal Distribution of Benchmark Entries')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_distribution.png", dpi=300, bbox_inches='tight')
    
    return monthly_counts


def create_overlap_matrix(scores_df: pd.DataFrame, output_dir: Path):
    """Create benchmark overlap matrix"""
    print("Creating benchmark overlap matrix...")
    
    # Create model-benchmark matrix
    model_benchmark_matrix = scores_df.pivot_table(
        index='model', 
        columns='benchmark', 
        values='performance', 
        aggfunc='count',
        fill_value=0
    )
    
    # Convert to binary (1 if model evaluated on benchmark, 0 otherwise)
    model_benchmark_binary = (model_benchmark_matrix > 0).astype(int)
    
    # Calculate overlap matrix (number of common models between each pair of benchmarks)
    overlap_matrix = model_benchmark_binary.T @ model_benchmark_binary
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Mask the diagonal and upper triangle for cleaner visualization
    mask = np.triu(np.ones_like(overlap_matrix), k=1)
    
    # Create heatmap
    sns.heatmap(overlap_matrix, 
                mask=mask,
                annot=True, 
                fmt='d', 
                cmap='Blues',
                square=True,
                ax=ax,
                cbar_kws={"shrink": .8})
    
    ax.set_title('Benchmark Overlap Matrix\n(Number of Common Models)', fontsize=14)
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Benchmark')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_overlap_matrix.png", dpi=300, bbox_inches='tight')
    
    return overlap_matrix, model_benchmark_binary


def analyze_benchmark_coverage(model_benchmark_binary: pd.DataFrame, output_dir: Path):
    """Analyze benchmark coverage statistics"""
    print("Analyzing benchmark coverage...")
    
    # Models per benchmark
    models_per_benchmark = model_benchmark_binary.sum(axis=0).sort_values(ascending=False)
    
    # Benchmarks per model
    benchmarks_per_model = model_benchmark_binary.sum(axis=1).sort_values(ascending=False)
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Top benchmarks by model count
    top_benchmarks = models_per_benchmark.head(15)
    ax1.barh(range(len(top_benchmarks)), top_benchmarks.values)
    ax1.set_yticks(range(len(top_benchmarks)))
    ax1.set_yticklabels(top_benchmarks.index)
    ax1.set_xlabel('Number of Models')
    ax1.set_title('Top 15 Benchmarks by Model Coverage')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Top models by benchmark count
    top_models = benchmarks_per_model.head(15)
    ax2.barh(range(len(top_models)), top_models.values)
    ax2.set_yticks(range(len(top_models)))
    ax2.set_yticklabels(top_models.index)
    ax2.set_xlabel('Number of Benchmarks')
    ax2.set_title('Top 15 Models by Benchmark Coverage')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_coverage_analysis.png", dpi=300, bbox_inches='tight')
    
    return models_per_benchmark, benchmarks_per_model


def generate_dataset_overview(scores_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive dataset overview"""
    print("Generating dataset overview...")
    
    # Basic statistics
    total_entries = len(scores_df)
    unique_models = scores_df['model'].nunique()
    unique_benchmarks = scores_df['benchmark'].nunique()
    
    # Date range
    dates = pd.to_datetime(scores_df['date'])
    date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
    
    # Performance statistics
    perf_stats = scores_df['performance'].describe()
    
    # Source breakdown
    source_counts = scores_df['source'].value_counts() if 'source' in scores_df.columns else None
    
    # Optimization status breakdown
    optim_counts = scores_df['optimized'].value_counts() if 'optimized' in scores_df.columns else None
    
    # Create benchmark list with statistics
    benchmark_stats = scores_df.groupby('benchmark').agg({
        'model': 'nunique',
        'performance': ['mean', 'std', 'min', 'max']
    }).round(4)
    benchmark_stats.columns = ['num_models', 'mean_perf', 'std_perf', 'min_perf', 'max_perf']
    benchmark_stats = benchmark_stats.sort_values('num_models', ascending=False)
    
    # Save overview to file
    overview_path = output_dir / "dataset_overview.txt"
    with open(overview_path, 'w') as f:
        f.write("BENCHMARK STITCHING DATASET OVERVIEW\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("BASIC STATISTICS:\n")
        f.write(f"  Total entries: {total_entries:,}\n")
        f.write(f"  Unique models: {unique_models}\n")
        f.write(f"  Unique benchmarks: {unique_benchmarks}\n")
        f.write(f"  Date range: {date_range}\n\n")
        
        f.write("PERFORMANCE STATISTICS:\n")
        for stat, value in perf_stats.items():
            f.write(f"  {stat}: {value:.4f}\n")
        f.write("\n")
        
        if source_counts is not None:
            f.write("SOURCE BREAKDOWN:\n")
            for source, count in source_counts.items():
                f.write(f"  {source}: {count} ({count/total_entries*100:.1f}%)\n")
            f.write("\n")
        
        if optim_counts is not None:
            f.write("OPTIMIZATION STATUS:\n")
            for status, count in optim_counts.items():
                f.write(f"  {'Optimized' if status else 'Not optimized'}: {count} ({count/total_entries*100:.1f}%)\n")
            f.write("\n")
        
        f.write("BENCHMARK STATISTICS (Top 15 by model count):\n")
        f.write(f"{'Benchmark':<30} {'Models':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
        f.write("-" * 80 + "\n")
        for benchmark, row in benchmark_stats.head(15).iterrows():
            f.write(f"{benchmark:<30} {row['num_models']:<8} {row['mean_perf']:<8.3f} "
                   f"{row['std_perf']:<8.3f} {row['min_perf']:<8.3f} {row['max_perf']:<8.3f}\n")
    
    # Save detailed benchmark stats
    benchmark_stats.to_csv(output_dir / "benchmark_statistics.csv")
    
    return {
        'total_entries': total_entries,
        'unique_models': unique_models,
        'unique_benchmarks': unique_benchmarks,
        'date_range': date_range,
        'performance_stats': perf_stats,
        'benchmark_stats': benchmark_stats
    }


def main():
    """Main analysis function"""
    print("Starting data exploration analysis...")
    
    # Setup environment
    setup_analysis_environment()
    setup_plotting_style()
    
    # Create output directory
    output_dir = Path("outputs/data_exploration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset overview
    overview_stats = generate_dataset_overview(scores_df, output_dir)
    
    # Analyze temporal distribution
    monthly_counts = analyze_temporal_distribution(scores_df, output_dir)
    
    # Create overlap matrix
    overlap_matrix, model_benchmark_binary = create_overlap_matrix(scores_df, output_dir)
    
    # Analyze benchmark coverage
    models_per_benchmark, benchmarks_per_model = analyze_benchmark_coverage(
        model_benchmark_binary, output_dir
    )
    
    # Summary statistics
    print("\n" + "="*60)
    print("DATA EXPLORATION SUMMARY")
    print("="*60)
    print(f"Total entries: {overview_stats['total_entries']:,}")
    print(f"Unique models: {overview_stats['unique_models']}")
    print(f"Unique benchmarks: {overview_stats['unique_benchmarks']}")
    print(f"Date range: {overview_stats['date_range']}")
    print(f"Average performance: {overview_stats['performance_stats']['mean']:.3f}")
    
    print(f"\nMost covered benchmark: {models_per_benchmark.index[0]} ({models_per_benchmark.iloc[0]} models)")
    print(f"Most evaluated model: {benchmarks_per_model.index[0]} ({benchmarks_per_model.iloc[0]} benchmarks)")
    
    print(f"\nResults saved to: {output_dir}")
    print("Generated files:")
    print("  - dataset_overview.txt")
    print("  - benchmark_statistics.csv")
    print("  - temporal_distribution.png")
    print("  - benchmark_overlap_matrix.png")
    print("  - benchmark_coverage_analysis.png")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()