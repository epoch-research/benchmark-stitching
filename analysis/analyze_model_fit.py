#!/usr/bin/env python3
"""
Core model fitting analysis - converts model_fit.ipynb to a Python script

This script:
1. Fits the statistical model to benchmark data
2. Analyzes model capabilities and benchmark difficulties over time
3. Generates capability growth rate estimates with bootstrap confidence intervals
4. Creates visualizations and saves results

Usage: python analyze_model_fit.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Local imports
from data_loader import scores_df, df_model
from fit import fit_statistical_model
from analysis_utils import (
    setup_analysis_environment, setup_plotting_style,
    prepare_model_data, prepare_benchmark_data,
    bootstrap_slope_analysis, calculate_capability_growth_rate,
    plot_capabilities_over_time, plot_benchmark_difficulties,
    save_results_summary
)


def main():
    """Main analysis function"""
    print("Starting model fit analysis...")
    
    # Setup environment
    setup_analysis_environment()
    setup_plotting_style()
    
    # Create output directory
    output_dir = Path("outputs/model_fit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fit the statistical model
    print("Fitting statistical model...")
    anchor_benchmark = "Winogrande"
    anchor_difficulty = 0
    anchor_slope = 1
    
    df_filtered, df_capabilities, df_benchmarks = fit_statistical_model(
        scores_df, anchor_benchmark, anchor_difficulty, anchor_slope
    )
    
    print(f"Model fitted with {len(df_capabilities)} models and {len(df_benchmarks)} benchmarks")
    
    # Prepare data
    df_capabilities_sorted = prepare_model_data(df_capabilities)
    df_benchmarks_sorted = prepare_benchmark_data(df_benchmarks)
    
    # Calculate capability growth rate
    print("Calculating capability growth rate...")
    growth_stats = calculate_capability_growth_rate(df_capabilities_sorted)
    
    print(f"Mean capability growth rate: {growth_stats['mean_slope']:.4f} ± {growth_stats['std_slope']:.4f} per year")
    print(f"95% CI: [{growth_stats['ci_2_5']:.4f}, {growth_stats['ci_97_5']:.4f}]")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Model capabilities over time
    fig1, ax1 = plot_capabilities_over_time(
        df_capabilities_sorted,
        title="Model Capabilities Over Time",
        save_path=output_dir / "capabilities_over_time.png"
    )
    
    # 2. Combined capabilities and difficulties over time
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top plot: Model capabilities
    ax2a.scatter(df_capabilities_sorted['date_obj'], 
                df_capabilities_sorted['estimated_capability'], 
                alpha=0.7, s=30, label='Model Capabilities')
    ax2a.set_ylabel('Estimated Capability')
    ax2a.set_title('Model Capabilities and Benchmark Difficulties Over Time')
    ax2a.grid(True, alpha=0.3)
    ax2a.legend()
    
    # Bottom plot: Benchmark difficulties
    ax2b.scatter(df_benchmarks_sorted['benchmark_release_date'], 
                df_benchmarks_sorted['estimated_difficulty'], 
                alpha=0.7, s=30, color='red', label='Benchmark Difficulties')
    ax2b.set_xlabel('Date')
    ax2b.set_ylabel('Estimated Difficulty')
    ax2b.grid(True, alpha=0.3)
    ax2b.legend()
    
    # Format dates
    for ax in [ax2a, ax2b]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "capabilities_and_difficulties_time.png", dpi=300, bbox_inches='tight')
    
    # 3. Model capabilities ranking
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    top_models = df_capabilities_sorted.nlargest(20, 'estimated_capability')
    
    y_pos = np.arange(len(top_models))
    bars = ax3.barh(y_pos, top_models['estimated_capability'])
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_models['model'])
    ax3.set_xlabel('Estimated Capability')
    ax3.set_title('Top 20 Model Capabilities')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_capabilities_ranking.png", dpi=300, bbox_inches='tight')
    
    # 4. Benchmark difficulties ranking
    fig4, ax4 = plot_benchmark_difficulties(
        df_benchmarks_sorted,
        title="Top 15 Most Difficult Benchmarks",
        top_n=15,
        save_path=output_dir / "benchmark_difficulties_ranking.png"
    )
    
    # Save data
    print("Saving data...")
    df_capabilities_sorted.to_csv(output_dir / "model_capabilities.csv", index=False)
    df_benchmarks_sorted.to_csv(output_dir / "benchmark_difficulties.csv", index=False)
    
    # Save results summary
    results = {
        "Model Fit Summary": {
            "num_models": len(df_capabilities),
            "num_benchmarks": len(df_benchmarks),
            "anchor_benchmark": anchor_benchmark,
            "anchor_difficulty": anchor_difficulty,
            "anchor_slope": anchor_slope
        },
        "Capability Growth Analysis": {
            "mean_growth_rate_per_year": growth_stats['mean_slope'],
            "std_growth_rate": growth_stats['std_slope'],
            "ci_2_5_percent": growth_stats['ci_2_5'],
            "ci_97_5_percent": growth_stats['ci_97_5'],
            "bootstrap_samples": len(growth_stats['slopes'])
        },
        "Top Models": {
            model: capability 
            for model, capability in zip(
                top_models['model'].head(5), 
                top_models['estimated_capability'].head(5)
            )
        },
        "Most Difficult Benchmarks": {
            benchmark: difficulty 
            for benchmark, difficulty in zip(
                df_benchmarks_sorted.nlargest(5, 'estimated_difficulty')['benchmark_name'],
                df_benchmarks_sorted.nlargest(5, 'estimated_difficulty')['estimated_difficulty']
            )
        }
    }
    
    save_results_summary(results, output_dir / "analysis_summary.txt")
    
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"Key findings:")
    print(f"  - Capability growth rate: {growth_stats['mean_slope']:.3f} ± {growth_stats['std_slope']:.3f} per year")
    print(f"  - Most capable model: {top_models.iloc[0]['model']} ({top_models.iloc[0]['estimated_capability']:.3f})")
    print(f"  - Most difficult benchmark: {df_benchmarks_sorted.nlargest(1, 'estimated_difficulty').iloc[0]['benchmark_name']}")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()