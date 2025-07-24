#!/usr/bin/env python3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
Algorithmic progress analysis - converts algorithmic_progress.ipynb to a Python script

This script:
1. Analyzes compute efficiency improvements over time
2. Calculates capability improvements at fixed compute budgets
3. Integrates with Parameter-Compute Database (PCD) data
4. Generates algorithmic progress metrics and visualizations

Usage: python analyze_algorithmic_progress.py
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
from datetime import datetime
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Local imports
from data_loader import scores_df, df_model
from fit import fit_statistical_model
from analysis_utils import (
    setup_analysis_environment, setup_plotting_style,
    prepare_model_data, bootstrap_slope_analysis,
    save_results_summary
)


def load_pcd_data():
    """Load Parameter-Compute Database data if available"""
    try:
        # Try to load PCD data - this would need to be adapted based on actual PCD format
        # For now, we'll create a placeholder that matches the expected structure
        pcd_url = "https://docs.google.com/spreadsheets/d/1AAIebjNsnJj_uKALHbXNfn3_YsT6sHXtCU0q7OIPuc4/export?format=csv&gid=0"
        pcd_data = pd.read_csv(pcd_url)
        print(f"Loaded PCD data with {len(pcd_data)} entries")
        return pcd_data
    except Exception as e:
        print(f"Could not load PCD data: {e}")
        print("Creating mock PCD data for demonstration...")
        
        # Create mock data that matches expected structure
        np.random.seed(42)
        n_models = 100
        mock_pcd = pd.DataFrame({
            'model_name': [f'model_{i}' for i in range(n_models)],
            'training_compute_flops': np.random.lognormal(25, 2, n_models),
            'parameters': np.random.lognormal(20, 1.5, n_models),
            'release_date': pd.date_range('2020-01-01', '2024-12-31', periods=n_models)
        })
        return mock_pcd


def match_models_with_pcd(df_capabilities: pd.DataFrame, pcd_data: pd.DataFrame):
    """Match model capabilities with PCD compute data"""
    print("Matching models with compute data...")
    
    # Simple name matching - in practice this would need more sophisticated matching
    df_capabilities['model_clean'] = df_capabilities['model'].str.lower().str.replace('-', '_')
    pcd_data['model_clean'] = pcd_data['model_name'].str.lower().str.replace('-', '_')
    
    # Merge datasets
    merged_df = df_capabilities.merge(
        pcd_data[['model_clean', 'training_compute_flops', 'parameters']], 
        on='model_clean', 
        how='inner'
    )
    
    print(f"Successfully matched {len(merged_df)} models with compute data")
    return merged_df


def analyze_compute_efficiency(merged_df: pd.DataFrame, output_dir: Path):
    """Analyze compute efficiency improvements over time"""
    print("Analyzing compute efficiency trends...")
    
    # Prepare data
    df = merged_df.copy()
    df['date_obj'] = pd.to_datetime(df['date'])
    df['log_compute'] = np.log10(df['training_compute_flops'])
    
    # Sort by date for analysis
    df = df.sort_values('date_obj')
    
    results = {}
    
    # For each model, find later models with same/better capability but less compute
    anchor_models = []
    efficiency_slopes = []
    
    for idx, anchor_row in df.iterrows():
        # Find models released later with at least same capability
        later_models = df[
            (df['date_obj'] > anchor_row['date_obj']) &
            (df['estimated_capability'] >= anchor_row['estimated_capability']) &
            (df['training_compute_flops'] <= anchor_row['training_compute_flops'])
        ]
        
        if len(later_models) >= 10:  # Need sufficient data points
            # Calculate efficiency improvement rate
            x_data = (later_models['date_obj'] - anchor_row['date_obj']).dt.days.values
            y_data = np.log10(later_models['training_compute_flops'].values)
            
            if len(x_data) > 5:  # Minimum data points for regression
                slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                
                # Convert slope to annual rate
                annual_slope = slope * 365.25
                efficiency_improvement = -annual_slope  # Negative slope means efficiency improvement
                
                anchor_models.append(anchor_row['model'])
                efficiency_slopes.append(efficiency_improvement)
                
                results[anchor_row['model']] = {
                    'annual_efficiency_improvement': efficiency_improvement,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'n_later_models': len(later_models)
                }
    
    # Create visualization
    if anchor_models:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(anchor_models))
        bars = ax.barh(y_pos, efficiency_slopes)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(anchor_models)
        ax.set_xlabel('Annual Compute Efficiency Improvement (log10 scale)')
        ax.set_title('Compute Efficiency Improvements by Anchor Model')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_dir / "compute_efficiency_improvements.png", dpi=300, bbox_inches='tight')
        
        # Summary statistics
        if efficiency_slopes:
            mean_improvement = np.mean(efficiency_slopes)
            std_improvement = np.std(efficiency_slopes)
            results['summary'] = {
                'mean_annual_improvement': mean_improvement,
                'std_annual_improvement': std_improvement,
                'num_anchor_models': len(anchor_models)
            }
    
    return results


def analyze_capability_at_fixed_compute(merged_df: pd.DataFrame, output_dir: Path):
    """Analyze capability improvements at fixed compute budgets"""
    print("Analyzing capability improvements at fixed compute...")
    
    df = merged_df.copy()
    df['date_obj'] = pd.to_datetime(df['date'])
    df['log_compute'] = np.log10(df['training_compute_flops'])
    
    results = {}
    anchor_models = []
    capability_slopes = []
    
    # For each model, find later models with same/less compute but better capability
    for idx, anchor_row in df.iterrows():
        later_models = df[
            (df['date_obj'] > anchor_row['date_obj']) &
            (df['training_compute_flops'] <= anchor_row['training_compute_flops']) &
            (df['estimated_capability'] >= anchor_row['estimated_capability'])
        ]
        
        if len(later_models) >= 10:
            # Calculate capability improvement rate
            x_data = (later_models['date_obj'] - anchor_row['date_obj']).dt.days.values
            y_data = later_models['estimated_capability'].values
            
            if len(x_data) > 5:
                slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                
                # Convert to annual rate
                annual_slope = slope * 365.25
                
                anchor_models.append(anchor_row['model'])
                capability_slopes.append(annual_slope)
                
                results[anchor_row['model']] = {
                    'annual_capability_improvement': annual_slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'n_later_models': len(later_models)
                }
    
    # Create visualization
    if anchor_models:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(anchor_models))
        bars = ax.barh(y_pos, capability_slopes)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(anchor_models)
        ax.set_xlabel('Annual Capability Improvement (at fixed compute)')
        ax.set_title('Capability Improvements at Fixed Compute Budget')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_dir / "capability_improvements_fixed_compute.png", dpi=300, bbox_inches='tight')
        
        # Summary statistics
        if capability_slopes:
            mean_improvement = np.mean(capability_slopes)
            std_improvement = np.std(capability_slopes)
            results['summary'] = {
                'mean_annual_improvement': mean_improvement,
                'std_annual_improvement': std_improvement,
                'num_anchor_models': len(anchor_models)
            }
    
    return results


def create_compute_capability_plot(merged_df: pd.DataFrame, output_dir: Path):
    """Create scatter plot of compute vs capability over time"""
    print("Creating compute vs capability visualization...")
    
    df = merged_df.copy()
    df['date_obj'] = pd.to_datetime(df['date'])
    df['log_compute'] = np.log10(df['training_compute_flops'])
    
    # Create time-based color mapping
    date_range = (df['date_obj'].max() - df['date_obj'].min()).days
    df['date_numeric'] = (df['date_obj'] - df['date_obj'].min()).dt.days
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(df['log_compute'], df['estimated_capability'], 
                        c=df['date_numeric'], cmap='viridis', 
                        alpha=0.7, s=50)
    
    ax.set_xlabel('Log10(Training Compute FLOPs)')
    ax.set_ylabel('Estimated Capability')
    ax.set_title('Model Capability vs Training Compute Over Time')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Days since earliest model')
    
    # Add trend line
    X = df['log_compute'].values.reshape(-1, 1)
    y = df['estimated_capability'].values
    model = LinearRegression()
    model.fit(X, y)
    
    x_trend = np.linspace(df['log_compute'].min(), df['log_compute'].max(), 100)
    y_trend = model.predict(x_trend.reshape(-1, 1))
    ax.plot(x_trend, y_trend, 'r--', alpha=0.7, label='Trend line')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "compute_vs_capability.png", dpi=300, bbox_inches='tight')
    
    return model.coef_[0], model.score(X, y)


def main():
    """Main analysis function"""
    print("Starting algorithmic progress analysis...")
    
    # Setup environment
    setup_analysis_environment()
    setup_plotting_style()
    
    # Create output directory
    output_dir = Path("outputs/algorithmic_progress")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fit the statistical model to get capabilities
    print("Fitting statistical model...")
    df_filtered, df_capabilities, df_benchmarks = fit_statistical_model(
        scores_df, "Winogrande", 0, 1
    )
    
    # Load compute data
    pcd_data = load_pcd_data()
    
    # Match models with compute data
    merged_df = match_models_with_pcd(df_capabilities, pcd_data)
    
    if len(merged_df) == 0:
        print("No models could be matched with compute data. Analysis cannot proceed.")
        return
    
    # Analyze compute efficiency
    efficiency_results = analyze_compute_efficiency(merged_df, output_dir)
    
    # Analyze capability improvements at fixed compute
    capability_results = analyze_capability_at_fixed_compute(merged_df, output_dir)
    
    # Create compute vs capability plot
    compute_slope, compute_r2 = create_compute_capability_plot(merged_df, output_dir)
    
    # Save detailed results
    merged_df.to_csv(output_dir / "models_with_compute_data.csv", index=False)
    
    # Compile results summary
    all_results = {
        "Analysis Overview": {
            "total_models_with_compute": len(merged_df),
            "compute_capability_correlation": compute_r2,
            "compute_slope": compute_slope
        },
        "Compute Efficiency Analysis": efficiency_results.get('summary', {}),
        "Capability at Fixed Compute": capability_results.get('summary', {})
    }
    
    save_results_summary(all_results, output_dir / "algorithmic_progress_summary.txt")
    
    # Print summary
    print("\n" + "="*60)
    print("ALGORITHMIC PROGRESS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Models with compute data: {len(merged_df)}")
    print(f"Compute-capability correlation (R²): {compute_r2:.3f}")
    
    if 'summary' in efficiency_results:
        eff_summary = efficiency_results['summary']
        print(f"Mean annual compute efficiency improvement: {eff_summary['mean_annual_improvement']:.3f} ± {eff_summary['std_annual_improvement']:.3f}")
    
    if 'summary' in capability_results:
        cap_summary = capability_results['summary']
        print(f"Mean annual capability improvement (fixed compute): {cap_summary['mean_annual_improvement']:.3f} ± {cap_summary['std_annual_improvement']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()