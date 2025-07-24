#!/usr/bin/env python3
"""
Sanity check script for model capabilities

This script loads the model capabilities from the fitted model and creates
a bar chart showing a random subset of 15 models with their estimated capabilities.
This helps validate that the model capability estimates make intuitive sense.

Usage: python sanity_check_fit.py [--seed SEED] [--n-models N]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random


def load_model_capabilities(file_path: str) -> pd.DataFrame:
    """Load model capabilities from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} model capabilities from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        print("Please run 'uv run analyze_model_fit.py' first to generate the capabilities file.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def create_capability_sanity_check(df: pd.DataFrame, 
                                 n_models: int = 15, 
                                 seed: int = None,
                                 save_path: str = None):
    """Create a bar chart showing random subset of model capabilities"""
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Sample random models
    if len(df) < n_models:
        print(f"Warning: Only {len(df)} models available, showing all of them")
        sampled_df = df.copy()
    else:
        sampled_df = df.sample(n=n_models, random_state=seed)
    
    # Sort by capability for better visualization
    sampled_df = sampled_df.sort_values('estimated_capability', ascending=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(sampled_df))
    bars = ax.barh(y_pos, sampled_df['estimated_capability'], alpha=0.7)
    
    # Color bars based on capability level
    colors = plt.cm.viridis(sampled_df['estimated_capability'] / sampled_df['estimated_capability'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sampled_df['model'], fontsize=10)
    ax.set_xlabel('Estimated Capability', fontsize=12)
    ax.set_title(f'Random Sample of {len(sampled_df)} Model Capabilities\n(Sanity Check)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, sampled_df['estimated_capability'])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{value:.3f}', ha='left', va='center', fontsize=9)
    
    # Add statistics text box
    stats_text = f"""Sample Statistics:
Min: {sampled_df['estimated_capability'].min():.3f}
Max: {sampled_df['estimated_capability'].max():.3f}
Mean: {sampled_df['estimated_capability'].mean():.3f}
Std: {sampled_df['estimated_capability'].std():.3f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig, ax, sampled_df


def print_model_summary(df: pd.DataFrame):
    """Print summary of the sampled models"""
    print("\n" + "="*60)
    print("SAMPLED MODELS SUMMARY")
    print("="*60)
    print(f"{'Model':<35} {'Capability':<12} {'Release Date':<12}")
    print("-" * 60)
    
    for _, row in df.sort_values('estimated_capability', ascending=False).iterrows():
        model_name = row['model'][:32] + "..." if len(row['model']) > 35 else row['model']
        capability = f"{row['estimated_capability']:.3f}"
        date = row.get('date', 'Unknown')[:10] if pd.notna(row.get('date')) else 'Unknown'
        print(f"{model_name:<35} {capability:<12} {date:<12}")
    
    print("\nQuick sanity checks:")
    print("- Do the most capable models look reasonable?")
    print("- Are recent models generally more capable than older ones?")
    print("- Do the capability scores seem to match your intuition about model performance?")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Sanity check model capabilities')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--n-models', type=int, default=15,
                       help='Number of models to sample (default: 15)')
    parser.add_argument('--input-file', default='outputs/model_fit/model_capabilities.csv',
                       help='Path to model capabilities CSV file')
    parser.add_argument('--output-dir', default='outputs/sanity_check',
                       help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    print("Model Capabilities Sanity Check")
    print("=" * 40)
    
    # Load the data
    df = load_model_capabilities(args.input_file)
    if df is None:
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the sanity check plot
    fig, ax, sampled_df = create_capability_sanity_check(
        df, 
        n_models=args.n_models, 
        seed=args.seed,
        save_path=output_dir / "capability_sanity_check.png"
    )
    
    # Print summary
    print_model_summary(sampled_df)
    
    # Save the sampled data
    sampled_df.to_csv(output_dir / "sampled_models.csv", index=False)
    print(f"\nSampled model data saved to: {output_dir / 'sampled_models.csv'}")
    
    # Show the plot
    plt.show()
    
    print(f"\nSanity check complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()