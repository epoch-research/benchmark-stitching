"""
Mini sweep to compare 2x vs 4x acceleration with 20 models/year, 5 benchmarks/year.
"""

import numpy as np
import pandas as pd
from analysis.analyze_software_singularity import (
    generate_data,
    estimated_capabilities,
    estimate_detection_for_single_trajectory
)

# Configuration
models_per_year = 20
benchmarks_per_year = 5
horizon_years = 6
cutoff_year = 2027
n_simulations = 10  # More simulations to see the pattern
random_seed_base = 42

print("="*80)
print("MINI SWEEP: Comparing 2x vs 4x acceleration (HIGH NOISE)")
print("="*80)
print(f"Configuration: {models_per_year} models/year, {benchmarks_per_year} benchmarks/year")
print(f"Noise levels: error_std=0.05, noise_std_model=0.4, noise_std_bench=0.4")
print(f"Running {n_simulations} simulations for each acceleration factor")
print()

results_2x = []
results_4x = []

for accel_factor, results_list in [(2.0, results_2x), (4.0, results_4x)]:
    print(f"\n{'-'*80}")
    print(f"Testing {accel_factor}x acceleration:")
    print(f"{'-'*80}")

    for sim_idx in range(n_simulations):
        seed = random_seed_base + sim_idx + int(accel_factor * 100)

        # Generate data with unique seed
        num_models = int(models_per_year * horizon_years)
        num_benchmarks = int(benchmarks_per_year * horizon_years)

        models_df, benchmarks_df, scores_df = generate_data(
            num_models=num_models,
            num_benchmarks=num_benchmarks,
            speedup_factor_model=accel_factor,
            time_range_start=2024,
            time_range_end=2024 + horizon_years,
            cutoff_year=cutoff_year,
            frac_eval=0.25,
            error_std=0.05,  # Increased from 0.025 (2x more noise in scores)
            elo_change=3.5,
            noise_std_model=0.4,  # Increased from 0.2 (2x more noise in capabilities)
            noise_std_bench=0.4,  # Increased from 0.2 (2x more noise in difficulties)
            frac_accelerate_models=1.0,
            random_seed=seed
        )

        # Estimate capabilities
        try:
            df_est = estimated_capabilities(models_df, benchmarks_df, scores_df)

            # Run detection
            result = estimate_detection_for_single_trajectory(
                models_df,
                benchmarks_df,
                scores_df,
                cutoff_year=cutoff_year,
                acceleration_factor=2.0,  # Looking for 2x or more
                verbose=False
            )

            if result['detected']:
                results_list.append({
                    'sim': sim_idx + 1,
                    'seed': seed,
                    'detected': True,
                    'years_to_detect': result['years_to_detect'],
                    'ratio': result['ratio']
                })
                print(f"  Sim {sim_idx+1:2d}: ✓ Detected at {result['years_to_detect']:.3f} years (ratio: {result['ratio']:.2f}x)")
            else:
                results_list.append({
                    'sim': sim_idx + 1,
                    'seed': seed,
                    'detected': False,
                    'years_to_detect': None,
                    'ratio': None
                })
                print(f"  Sim {sim_idx+1:2d}: ✗ Not detected")
        except Exception as e:
            print(f"  Sim {sim_idx+1:2d}: ✗ Error - {e}")
            results_list.append({
                'sim': sim_idx + 1,
                'seed': seed,
                'detected': False,
                'years_to_detect': None,
                'ratio': None
            })

# Summary statistics
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

df_2x = pd.DataFrame(results_2x)
df_4x = pd.DataFrame(results_4x)

print(f"\n2x Acceleration:")
print(f"  Detection rate: {df_2x['detected'].sum()}/{len(df_2x)} = {df_2x['detected'].mean():.1%}")
if df_2x['detected'].any():
    detected_2x = df_2x[df_2x['detected']]
    print(f"  Mean time to detect: {detected_2x['years_to_detect'].mean():.3f} years")
    print(f"  Std time to detect: {detected_2x['years_to_detect'].std():.3f} years")
    print(f"  Min time to detect: {detected_2x['years_to_detect'].min():.3f} years")
    print(f"  Max time to detect: {detected_2x['years_to_detect'].max():.3f} years")

print(f"\n4x Acceleration:")
print(f"  Detection rate: {df_4x['detected'].sum()}/{len(df_4x)} = {df_4x['detected'].mean():.1%}")
if df_4x['detected'].any():
    detected_4x = df_4x[df_4x['detected']]
    print(f"  Mean time to detect: {detected_4x['years_to_detect'].mean():.3f} years")
    print(f"  Std time to detect: {detected_4x['years_to_detect'].std():.3f} years")
    print(f"  Min time to detect: {detected_4x['years_to_detect'].min():.3f} years")
    print(f"  Max time to detect: {detected_4x['years_to_detect'].max():.3f} years")

# Compare
if df_2x['detected'].any() and df_4x['detected'].any():
    detected_2x = df_2x[df_2x['detected']]['years_to_detect']
    detected_4x = df_4x[df_4x['detected']]['years_to_detect']
    diff = detected_2x.mean() - detected_4x.mean()
    print(f"\nDifference (2x - 4x): {diff:+.3f} years")
    print(f"  → 4x is detected {'faster' if diff > 0 else 'slower'} by {abs(diff):.3f} years on average")

print("\n" + "="*80)
