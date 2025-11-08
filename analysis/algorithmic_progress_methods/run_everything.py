#!/usr/bin/env python3
"""Run all algorithmic progress analysis methods with various configurations.

This script executes all analysis methods (buckets, linear model, hierarchical median)
with different data sources and model filtering options to provide a comprehensive
view of algorithmic progress estimates.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import argparse


# Define all configurations to run
CONFIGURATIONS = [
    {
        "name": "Internal data (all models)",
        "flags": [],
    },
    {
        "name": "Internal data (excluding distilled)",
        "flags": ["--exclude-distilled"],
    },
    {
        "name": "Internal data (excluding all distilled)",
        "flags": ["--exclude-distilled", "--include-low-confidence"],
    },
    {
        "name": "Website data (all models)",
        "flags": ["--use-website-data"],
    },
    {
        "name": "Website data (excluding distilled)",
        "flags": ["--use-website-data", "--exclude-distilled"],
    },
    {
        "name": "Website data (excluding all distilled)",
        "flags": ["--use-website-data", "--exclude-distilled", "--include-low-confidence"],
    },
]

# Linear model also supports frontier-only
LINEAR_MODEL_EXTRA_CONFIGS = [
    {
        "name": "Internal data (frontier only)",
        "flags": ["--frontier-only"],
    },
    {
        "name": "Internal data (frontier only, excluding distilled)",
        "flags": ["--frontier-only", "--exclude-distilled"],
    },
]


def run_command(cmd: List[str], description: str, dry_run: bool = False) -> bool:
    """Execute a command with proper error handling.

    Args:
        cmd: Command and arguments as list
        description: Human-readable description of what's running
        dry_run: If True, print command without executing

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("(DRY RUN - not actually executing)")
        return True

    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent.parent)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False


def run_buckets_method(config: Dict, dry_run: bool = False,
                      bucket_size: float = None, label_points: bool = False) -> bool:
    """Run the buckets method with specified configuration.

    Args:
        config: Configuration dictionary with 'name' and 'flags'
        dry_run: If True, print commands without executing
        bucket_size: Optional specific bucket size (default uses method default)
        label_points: If True, label data points with model names

    Returns:
        bool: True if successful
    """
    cmd = [sys.executable, "analysis/algorithmic_progress_methods/buckets/main.py"]
    cmd.extend(config["flags"])

    if bucket_size is not None:
        cmd.extend(["--eci-bucket-size", str(bucket_size)])

    if label_points:
        cmd.append("--label-points")

    return run_command(cmd, f"Buckets Method - {config['name']}", dry_run)


def run_buckets_sensitivity(config: Dict, dry_run: bool = False,
                           n_bucket_sizes: int = 5) -> bool:
    """Run buckets method sensitivity analysis.

    Args:
        config: Configuration dictionary with 'name' and 'flags'
        dry_run: If True, print commands without executing
        n_bucket_sizes: Number of bucket sizes to test

    Returns:
        bool: True if successful
    """
    cmd = [sys.executable, "analysis/algorithmic_progress_methods/buckets/main.py"]
    cmd.extend(config["flags"])
    cmd.extend(["--sweep-bucket-sizes", "--n-bucket-sizes", str(n_bucket_sizes)])

    return run_command(
        cmd,
        f"Buckets Sensitivity Analysis - {config['name']}",
        dry_run
    )


def run_hierarchical_median(config: Dict, dry_run: bool = False,
                           n_bucket_sizes: int = 7, n_iter: int = 30000) -> bool:
    """Run hierarchical median estimator.

    Args:
        config: Configuration dictionary with 'name' and 'flags'
        dry_run: If True, print commands without executing
        n_bucket_sizes: Number of bucket sizes to sweep
        n_iter: Number of Gibbs iterations

    Returns:
        bool: True if successful
    """
    cmd = [sys.executable, "analysis/algorithmic_progress_methods/buckets/hierarchical_median.py"]
    cmd.extend(config["flags"])
    cmd.extend([
        "--n-bucket-sizes", str(n_bucket_sizes),
        "--n-iter", str(n_iter)
    ])

    return run_command(
        cmd,
        f"Hierarchical Median - {config['name']}",
        dry_run
    )


def run_linear_model(config: Dict, dry_run: bool = False,
                    show_predicted_frontier: bool = False,
                    label_points: bool = False) -> bool:
    """Run the linear model method.

    Args:
        config: Configuration dictionary with 'name' and 'flags'
        dry_run: If True, print commands without executing
        show_predicted_frontier: If True, show predicted Pareto frontier
        label_points: If True, label points with ECI values

    Returns:
        bool: True if successful
    """
    cmd = [sys.executable, "analysis/algorithmic_progress_methods/linear_model/main.py"]
    cmd.extend(config["flags"])

    if show_predicted_frontier:
        cmd.append("--show-predicted-frontier")

    if label_points:
        cmd.append("--label-points")

    return run_command(cmd, f"Linear Model - {config['name']}", dry_run)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run all algorithmic progress methods with various configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything
  python run_everything.py

  # Run only buckets method
  python run_everything.py --buckets-only

  # Run only linear model
  python run_everything.py --linear-only

  # Run with minimal iterations (faster, for testing)
  python run_everything.py --quick

  # Preview commands without executing
  python run_everything.py --dry-run

  # Run only internal data configurations (skip website data)
  python run_everything.py --skip-website
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing them')
    parser.add_argument('--buckets-only', action='store_true',
                       help='Only run buckets method (skip linear model)')
    parser.add_argument('--linear-only', action='store_true',
                       help='Only run linear model (skip buckets method)')
    parser.add_argument('--hierarchical-only', action='store_true',
                       help='Only run hierarchical median estimator')
    parser.add_argument('--skip-sensitivity', action='store_true',
                       help='Skip bucket size sensitivity analysis')
    parser.add_argument('--skip-hierarchical', action='store_true',
                       help='Skip hierarchical median estimator')
    parser.add_argument('--skip-website', action='store_true',
                       help='Skip website data configurations')
    parser.add_argument('--quick', action='store_true',
                       help='Use reduced iterations for faster execution (testing mode)')
    parser.add_argument('--label-points', action='store_true',
                       help='Label data points in plots')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue running even if some analyses fail')

    args = parser.parse_args()

    # Determine which methods to run
    run_buckets = not args.linear_only and not args.hierarchical_only
    run_linear = not args.buckets_only and not args.hierarchical_only
    run_hierarchical = not args.buckets_only and not args.linear_only and not args.skip_hierarchical

    # If hierarchical-only is set, only run that
    if args.hierarchical_only:
        run_buckets = False
        run_linear = False
        run_hierarchical = True

    # Filter configurations based on skip-website flag
    configs = CONFIGURATIONS
    if args.skip_website:
        configs = [c for c in configs if "--use-website-data" not in c["flags"]]

    # Adjust iterations for quick mode
    n_bucket_sizes = 3 if args.quick else 5
    hierarchical_n_bucket_sizes = 5 if args.quick else 7
    hierarchical_n_iter = 10000 if args.quick else 30000

    # Track results
    total_runs = 0
    successful_runs = 0
    failed_runs = []

    print("\n" + "=" * 80)
    print("ALGORITHMIC PROGRESS METHODS - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"\nConfigurations to run: {len(configs)}")
    print(f"Buckets method: {'Yes' if run_buckets else 'No'}")
    print(f"Linear model: {'Yes' if run_linear else 'No'}")
    print(f"Hierarchical median: {'Yes' if run_hierarchical else 'No'}")
    print(f"Sensitivity analysis: {'No' if args.skip_sensitivity else 'Yes'}")
    if args.quick:
        print("\n⚡ QUICK MODE: Using reduced iterations for faster execution")
    if args.dry_run:
        print("\n🔍 DRY RUN MODE: Commands will be printed but not executed")
    print()

    # Run buckets method
    if run_buckets:
        print("\n" + "🔲" * 40)
        print("BUCKETS METHOD")
        print("🔲" * 40)

        for config in configs:
            total_runs += 1
            success = run_buckets_method(
                config,
                dry_run=args.dry_run,
                label_points=args.label_points
            )
            if success:
                successful_runs += 1
            else:
                failed_runs.append(f"Buckets - {config['name']}")
                if not args.continue_on_error:
                    print("\n❌ Stopping due to error. Use --continue-on-error to continue.")
                    sys.exit(1)

        # Run sensitivity analysis
        if not args.skip_sensitivity:
            print("\n" + "📊" * 40)
            print("BUCKETS SENSITIVITY ANALYSIS")
            print("📊" * 40)

            for config in configs:
                total_runs += 1
                success = run_buckets_sensitivity(
                    config,
                    dry_run=args.dry_run,
                    n_bucket_sizes=n_bucket_sizes
                )
                if success:
                    successful_runs += 1
                else:
                    failed_runs.append(f"Buckets Sensitivity - {config['name']}")
                    if not args.continue_on_error:
                        print("\n❌ Stopping due to error. Use --continue-on-error to continue.")
                        sys.exit(1)

    # Run hierarchical median estimator
    if run_hierarchical:
        print("\n" + "📈" * 40)
        print("HIERARCHICAL MEDIAN ESTIMATOR")
        print("📈" * 40)

        for config in configs:
            total_runs += 1
            success = run_hierarchical_median(
                config,
                dry_run=args.dry_run,
                n_bucket_sizes=hierarchical_n_bucket_sizes,
                n_iter=hierarchical_n_iter
            )
            if success:
                successful_runs += 1
            else:
                failed_runs.append(f"Hierarchical Median - {config['name']}")
                if not args.continue_on_error:
                    print("\n❌ Stopping due to error. Use --continue-on-error to continue.")
                    sys.exit(1)

    # Run linear model
    if run_linear:
        print("\n" + "📉" * 40)
        print("LINEAR MODEL METHOD")
        print("📉" * 40)

        # Include extra frontier-only configurations for linear model
        linear_configs = configs + LINEAR_MODEL_EXTRA_CONFIGS
        if args.skip_website:
            linear_configs = [c for c in linear_configs
                            if "--use-website-data" not in c["flags"]]

        for config in linear_configs:
            total_runs += 1
            success = run_linear_model(
                config,
                dry_run=args.dry_run,
                label_points=args.label_points
            )
            if success:
                successful_runs += 1
            else:
                failed_runs.append(f"Linear Model - {config['name']}")
                if not args.continue_on_error:
                    print("\n❌ Stopping due to error. Use --continue-on-error to continue.")
                    sys.exit(1)

    # Print summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total analyses: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {len(failed_runs)}")

    if failed_runs:
        print("\n❌ Failed analyses:")
        for failure in failed_runs:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print("\n✅ All analyses completed successfully!")
        print("\nResults are organized by method and configuration in:")
        print("  outputs/algorithmic_progress_methods/")
        print("\nDirectory structure:")
        print("  buckets/")
        print("    internal_with_distilled_all_models/")
        print("    internal_no_distilled_all_models/")
        print("    website_with_distilled_all_models/")
        print("    ...")
        print("  linear_model/")
        print("    internal_with_distilled_all_models/")
        print("    internal_with_distilled_frontier_only/")
        print("    ...")


if __name__ == "__main__":
    main()
