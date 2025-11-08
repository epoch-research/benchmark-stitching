#!/usr/bin/env python3
"""Display a summary table of algorithmic progress results from all methods.

This script scans the output directories and collects results from:
- Buckets method
- Hierarchical median estimator
- Linear model method

It then displays them in a formatted table showing median OOMs/year of
algorithmic progress with 95% confidence intervals.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from result_collector import collect_all_results, print_summary_table


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("ALGORITHMIC PROGRESS RESULTS SUMMARY")
    print("=" * 80)
    print("\nScanning outputs/algorithmic_progress_methods/ for results...")

    results = collect_all_results()

    if not results:
        print("\n⚠️  No results found!")
        print("\nPlease run the analyses first using:")
        print("  python run_everything.py")
        print("\nOr run specific methods:")
        print("  python buckets/main.py")
        print("  python buckets/hierarchical_median.py")
        print("  python linear_model/main.py")
        sys.exit(1)

    print(f"\nFound {len(results)} result(s) across different methods and configurations.\n")

    print_summary_table(results)


if __name__ == "__main__":
    main()
