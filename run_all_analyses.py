#!/usr/bin/env python3
"""
Main runner script for all benchmark stitching analyses

This script runs all the converted analysis scripts in sequence and provides
a unified interface for executing the complete benchmark stitching analysis pipeline.

Usage: 
    python run_all_analyses.py [--analyses ANALYSIS1,ANALYSIS2] [--output-dir DIR]
    
Available analyses:
    - data_exploration: Dataset overview and structure analysis
    - model_fit: Core statistical model fitting and capability ranking
    - algorithmic_progress: Compute efficiency and algorithmic progress analysis
    - forecasting: Future capability forecasting and validation
    - robustness: Robustness testing and sensitivity analysis
    - all: Run all analyses (default)

Examples:
    python run_all_analyses.py
    python run_all_analyses.py --analyses model_fit,forecasting
    python run_all_analyses.py --output-dir custom_outputs
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import time


def run_analysis(script_name: str, args: list = None) -> dict:
    """Run a single analysis script and return execution info"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Build command
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    try:
        # Run the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        execution_time = time.time() - start_time
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        
        if success:
            print(f"✓ {script_name} completed successfully in {execution_time:.1f}s")
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
        
        return {
            'script': script_name,
            'success': success,
            'execution_time': execution_time,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"✗ {script_name} failed with exception: {e}")
        return {
            'script': script_name,
            'success': False,
            'execution_time': execution_time,
            'return_code': -1,
            'error': str(e)
        }


def setup_output_directory(base_dir: str) -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"batch_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_execution_summary(results: list, output_dir: Path):
    """Create summary of all analysis executions"""
    summary_path = output_dir / "execution_summary.txt"
    
    total_time = sum(r['execution_time'] for r in results)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    with open(summary_path, 'w') as f:
        f.write("BENCHMARK STITCHING ANALYSIS EXECUTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Total Execution Time: {total_time:.1f} seconds\n")
        f.write(f"Analyses Run: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        
        f.write("INDIVIDUAL ANALYSIS RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        for result in results:
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            f.write(f"{result['script']:<25} | {status:<10} | {result['execution_time']:.1f}s\n")
        
        f.write("\nDETAILED RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        for result in results:
            f.write(f"\n{result['script']}:\n")
            f.write(f"  Success: {result['success']}\n")
            f.write(f"  Execution Time: {result['execution_time']:.1f}s\n")
            f.write(f"  Return Code: {result['return_code']}\n")
            
            if not result['success']:
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                if result.get('stderr'):
                    f.write(f"  STDERR: {result['stderr'][:500]}...\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run benchmark stitching analyses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available analyses:
  data_exploration     - Dataset overview and structure analysis
  model_fit           - Core statistical model fitting and capability ranking  
  algorithmic_progress - Compute efficiency and algorithmic progress analysis
  forecasting         - Future capability forecasting and validation
  all                 - Run all analyses (default)

Examples:
  python run_all_analyses.py
  python run_all_analyses.py --analyses model_fit,forecasting
  python run_all_analyses.py --output-dir custom_outputs
        """
    )
    
    parser.add_argument('--analyses', 
                       default='all',
                       help='Comma-separated list of analyses to run (default: all)')
    
    parser.add_argument('--output-dir',
                       default='outputs',
                       help='Base output directory (default: outputs)')
    
    parser.add_argument('--cutoff-date',
                       default='2024-07-01',
                       help='Cutoff date for forecasting validation (default: 2024-07-01)')
    
    parser.add_argument('--forecast-years',
                       type=int,
                       default=3,
                       help='Number of years to forecast (default: 3)')
    
    parser.add_argument('--no-plots',
                       action='store_true',
                       help='Suppress plot display (useful for batch execution)')
    
    args = parser.parse_args()
    
    # Define available analyses
    available_analyses = {
        'data_exploration': {
            'script': 'analysis/analyze_data_exploration.py',
            'args': []
        },
        'model_fit': {
            'script': 'analysis/analyze_model_fit.py', 
            'args': []
        },
        'algorithmic_progress': {
            'script': 'analysis/analyze_algorithmic_progress.py',
            'args': []
        },
        'forecasting': {
            'script': 'analysis/analyze_forecasting.py',
            'args': ['--cutoff-date', args.cutoff_date, 
                    '--forecast-years', str(args.forecast_years)]
        },
        'robustness': {
            'script': 'analysis/analyze_robustness.py',
            'args': []
        }
    }
    
    # Determine which analyses to run
    if args.analyses.lower() == 'all':
        analyses_to_run = list(available_analyses.keys())
    else:
        analyses_to_run = [a.strip() for a in args.analyses.split(',')]
        
        # Validate analysis names
        invalid = [a for a in analyses_to_run if a not in available_analyses]
        if invalid:
            print(f"Error: Unknown analyses: {', '.join(invalid)}")
            print(f"Available analyses: {', '.join(available_analyses.keys())}")
            sys.exit(1)
    
    # Setup environment
    if args.no_plots:
        os.environ['MPLBACKEND'] = 'Agg'  # Use non-interactive backend
    
    # Create output directory
    base_output_dir = setup_output_directory(args.output_dir)
    
    print("BENCHMARK STITCHING ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Analyses to run: {', '.join(analyses_to_run)}")
    print(f"Output directory: {base_output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify all analysis scripts exist
    missing_scripts = []
    for analysis in analyses_to_run:
        script_path = Path(available_analyses[analysis]['script'])
        if not script_path.exists():
            missing_scripts.append(script_path)
    
    if missing_scripts:
        print(f"\nError: Missing analysis scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        print("\nPlease ensure all analysis scripts are present in the current directory.")
        sys.exit(1)
    
    # Run analyses
    results = []
    start_time = time.time()
    
    for analysis in analyses_to_run:
        config = available_analyses[analysis]
        result = run_analysis(config['script'], config['args'])
        results.append(result)
        
        # Stop on first failure if desired (could add this as an option)
        # if not result['success']:
        #     print(f"\nStopping execution due to failure in {analysis}")
        #     break
    
    total_time = time.time() - start_time
    
    # Create execution summary
    create_execution_summary(results, base_output_dir)
    
    # Print final summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print("EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Successful analyses: {successful}/{len(results)}")
    print(f"Failed analyses: {failed}/{len(results)}")
    print(f"Results saved to: {base_output_dir}")
    print(f"Execution summary: {base_output_dir}/execution_summary.txt")
    
    if failed > 0:
        print(f"\nFailed analyses:")
        for result in results:
            if not result['success']:
                print(f"  - {result['script']}")
    
    # Exit with error code if any analysis failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()