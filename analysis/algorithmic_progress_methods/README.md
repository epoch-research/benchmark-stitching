# Algorithmic Progress Analysis Methods

This directory contains different statistical methods for analyzing algorithmic progress in AI by separating compute scaling effects from algorithmic improvements.

## 📁 Directory Structure

```
algorithmic_progress_methods/
├── shared/                          # Shared utilities (~1500 lines)
│   ├── __init__.py
│   ├── data_loading.py             # Data loading & preprocessing (~270 lines)
│   ├── bootstrap.py                # Bootstrap analysis utilities (~170 lines)
│   ├── cli_utils.py                # CLI argument parsing (~165 lines)
│   └── plotting/                    # Plotting utilities (~950 lines)
│       ├── __init__.py
│       ├── base.py                 # Basic plotting utilities (~90 lines)
│       ├── distributions.py        # Histogram & bootstrap plots (~140 lines)
│       ├── regressions.py          # Scatter + fit plots (~185 lines)
│       ├── diagnostics.py          # Uncertainty diagnostics (~205 lines)
│       └── unified_plots.py        # Unified plotting functions (~285 lines)
│
├── buckets/                         # Buckets method
│   ├── analysis.py                 # Core analysis logic
│   ├── plotting.py                 # Visualization functions
│   ├── main.py                     # CLI entry point
│   ├── hierarchical_median.py      # Hierarchical median estimator
│   ├── hierarchical_median_visualization.py  # Visualization for hierarchical model
│   └── BUCKET_NOTES.md             # Methodology documentation
│
├── linear_model/                    # Linear regression method
│   ├── analysis.py                 # Core analysis logic
│   ├── plotting.py                 # Visualization functions
│   ├── main.py                     # CLI entry point
│   ├── LINEAR_NOTES.md             # Methodology documentation
│   └── plot_predicted_pareto_frontier.py  # Pareto frontier visualization
│
├── families/                        # Model families method (in development)
│   └── FAMILIES_NOTES.md
│
├── run_everything.py               # Run all methods with all configurations
├── result_collector.py             # Collect and summarize results
├── show_results_summary.py         # Display results summary
└── README.md                        # This file
```

## 📊 Methods Overview

### Buckets Method
Groups models into bins based on their estimated capability index (ECI) and computes how much less training compute is required over time to achieve the same capability level. This approach:
- Divides the capability and compute space into discrete buckets
- Analyzes temporal trends within each bucket
- Estimates compute reduction rates via linear regression
- Provides robustness through sensitivity analysis over different bucket sizes

See [BUCKET_NOTES.md](buckets/BUCKET_NOTES.md) for detailed methodology.

### Linear Model Method
Fits a multivariate linear regression to model capability as a function of training compute and release date. This approach:
- Directly estimates the contribution of compute scaling vs. algorithmic progress
- Uses bootstrap resampling for uncertainty quantification
- Visualizes results with effective capability index (ECI) contours
- Enables forecasting of future frontier model capabilities

See [LINEAR_NOTES.md](linear_model/LINEAR_NOTES.md) for detailed methodology.

## 🔧 Usage

### Quick Start - Run Everything

The easiest way to run all analyses with all configurations:

```bash
# Run all methods with all configurations (internal + website data, with/without distilled)
cd analysis/algorithmic_progress_methods
python run_everything.py

# Quick preview of what will run (dry run)
python run_everything.py --dry-run

# Run with reduced iterations (faster, for testing)
python run_everything.py --quick

# Run only specific methods
python run_everything.py --buckets-only
python run_everything.py --linear-only
python run_everything.py --hierarchical-only

# Skip website data configurations (faster)
python run_everything.py --skip-website

# Continue even if some analyses fail
python run_everything.py --continue-on-error

# See all options
python run_everything.py --help
```

The `run_everything.py` script automatically runs:
- **Buckets method** with 6 configurations (internal/website × all/exclude-med-high-distilled/exclude-all-distilled)
- **Buckets sensitivity analysis** for each configuration (can skip with `--skip-sensitivity`)
- **Hierarchical median estimator** for each configuration (can skip with `--skip-hierarchical`)
- **Linear model** with 8 configurations (adds frontier-only variants)

Results are organized in `outputs/algorithmic_progress_methods/` by method and configuration.

### Individual Method Usage

#### Buckets Method

```bash
# Basic usage
cd analysis/algorithmic_progress_methods/buckets
python main.py

# With options
python main.py --eci-bucket-size 0.3 --exclude-distilled

# Sensitivity analysis
python main.py --sweep-bucket-sizes --n-bucket-sizes 5

# Hierarchical median estimator (sweeps bucket sizes and pools results)
python hierarchical_median.py --n-bucket-sizes 7 --n-iter 30000

# Same hierarchical workflow via the buckets CLI (analysis + plots)
python main.py --run-hierarchical-median --hierarchical-n-bucket-sizes 7

# Help
python main.py --help
```

**Common Options:**
- `--eci-bucket-size`: Width of ECI buckets (default: 0.5)
- `--compute-bucket-size`: Width of compute buckets (default: 0.5 OOMs)
- `--min-models`: Minimum SOTA models per bucket (default: 3)
- `--exclude-med-high-distilled`: Exclude medium and high confidence distilled models
- `--exclude-distilled`: Exclude ALL distilled models (all confidence levels)
- `--use-website-data`: Use website data instead of fitted data
- `--min-release-date`: Only include models released on or after specified date (YYYY-MM-DD)
- `--sweep-bucket-sizes`: Run sensitivity analysis
- `--run-hierarchical-median`: Sweep bucket sizes, fit the hierarchical model, and generate diagnostics
- `--label-points`: Label data points with model names

#### Linear Model Method

```bash
cd analysis/algorithmic_progress_methods/linear_model
python main.py

# With options
python main.py --exclude-distilled --frontier-only

# Help
python main.py --help
```

**Options:**
- `--exclude-med-high-distilled`: Exclude medium and high confidence distilled models
- `--exclude-distilled`: Exclude ALL distilled models (all confidence levels)
- `--use-website-data`: Use website data instead of fitted data
- `--frontier-only`: Only include models on the Pareto frontier at release
- `--show-predicted-frontier`: Show predicted Pareto frontier for each month
- `--label-points`: Label data points with ECI values
- `--min-release-date`: Only include models released on or after specified date (YYYY-MM-DD)
- `--contour-spacing`: Spacing between ECI contour lines
- `--color-contours`: Color ECI contour lines by their value using viridis colormap
- `--eci-min` / `--eci-max`: Control displayed ECI range on plot

## 📊 Output Structure

All methods save outputs to `outputs/algorithmic_progress_methods/{method_name}/`:

```
outputs/algorithmic_progress_methods/
├── buckets/
│   ├── compute_reduction_results{suffix}.csv
│   ├── capability_gains_results{suffix}.csv
│   ├── compute_reduction_analysis{suffix}.png
│   ├── capability_gains_analysis{suffix}.png
│   ├── all_bucket_regressions_compute_reduction{suffix}.png
│   ├── bootstrap_distributions_compute_reduction{suffix}.png
│   ├── hierarchical_median_observations{suffix}.csv
│   ├── hierarchical_median_summary{suffix}.json
│   └── hierarchical_median_diagnostics{suffix}.png / .svg
│
└── linear_model/
    ├── compute_vs_date_with_eci{suffix}.png / .svg
    ├── bootstrap_uncertainty_diagnostics{suffix}.png
    ├── coefficient_correlation{suffix}.png
    └── compute_year_tradeoff_distribution{suffix}.png
```

**Suffix patterns:**
- `_no_distilled` - Excluded medium/high confidence distilled models (--exclude-med-high-distilled)
- `_no_distilled_all` - Excluded ALL distilled models including low confidence (--exclude-distilled)
- `_website` - Used website data instead of fitted data
- `_frontier_only` - Only Pareto frontier models

## 🎯 Key Insights

Both methods aim to decompose AI capability improvements into:

1. **Compute scaling**: Improvements from using more training compute
2. **Algorithmic progress**: Improvements from better algorithms, architectures, and training techniques

By analyzing models with known training compute and estimated capabilities, we can quantify:
- How much compute is saved per year for achieving the same capability (compute efficiency gains)
- How much capability improves per year at fixed compute budgets (algorithmic improvements)
- The relative contributions of scaling vs. innovation to frontier progress

---

**Last updated:** 2025
