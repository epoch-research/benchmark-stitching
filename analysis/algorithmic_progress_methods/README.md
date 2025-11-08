# Algorithmic Progress Analysis Methods

This directory contains different statistical methods for analyzing algorithmic progress in AI by separating compute scaling effects from algorithmic improvements.

## 📁 Directory Structure

```
algorithmic_progress_methods/
├── shared/                          # Shared utilities (~900 lines)
│   ├── __init__.py
│   ├── data_loading.py             # Data loading & preprocessing (~200 lines)
│   ├── bootstrap.py                # Bootstrap analysis utilities (~170 lines)
│   ├── cli_utils.py                # CLI argument parsing (~100 lines)
│   └── plotting/                    # Plotting utilities (~600 lines)
│       ├── __init__.py
│       ├── base.py                 # Basic plotting utilities (~80 lines)
│       ├── distributions.py        # Histogram & bootstrap plots (~140 lines)
│       ├── regressions.py          # Scatter + fit plots (~180 lines)
│       └── diagnostics.py          # Uncertainty diagnostics (~200 lines)
│
├── buckets/                         # Buckets method
│   ├── analysis.py                 # Core analysis logic
│   ├── plotting.py                 # Visualization functions
│   ├── main.py                     # CLI entry point
│   └── BUCKET_NOTES.md             # Methodology documentation
│
├── linear_model/                    # Linear regression method
│   ├── analysis.py                 # Core analysis logic
│   ├── main.py                     # CLI entry point
│   ├── LINEAR_NOTES.md             # Methodology documentation
│   └── plot_compute_vs_date_with_eci.py
│
├── families/                        # Model families method (in development)
│   └── FAMILIES_NOTES.md
│
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

### Buckets Method

```bash
# Activate environment
source .venv/bin/activate

# Basic usage
cd analysis/algorithmic_progress_methods/buckets
python main.py

# With options
python main.py --eci-bucket-size 0.3 --exclude-distilled

# Sensitivity analysis
python main.py --sweep-bucket-sizes --n-bucket-sizes 5

# Help
python main.py --help
```

**Common Options:**
- `--eci-bucket-size`: Width of ECI buckets (default: 0.5)
- `--compute-bucket-size`: Width of compute buckets (default: 0.5 OOMs)
- `--min-models`: Minimum SOTA models per bucket (default: 3)
- `--exclude-distilled`: Exclude distilled models
- `--use-website-data`: Use website data instead of fitted data
- `--sweep-bucket-sizes`: Run sensitivity analysis

### Linear Model Method

Currently uses the original script:

```bash
cd analysis/algorithmic_progress_methods/linear_model
python plot_compute_vs_date_with_eci.py --help
```

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
│   └── bootstrap_distributions_compute_reduction{suffix}.png
│
└── linear_model/
    ├── compute_vs_date_with_eci{suffix}.png
    ├── bootstrap_uncertainty_diagnostics{suffix}.png
    └── fitted_capabilities_cache{suffix}.pkl
```

**Suffix patterns:**
- `_no_distilled` - Excluded high/medium confidence distilled models
- `_no_distilled_all` - Excluded all distilled models (including low confidence)
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
