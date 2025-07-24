# benchmark-stitching
This project contains code for the paper "A Rosetta Stone for AI Benchmarks". 

## Core Infrastructure
- `data_loader.py`: Load and process benchmark data from internal and external sources
- `fit.py`: Statistical model fitting using sigmoid function to relate capabilities and difficulties
- `analysis_utils.py`: Shared utilities for common analysis patterns and visualizations

## Analysis Scripts
Run individual analyses using these Python scripts (recommended over notebooks):

- `analysis/analyze_data_exploration.py`: Dataset overview, temporal patterns, and benchmark coverage analysis
- `analysis/analyze_model_fit.py`: Core statistical model fitting, capability rankings, and growth rate estimation
- `analysis/analyze_algorithmic_progress.py`: Compute efficiency improvements and algorithmic progress metrics
- `analysis/analyze_forecasting.py`: Future capability forecasting with uncertainty bounds and validation
- `analysis/analyze_robustness.py`: Robustness testing via benchmark dropping, cross-validation, and anchor sensitivity
- `analysis/sanity_check_fit.py`: Visual validation of model capability estimates with random sampling
- `run_all_analyses.py`: Main runner script to execute all analyses with unified output management

### Quick Start
```bash
# Install dependencies (if needed)
uv add numpy pandas matplotlib scipy seaborn jupyter statsmodels scikit-learn

# Run all analyses
uv run run_all_analyses.py

# Run specific analyses  
uv run run_all_analyses.py --analyses model_fit,forecasting

# Run individual analysis
uv run analysis/analyze_model_fit.py

# Sanity check model capabilities
uv run analysis/sanity_check_fit.py
```

### Batch Execution
Use `run_all_analyses.py` to execute multiple analyses with unified output management:

```bash
# Run all analyses with custom settings
uv run run_all_analyses.py --cutoff-date 2024-07-01 --forecast-years 5

# Run without displaying plots (for batch/server execution)
uv run run_all_analyses.py --no-plots

# Get help
uv run run_all_analyses.py --help
```

## Jupyter Notebooks (Legacy)
Original analysis notebooks are still available in `notebooks/`:
- `model_fit.ipynb`: basic model fit based on `fit.py`
- `data_exploration.ipynb`: analyze the data that is processed in `data_loader.py`
- `algorithmic_progress.ipynb`: algorithmic progress analysis
- `predicting_capabilities.ipynb`: capability forecasting analysis
- `cross_validation.ipynb`: compare sigmoid and clipped linear models
- `optimization_pressure.ipynb`: compare fits when you subset to benchmarks that are or aren't optimized for
- `benchmark_inclusion.ipynb`: analyze what happens if you drop some fraction of benchmarks at random
- `change_anchor.ipynb`: see what happens if you change the anchor from winogrande with difficulty 0 and slope param 1

## Output Structure
All analyses save results to `outputs/[analysis_name]/` with:
- Generated plots (PNG files)
- Data tables (CSV files) 
- Analysis summaries (TXT files)
- Detailed parameters and statistics

See `CLAUDE.md` for comprehensive documentation of the methodology, repository structure, and usage examples.