# AI Benchmark Stitching Project

## Overview
This repository contains the code and analysis for the paper "A Rosetta Stone for AI Benchmarks" by Epoch AI and Google DeepMind. The project creates a unified framework for comparing AI model capabilities across different benchmarks by "stitching" them together onto a common statistical scale.

## Core Methodology
The project uses a statistical model that relates model capabilities (C_m) and benchmark difficulties (D_b) to performance scores via a sigmoid function:

```
performance(m,b) = σ(α_b(C_m - D_b))
```

Where:
- `C_m` = capability of model m on a latent scale
- `D_b` = difficulty of benchmark b on the same scale  
- `α_b` = slope parameter controlling difficulty distribution for benchmark b
- `σ` = sigmoid function

The model addresses two identifiability issues:
1. **Multiplicative rescale**: Fixed by setting anchor slope α_Winogrande = 1
2. **Additive shift**: Fixed by setting anchor difficulty D_Winogrande = 0

## Repository Structure

### Core Scripts
- **`data_loader.py`**: Loads and processes benchmark data from internal (Epoch AI) and external sources. Handles ~25 different benchmarks including MMLU, BBH, FrontierMath, GPQA Diamond, etc. Note: MCBench is excluded as arena winrates don't fit the benchmark stitching paradigm.
- **`fit.py`**: Implements the statistical model fitting using scipy's least squares optimization with Trust Region Reflective algorithm.

### Analysis Scripts (`analysis/`)
- **`analyze_data_exploration.py`**: Dataset structure analysis and coverage statistics
- **`analyze_model_fit.py`**: Core model fitting and capability ranking
- **`analyze_algorithmic_progress.py`**: Compute efficiency vs algorithmic improvements
- **`analyze_forecasting.py`**: Frontier model capability forecasting
- **`analyze_robustness.py`**: Robustness testing and cross-validation
- **`sanity_check_fit.py`**: Visual validation of model capability estimates

### Data Directory (`data/`)
- **`benchmarks_runs.csv`**: Internal Epoch AI benchmark evaluation results
- **`external_benchmark_*.csv`**: External benchmark data from various sources
- **`model_versions.csv`**: Model metadata including release dates and identifiers

### Original Notebooks (`notebooks/`) - Converted to Python Scripts
The original Jupyter notebooks have been converted to standalone Python scripts in the `analysis/` directory:
1. `model_fit.ipynb` → `analyze_model_fit.py`
2. `data_exploration.ipynb` → `analyze_data_exploration.py`
3. `algorithmic_progress.ipynb` → `analyze_algorithmic_progress.py`
4. `predicting_capabilities.ipynb` → `analyze_forecasting.py`
5. `cross_validation.ipynb` & `benchmark_inclusion.ipynb` → `analyze_robustness.py`

## Quick Start

### Installation
Dependencies are managed with `uv`:
```bash
uv add numpy pandas matplotlib scipy seaborn statsmodels scikit-learn
```

### Running All Analyses
```bash
# Run complete analysis pipeline
uv run run_all_analyses.py

# Run specific analysis
uv run analysis/analyze_model_fit.py

# Run with custom parameters
uv run analysis/analyze_forecasting.py --top-n-models 1 --forecast-years 3
```

### Basic Python Usage
```python
from data_loader import scores_df
from fit import fit_statistical_model

# Fit the model using Winogrande as anchor
df, model_capabilities_df, benchmark_params_df = fit_statistical_model(
    scores_df, 
    anchor_benchmark="Winogrande", 
    anchor_difficulty=0, 
    anchor_slope=1
)
```

## Key Findings

### Model Capabilities
- Recent reasoning models (o3, Gemini 2.5 Pro) rank highest
- Clear temporal progression in capability estimates
- Frontier capability growth: ~0.24-0.34 units per year

### Benchmark Difficulties  
- Establishes unified difficulty scale across diverse evaluation tasks
- FrontierMath and advanced reasoning benchmarks are most challenging
- Recent benchmarks tend to be "slightly beyond current frontier"

### Algorithmic Progress
- Compute efficiency improving 1-10× per year for same capability levels
- Capability improvements of 0.1-0.5 units per year at fixed compute budgets
- Clear separation between scaling effects and algorithmic advances

## Analysis Script Documentation

### 1. Data Exploration (`analyze_data_exploration.py`)
**Purpose**: Understand dataset structure, coverage, and quality

**Key Analyses**:
- Temporal distribution of benchmark entries
- Benchmark overlap matrix (how many models shared between benchmarks)
- Coverage statistics (most evaluated models/benchmarks)
- Dataset overview with comprehensive statistics

**Outputs**: `outputs/data_exploration/`
- `temporal_distribution.png`: Monthly data collection patterns
- `benchmark_overlap_matrix.png`: Model overlap heatmap
- `dataset_overview.txt`: Comprehensive statistics summary

---

### 2. Model Fitting (`analyze_model_fit.py`) 
**Purpose**: Core statistical model fitting and capability ranking

**Key Analyses**:
- Sigmoid model fitting with Winogrande anchor
- Bootstrap growth rate estimation (10,000 samples)
- Model capability and benchmark difficulty rankings
- Temporal trend analysis

**Outputs**: `outputs/model_fit/`
- `model_capabilities.csv`: All models with estimated capabilities
- `benchmark_difficulties.csv`: All benchmarks with difficulty estimates
- `capabilities_over_time.png`: Capability progression visualization
- `analysis_summary.txt`: Growth rates and confidence intervals

---

### 3. Algorithmic Progress (`analyze_algorithmic_progress.py`)
**Purpose**: Separate compute scaling from algorithmic improvements

**Key Analyses**:
- Compute data integration with capability estimates
- Compute efficiency analysis (same capability, less compute over time)
- Fixed-compute analysis (better capability at same compute over time)
- Scaling relationship analysis

**Outputs**: `outputs/algorithmic_progress/`
- `compute_efficiency_improvements.png`: Efficiency gains over time
- `capability_improvements_fixed_compute.png`: Capability gains at fixed budgets
- `algorithmic_progress_summary.txt`: Quantified improvement rates

---

### 4. Forecasting (`analyze_forecasting.py`)
**Purpose**: Predict future frontier AI capabilities

**Key Analyses**:
- **Frontier model identification**: Selects models that were top-N at their release date
- Historical forecast validation using 2024-07-01 cutoff
- Linear extrapolation with confidence/prediction intervals
- Benchmark saturation timeline predictions

**Command Options**:
- `--top-n-models N` (default: 1 for pure frontier tracking)
- `--forecast-years N` (default: 3)
- `--cutoff-date YYYY-MM-DD` (default: 2024-07-01)

**Outputs**: `outputs/forecasting/`
- `capability_forecast_3yr.png`: Frontier trend with forecast intervals
- `forecast_validation_2024-07-01.png`: Historical prediction accuracy
- `benchmark_saturation_forecasts.csv`: Timeline to 50% performance

---

### 5. Robustness Testing (`analyze_robustness.py`)
**Purpose**: Validate methodology robustness across different choices

**Key Analyses**:
- Benchmark inclusion robustness (100 iterations dropping 30% of benchmarks)
- Cross-validation: sigmoid vs clipped linear model comparison
- Anchor sensitivity across different reference benchmarks
- Statistical stability via bootstrap methods

**Outputs**: `outputs/robustness/`
- `benchmark_inclusion_robustness.png`: Growth rate stability analysis
- `model_comparison.png`: Sigmoid vs linear model performance
- `robustness_summary.txt`: Stability metrics and confidence ranges

---

### 6. Sanity Check (`sanity_check_fit.py`)
**Purpose**: Visual validation of model capability estimates

**Key Features**:
- Random sampling of models with reproducible seeds
- Color-coded capability visualization
- Statistical summary and reasonableness assessment prompts

**Outputs**: `outputs/sanity_check/`
- `capability_sanity_check.png`: Capability ranking visualization
- `sampled_models.csv`: Sample data for detailed inspection

## Important Data Processing

### Filtering Rules
- Models evaluated on ≤3 benchmarks are excluded (configurable)
- Performance scores outside [0,1] range are clipped
- Duplicate (model, benchmark) pairs aggregated using minimum score
- Specific outliers removed (e.g., o3-mini/o4-mini on SWE-Bench due to function calling issues)

### Benchmark Selection
- **Included**: ~25 benchmarks spanning reasoning, knowledge, coding, and specialized tasks
- **Excluded**: MCBench (arena winrates don't fit stitching paradigm), OSUniverse (agent scaffold sensitive)
- **Anchor**: Winogrande used as reference point (difficulty=0, slope=1)

## Applications & Use Cases

- **Model Comparison**: Compare capabilities across models not evaluated on same benchmarks
- **Benchmark Analysis**: Understand relative difficulties of evaluation tasks
- **Progress Tracking**: Monitor AI capability improvements over time with uncertainty quantification
- **Forecasting**: Predict frontier model capabilities and benchmark saturation timelines
- **Research Planning**: Identify capability gaps and benchmark development priorities

## Technical Notes

- **Optimization**: Uses Trust Region Reflective algorithm (scipy.optimize.least_squares)
- **Uncertainty**: Bootstrap methods throughout for confidence intervals
- **Validation**: Extensive robustness testing and cross-validation
- **Performance**: Model fitting typically converges in seconds
- **Scalability**: Framework handles hundreds of models and dozens of benchmarks efficiently

## Paper Context
This work provides a statistical alternative to human preference-based rankings (like ELO scores) by aggregating existing benchmark results without requiring additional evaluations. It enables analysis of AI capability trends that would be difficult to spot with individual benchmarks alone, offering a "Rosetta Stone" for translating performance across different evaluation frameworks.