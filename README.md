# benchmark-stitching
This project contains code for the paper "A Rosetta Stone for AI Benchmarks". 

## Core Infrastructure
- `data_loader.py`: Load and process benchmark data from internal and external sources
- `fit.py`: Statistical model fitting using sigmoid function to relate capabilities and difficulties
- `analysis_utils.py`: Shared utilities for common analysis patterns and visualizations

## Analysis files
All analysis files can be found in `analysis/`:
- `algorithmic_progress_new.ipynb`: Algorithmic progress analysis (Section 3.2)
- `benchmark_inclusion.ipynb`: Robustness check varying benchmark inclusion (Appendix E.3)
- `change_anchor.ipynb`: Robustness check varying the benchmark anchor (Appendix E.2)
- `data_exploration.ipynb`: Creating plots to visualize benchmarking data
- `forecasting.ipynb`: Simple projections of the estimated capability trend (Section 3.2.1)
- `model_fit.ipynb`: Analysis notebook testing model fit and creating plots of model capabilities/benchmark difficulties over time
- `optimization_pressure.py`: Testing whether or not we see higher estimated capabilities among benchmarks that have been heavily "optimized for" by AI labs (Section 3.1.2 and Appendix E.3.2)
- `sigmoid_vs_linear_fit.ipynb`: Cross-validation varying the assumption that the map from `capabilities - difficulties` to performance is sigmoidal (Section E.4)
- `software_singularity.py`: Synthetic data generation and analysis to see if our framework can detect rapid accelerations in model capabilities (Section 3.3)

## Output Structure
All analyses save results to `outputs/[analysis_name]/` with:
- Generated plots (PNG files)
- Data tables (CSV files) 
- Analysis summaries (TXT files)
- Detailed parameters and statistics

See `CLAUDE.md` for comprehensive documentation of the methodology, repository structure, and usage examples.

## Creating paper figures and tables
- Figure 0: Illustrating how benchmark stitching works in the first place - `model_fit.ipynb`
- Figure 1: Estimated model capabilities and benchmark difficulties over time, with error bars - `model_fit.ipynb`
- Figure 2: Top 10 models and benchmarks - `model_fit.ipynb`
- Figure 3: Map to METR time horizon - `model_fit.ipynb`
- Figure 4: Residuals on SWE-Bench verified and GeoBench - `model_fit.ipynb`
- Table 1: Example capabilities scores - `model_fit.ipynb`
- Figure 5: 3 year forecast - `forecasting.ipynb`
- Figure 6: Forecast validation - `forecasting.ipynb`
- Figure 7: Capabilities vs log training compute - `algorithmic_progress_new.ipynb`
- Table 2: Algorithmic progress estimates - `algorithmic_progress_new.ipynb`
- Figure 8: Synthetic data acceleration detection - `software_singularity.py` (use the `--plot-only` flag)
- Figure 9: Detecting acceleration in actual data - `forecasting.ipynb`
- Table 3: Internal benchmarks - N/A
- Table 4: External benchmarks - N/A
- Figure 10: Benchmark overlap - `data_exploration.ipynb`
- Table 5: Algorithmic progress estimates without dropping distilled models - `algorithmic_progress_new.ipynb`
- Figure 11: Scale-dependence of algorithmic progress - N/A
- Table 6: Algorithmic progress (compute reduction) via direct observation - `algorithmic_progress_new.ipynb`
- Table 7: Algorithmic progress (capability growth) via direct observation - `algorithmic_progress_new.ipynb`
- Figure 12: Algorithmic progress via direct observation - `algorithmic_progress_new.ipynb`
- Figure 13: Synthetic data example - `software_singularity.py`
- Figure 14: Residuals for synthetic data - `model_fit.ipynb`
- Figure 15: Varying the amount of benchmark overlap - `model_fit.ipynb` (change the overlap in `data_loader.py`)
- Figure 16: Robustness check changing benchmark anchors - `change_anchor.ipynb`
- Figure 17: Checking if we can measure an effect from benchmarks being "optimized-for" - `optimization_pressure.ipynb`
- Table 8: Classification of benchmarks into optimized-for/not-optimized-for - N/A
- Figure 18: Temporal distribution of our benchmarking data - `data_exploration.ipynb`
- Table 9: Cross validation comparing sigmoid and clipped linear models - `cross_validation.ipynb`
- Figure 19: Analysis with older data (change this in `data_loader.py`)
- Table 10: Estimated GPT jumps based on older data - `model_fit.ipynb`
- Figure 20: residuals on each benchmark - `model_fit.ipynb`