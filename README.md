# benchmark-stitching
This project contains code for the paper "A Rosetta Stone for AI Benchmarks". 

core scripts
- `data_loader.py`
- `fit.py`

to do analysis, work out of the ipython notebooks:
- `model_fit.ipynb`: basic model fit based on `fit.py`
- `data_exploration.ipynb`: analyze the data that is processed in `data_loader.py`
- `algorithmic_progress.ipynb`: algorithmic progress analysis
- `predicting_capabilities.ipynb`: what it sounds like
- `cross_validation.ipynb`: compare sigmoid and clipped linear models
- `optimization_pressure.ipynb`: compare fits when you subset to benchmarks that are or aren't optimized for
- `benchmark_inclusion`: analyze what happens if you drop some fraction of benchmarks at random
- `change_anchor`: see what happens if you change the anchor from winogrande with difficulty 0 and slope param 1.