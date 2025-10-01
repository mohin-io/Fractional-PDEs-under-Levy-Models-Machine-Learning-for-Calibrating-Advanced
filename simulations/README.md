# Simulations Directory

This directory contains all simulation runs, organized by Lévy model type and analysis category.

## Structure

```
simulations/
├── variance_gamma/         # Variance Gamma model simulations
│   ├── runs/              # Individual simulation runs (.json config + results)
│   ├── plots/             # Model-specific plots
│   └── results.json       # Aggregated results summary
│
├── cgmy/                  # CGMY model simulations
│   ├── runs/              # Individual simulation runs
│   ├── plots/             # Model-specific plots
│   └── results.json       # Aggregated results summary
│
└── comparison/            # Cross-model comparisons
    ├── plots/             # Comparison visualizations
    └── benchmarks.json    # Performance benchmarks
```

## Running Simulations

### Variance Gamma Calibration
```bash
python -m models.generate_dataset --model VarianceGamma --samples 100000
python -m features.build_features
python -m models.calibration_net.train
python -m analysis.out_of_sample --model-path models/calibration_net/mlp_calibration_model.h5
```

### CGMY Calibration
```bash
python -m models.generate_dataset --model CGMY --samples 100000
python -m features.build_features
python -m models.calibration_net.train --output-dim 4
python -m analysis.out_of_sample
```

### Bayesian Calibration
```bash
python -m models.bayesian_calibration.mcmc --samples 5000 --chains 4
```

## Simulation Naming Convention

Individual runs are stored with timestamps:
```
simulations/variance_gamma/runs/vg_calibration_20251001_143022.json
simulations/cgmy/runs/cgmy_bayesian_20251001_150145.json
```

Each run file contains:
- **config**: Hyperparameters, model settings
- **metrics**: Train/test loss, MAE, R²
- **timing**: Execution time, timestamps
- **artifacts**: Paths to saved models, plots

## Plot Organization

Plots are automatically generated and saved to respective `plots/` directories:

**Variance Gamma**:
- `vg_training_curves.png`
- `vg_prediction_scatter.png`
- `vg_residual_analysis.png`
- `vg_parameter_distribution.png`

**CGMY**:
- `cgmy_training_curves.png`
- `cgmy_prediction_scatter.png`
- `cgmy_residual_analysis.png`
- `cgmy_parameter_distribution.png`

**Comparison**:
- `vg_vs_cgmy_accuracy.png`
- `model_speed_benchmark.png`
- `ensemble_performance.png`

## Results Format

`results.json` aggregates all runs for a model:
```json
{
  "model_type": "VarianceGamma",
  "total_runs": 5,
  "best_run": {
    "timestamp": "2025-10-01T14:30:22",
    "test_mse": 0.0008,
    "test_mae": 0.015,
    "test_r2": 0.967,
    "config": {...}
  },
  "average_metrics": {
    "test_mse": 0.0012,
    "test_mae": 0.018,
    "test_r2": 0.954
  },
  "runs": [...]
}
```

## Reproducibility

All simulations include:
1. Random seeds in config
2. Dataset checksums
3. Library versions (requirements.txt snapshot)
4. Git commit hash

To reproduce a run:
```bash
python reproduce_simulation.py --run-id vg_calibration_20251001_143022
```

---

**Last Updated**: 2025-10-01
