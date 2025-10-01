# Outputs Directory

This directory contains all generated outputs: figures, tables, and reports.

## Structure

```
outputs/
├── figures/               # All plots and visualizations (PNG, PDF)
├── tables/                # Data tables (CSV, Markdown, LaTeX)
└── reports/               # Generated analysis reports (HTML, PDF)
```

## Figure Categories

### Pricing Engine Validation
- `vg_price_surface.png` - 3D surface plot of VG option prices
- `cgmy_implied_vol_surface.png` - Implied volatility surface
- `greeks_heatmap.png` - Delta, Gamma, Vega visualization
- `parameter_space_coverage.png` - Sobol sampling uniformity
- `sample_option_surfaces.png` - Example training data

### Model Performance
- `training_curves.png` - Loss vs epoch (train/val)
- `prediction_accuracy.png` - Actual vs predicted scatter plots
- `residual_analysis.png` - Q-Q plots, residuals vs fitted
- `model_comparison.png` - MLP vs CNN vs ResNet vs Ensemble
- `cross_validation.png` - K-fold scores box plots
- `forward_walking.png` - MAE over time windows

### Benchmarking
- `ml_vs_traditional_benchmark.png` - Speed comparison (log scale)
- `speed_accuracy_tradeoff.png` - Pareto frontier
- `inference_speed_benchmark.png` - Latency distribution
- `model_size_comparison.png` - Parameters vs accuracy

### Bayesian Analysis
- `mcmc_trace_plots.png` - Chain diagnostics
- `posterior_distributions.png` - Parameter posteriors with HDI
- `parameter_correlations.png` - Corner plot
- `mcmc_diagnostics.png` - R-hat, ESS statistics
- `predictive_uncertainty.png` - Fan chart for option prices
- `calibration_confidence.png` - 2D confidence regions

### Sensitivity Analysis
- `sobol_sensitivity.png` - First/total-order indices
- `sensitivity_heatmap.png` - Jacobian visualization
- `parameter_perturbation.png` - Output variation
- `feature_importance.png` - SHAP values

### System Architecture
- `system_architecture.png` - High-level system diagram
- `data_pipeline_flow.png` - Data flow diagram
- `ml_workflow.png` - Training workflow
- `api_architecture.png` - API system design
- `bayesian_workflow.png` - MCMC process flow

### Interactive
- `interactive_dashboard.html` - Plotly dashboard (view in browser)

## Table Formats

Tables are saved in multiple formats for different use cases:

**CSV** (`.csv`): Machine-readable, Excel-compatible
**Markdown** (`.md`): GitHub-rendered tables
**LaTeX** (`.tex`): Academic paper inclusion

### Key Tables
- `performance_summary.{csv,md,tex}` - All model metrics
- `hyperparameter_grid.{csv,md}` - Tuning results
- `bayesian_summary.csv` - Posterior statistics
- `benchmark_results.{csv,md}` - Speed/accuracy comparison

## Reports

### HTML Reports (interactive)
- `validation_report.html` - Full validation suite results
- `model_comparison_report.html` - Multi-model analysis
- `bayesian_analysis_report.html` - MCMC diagnostics

### PDF Reports (publication-ready)
- `executive_summary.pdf` - One-page project summary
- `technical_report.pdf` - Complete methodology & results

## Naming Conventions

**Figures**: `{category}_{description}_{timestamp}.{ext}`
Example: `validation_residuals_20251001.png`

**Tables**: `{name}_{version}.{format}`
Example: `performance_summary_v2.csv`

**Reports**: `{type}_report_{date}.{format}`
Example: `validation_report_20251001.html`

## Figure Guidelines

All figures should:
1. **Resolution**: 300 DPI for publications, 150 DPI for web
2. **Format**: PNG for raster, PDF for vector graphics
3. **Size**: Max 10MB per file
4. **Labels**: Clear axis labels, legends, titles
5. **Colors**: Colorblind-friendly palettes (seaborn default)
6. **Fonts**: Readable at thumbnail size (min 10pt)

### Example Code
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), dpi=300)
# ... plotting code ...
plt.tight_layout()
plt.savefig('outputs/figures/my_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/my_plot.pdf')  # Vector version
plt.close()
```

## Table Guidelines

Use Pandas for consistent formatting:
```python
import pandas as pd

df = pd.DataFrame(results)

# Save in all formats
df.to_csv('outputs/tables/results.csv', index=False)
df.to_markdown('outputs/tables/results.md', index=False)
df.to_latex('outputs/tables/results.tex', index=False)
```

## Report Generation

Reports are generated via scripts:
```bash
python analysis/generate_validation_report.py --output outputs/reports/validation_report.html
python analysis/generate_executive_summary.py --output outputs/reports/executive_summary.pdf
```

## Version Control

- **Git Tracking**: `.gitignore` excludes large binary outputs
- **DVC** (optional): Track large files with Data Version Control
- **Archival**: Important outputs archived to `outputs/archive/YYYY-MM-DD/`

## Reproducing Outputs

All outputs can be regenerated:
```bash
# Regenerate all figures
python analysis/regenerate_all_plots.py

# Regenerate specific category
python analysis/regenerate_all_plots.py --category bayesian

# Regenerate from specific run
python analysis/regenerate_all_plots.py --run-id vg_calibration_20251001_143022
```

---

**Last Updated**: 2025-10-01
