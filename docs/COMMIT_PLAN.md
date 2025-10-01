# Commit Sequence Plan

This document outlines the atomic commit sequence for implementing the Lévy Model Calibration Engine, following the plan in [PLAN.md](PLAN.md).

## Commit Strategy

- **Atomic commits**: Each commit represents a single logical change
- **Conventional commits**: Format `<type>(<scope>): <description>`
- **Small, focused**: Prefer many small commits over large ones
- **Buildable**: Each commit should leave the codebase in a working state
- **Sequenced**: Commits ordered by dependencies

---

## Phase 0: Repository Setup & Documentation (6 commits)

### Commit 1: Project planning documentation
```bash
git add docs/PLAN.md docs/ARCHITECTURE.md
git commit -m "docs: add comprehensive project plan and architecture documentation

- Add PLAN.md with 6-phase implementation strategy (18-24 days)
- Add ARCHITECTURE.md with system design and component details
- Document data pipeline, ML workflow, and API architecture
- Include performance benchmarks and scalability considerations
"
```

### Commit 2: Simulation and output directory structure
```bash
git add simulations/ outputs/ .gitignore
git commit -m "feat(structure): create simulation and output directory organization

- Create simulations/{variance_gamma,cgmy,comparison}/{runs,plots}
- Create outputs/{figures,tables,reports}
- Add README.md for each directory with usage guidelines
- Update .gitignore to exclude large binary outputs
"
```

### Commit 3: Update README to 2025 standards
```bash
git add README.md
git commit -m "docs: update README.md to 2025 recruiter-friendly standards

- Add problem statement and solution overview
- Include quickstart demo and full workflow
- Add performance benchmarks table and architecture diagram
- Document API usage with curl examples
- Add 'For Recruiters' section highlighting skills
- Include roadmap and project stats
"
```

### Commit 4: Commit plan documentation
```bash
git add docs/COMMIT_PLAN.md
git commit -m "docs: add detailed commit sequence plan

- Document all planned commits for 6 implementation phases
- Include commit messages and file changes
- Estimate 37-47 atomic commits total
- Align with PLAN.md timeline
"
```

### Commit 5: Update CLAUDE.md with new structure
```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with simulation/output structure

- Add simulations/ and outputs/ directory documentation
- Update common commands for visualization generation
- Document commit sequence strategy
- Add visual asset organization guidelines
"
```

### Commit 6: Add placeholder .gitkeep files
```bash
find simulations outputs -type d -empty -exec touch {}/.gitkeep \;
git add simulations/ outputs/
git commit -m "chore: add .gitkeep files to track empty directories"
```

---

## Phase 1: Pricing Engine Enhancement (7-9 commits)

### Commit 7: Enhance Carr-Madan pricer with adaptive damping
```bash
git add models/pricing_engine/fourier_pricer.py
git commit -m "feat(pricing): add adaptive damping and improved interpolation

- Implement adaptive alpha parameter selection
- Add CubicSpline interpolation for better accuracy
- Increase default N to 2^12 for finer resolution
- Add docstring examples and parameter validation
"
```

### Commit 8: Add put option pricing via put-call parity
```bash
git add models/pricing_engine/fourier_pricer.py
git commit -m "feat(pricing): implement put option pricing

- Add option_type parameter ('call' or 'put')
- Implement put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
- Add dividend yield parameter q
- Add unit tests for put-call parity verification
"
```

### Commit 9: Implement Greeks computation
```bash
git add models/pricing_engine/fourier_pricer.py tests/test_pricing.py
git commit -m "feat(pricing): add Greeks computation via finite differences

- Implement compute_greeks() for Delta, Gamma, Vega, Theta, Rho
- Use central finite differences with adaptive epsilon
- Add Richardson extrapolation for improved accuracy
- Add unit tests verifying Greeks properties (e.g., Delta in [0,1])
"
```

### Commit 10: Validate pricing against Monte Carlo
```bash
git add tests/test_pricing_validation.py simulations/variance_gamma/runs/
git commit -m "test(pricing): validate Fourier pricer against Monte Carlo

- Implement Monte Carlo pricer with 10M paths
- Compare VG and CGMY prices (relative error < 0.1%)
- Add validation plots to outputs/figures/
- Store validation results in simulations/variance_gamma/runs/
"
```

### Commit 11: Add PIDE solver (optional enhancement)
```bash
git add models/pricing_engine/pide_solver.py
git commit -m "feat(pricing): implement fractional PIDE solver

- Add FractionalPIDESolver class using finite differences
- Implement implicit-explicit time stepping
- Add quadrature for jump integral term
- Compare with Fourier pricer (outputs/figures/pide_vs_fourier.png)
"
```

### Commit 12: Generate CGMY dataset
```bash
git add models/generate_dataset_cgmy.py
git commit -m "feat(data): add CGMY model dataset generation

- Create generate_dataset_cgmy.py with C,G,M,Y parameter ranges
- Use same Sobol sampling strategy as VG
- Generate 100k samples with progress logging
- Output to data/synthetic/cgmy_training_data.parquet
"
```

### Commit 13: Add market noise simulation
```bash
git add models/add_market_noise.py
git commit -m "feat(data): implement market microstructure noise

- Add bid-ask spread simulation (±0.5% to ±5%)
- Implement missing data patterns (sparse option chains)
- Add Gaussian measurement noise
- Create noisy dataset variants for robustness testing
"
```

### Commit 14: Refactor dataset generation for modularity
```bash
git add models/generate_dataset.py models/dataset_utils.py
git commit -m "refactor(data): modularize dataset generation

- Extract common functions to dataset_utils.py
- Add CLI arguments for model_type, num_samples, noise_level
- Support parallel processing with multiprocessing.Pool
- Add progress bars with tqdm
"
```

### Commit 15: Create pricing validation plots
```bash
git add analysis/plot_pricing_validation.py outputs/figures/
git commit -m "docs: generate pricing engine validation visualizations

- Add VG price surface 3D plot
- Add CGMY implied volatility surface heatmap
- Add Greeks heatmap (Delta, Gamma, Vega)
- Add parameter space coverage histograms
- Save publication-quality figures (300 DPI)
"
```

---

## Phase 2: Neural Network Architecture (8-10 commits)

### Commit 16: Enhance baseline MLP with batch normalization
```bash
git add models/calibration_net/model.py
git commit -m "feat(model): add batch normalization to MLP architecture

- Add BatchNormalization layers after Dense layers
- Implement learning rate schedule (ReduceLROnPlateau)
- Add early stopping with patience=10
- Track training metrics in callbacks
"
```

### Commit 17: Implement CNN architecture
```bash
git add models/calibration_net/architectures.py
git commit -m "feat(model): add CNN architecture for option surfaces

- Implement build_cnn_model() treating surface as 2D image
- Add Conv2D layers (32, 64 filters) with ReLU activation
- Use MaxPooling for translation invariance
- Add architecture diagram to docs/
"
```

### Commit 18: Implement ResNet architecture
```bash
git add models/calibration_net/architectures.py
git commit -m "feat(model): add residual network with skip connections

- Implement ResidualBlock class with identity mapping
- Stack 3 ResBlocks with 256 units each
- Add batch normalization in each block
- Prevents vanishing gradients in deep networks
"
```

### Commit 19: Implement ensemble framework
```bash
git add models/calibration_net/ensemble.py
git commit -m "feat(model): create calibration ensemble framework

- Implement CalibrationEnsemble class
- Support simple averaging and weighted aggregation
- Add stacking with meta-learner option
- Load multiple trained models for prediction
"
```

### Commit 20: Add mixed precision training
```bash
git add models/calibration_net/train.py
git commit -m "perf(training): enable mixed precision training for speedup

- Enable tf.keras.mixed_precision for GPU acceleration
- Add gradient scaling to prevent underflow
- Measure training speedup (1.5-2x faster)
- Maintain numerical stability
"
```

### Commit 21: Implement model comparison script
```bash
git add analysis/model_comparison.py
git commit -m "test(model): add comprehensive model comparison framework

- Compare MLP, CNN, ResNet, Ensemble on test set
- Measure accuracy (MSE, MAE, R²) and speed (inference time)
- Generate comparison table (outputs/tables/model_comparison.csv)
- Plot training curves side-by-side
"
```

### Commit 22: Generate model comparison visualizations
```bash
git add outputs/figures/ analysis/plot_model_comparison.py
git commit -m "docs: create model architecture comparison plots

- Training curves comparison (4 models overlaid)
- Prediction accuracy box plots
- Inference speed benchmark bar chart
- Speed-accuracy tradeoff scatter plot
"
```

### Commit 23: Fix overfitting in deep networks
```bash
git add models/calibration_net/model.py
git commit -m "fix(model): add regularization to reduce overfitting

- Increase dropout rates (0.2 → 0.3)
- Add L2 regularization (λ=1e-4)
- Reduce model capacity (256→128→64 → 128→64→32)
- Improve generalization gap from 15% to 5%
"
```

### Commit 24: Add hyperparameter tuning results
```bash
git add simulations/comparison/hyperparameter_tuning.json outputs/tables/
git commit -m "docs: document hyperparameter tuning results

- Grid search over learning rate, dropout, hidden units
- Best config: lr=1e-3, dropout=0.3, units=[128,64,32]
- Save tuning results to simulations/comparison/
- Add hyperparameter table to outputs/tables/
"
```

### Commit 25: Optimize training pipeline
```bash
git add models/calibration_net/train.py
git commit -m "perf(training): optimize data loading and preprocessing

- Use tf.data.Dataset API for efficient batching
- Add prefetching and caching
- Parallelize data augmentation
- Reduce training time by 30%
"
```

---

## Phase 3: Bayesian Calibration (6-8 commits)

### Commit 26: Implement full MCMC calibration
```bash
git add models/bayesian_calibration/mcmc.py
git commit -m "feat(bayesian): implement complete MCMC calibration with PyMC3

- Add BayesianCalibrator class with build_model()
- Define priors: LogNormal(σ), Gamma(ν), Normal(θ)
- Implement likelihood using Fourier pricer
- Run NUTS sampler with 4 chains, 5000 samples each
"
```

### Commit 27: Add MCMC diagnostics and visualization
```bash
git add models/bayesian_calibration/mcmc.py outputs/figures/
git commit -m "feat(bayesian): add posterior analysis and diagnostics

- Implement posterior_analysis() method
- Generate trace plots, posterior distributions, pair plots
- Compute R-hat, ESS, MCSE statistics
- Save diagnostics to outputs/tables/bayesian_summary.csv
"
```

### Commit 28: Implement variational inference
```bash
git add models/bayesian_calibration/variational.py
git commit -m "feat(bayesian): add variational inference for fast calibration

- Implement VariationalCalibrator using ADVI
- Approximate posterior as Gaussian (mean-field)
- Achieve 100x speedup over MCMC (60s vs 10min)
- Trade slight accuracy for speed
"
```

### Commit 29: Add uncertainty propagation
```bash
git add models/bayesian_calibration/uncertainty_propagation.py
git commit -m "feat(bayesian): implement uncertainty propagation to pricing

- Add propagate_uncertainty() function
- Compute predictive distribution for new options
- Calculate mean, std, and 95% HDI for prices
- Generate fan charts showing uncertainty bands
"
```

### Commit 30: Validate posterior convergence
```bash
git add tests/test_bayesian.py simulations/variance_gamma/runs/
git commit -m "test(bayesian): validate MCMC convergence diagnostics

- Check R-hat < 1.01 for all parameters
- Verify ESS > 1000 (effective sample size)
- Test posterior coverage: 95% CIs cover true params 96% of time
- Store convergence plots in simulations/
"
```

### Commit 31: Optimize MCMC with JAX/NumPyro
```bash
git add models/bayesian_calibration/mcmc_jax.py
git commit -m "perf(bayesian): add GPU-accelerated MCMC with NumPyro

- Reimplement model using NumPyro (JAX backend)
- Enable GPU acceleration for 10x speedup
- Maintain numerical stability with float64
- Benchmark: 10min → 1min on V100 GPU
"
```

### Commit 32: Generate Bayesian analysis visualizations
```bash
git add outputs/figures/ analysis/plot_bayesian.py
git commit -m "docs: create comprehensive Bayesian analysis plots

- MCMC trace plots with burn-in period
- Posterior distributions with prior overlays
- Parameter correlation corner plots
- Predictive uncertainty fan charts
- Calibration confidence regions (2D contours)
"
```

### Commit 33: Add Bayesian model comparison
```bash
git add analysis/bayesian_model_comparison.py
git commit -m "feat(bayesian): implement Bayesian model comparison

- Compute WAIC (Widely Applicable Information Criterion)
- Compare VG vs CGMY model fit
- Add Bayes factors for model selection
- Plot posterior predictive checks
"
```

---

## Phase 4: Validation & Analysis (9-11 commits)

### Commit 34: Enhance out-of-sample validation
```bash
git add analysis/out_of_sample.py
git commit -m "feat(validation): add comprehensive residual analysis

- Implement Q-Q plots for normality testing
- Add residuals vs fitted value plots
- Test autocorrelation of errors
- Perform Shapiro-Wilk normality test
"
```

### Commit 35: Implement k-fold cross-validation
```bash
git add analysis/cross_validation.py
git commit -m "feat(validation): add k-fold cross-validation framework

- Implement stratified 5-fold CV
- Ensure balanced parameter ranges in each fold
- Track per-fold metrics (MSE, MAE, R²)
- Plot box plots of cross-validation scores
"
```

### Commit 36: Enhance forward-walking validation
```bash
git add analysis/forward_walking.py
git commit -m "feat(validation): implement expanding window temporal validation

- Add expanding window approach (train on [0:n], test on [n:n+k])
- Track model drift over time windows
- Plot MAE evolution across windows
- Detect concept drift with statistical tests
"
```

### Commit 37: Implement Sobol sensitivity analysis
```bash
git add analysis/sensitivity_analysis.py requirements.txt
git commit -m "feat(validation): add global Sobol sensitivity analysis

- Use SALib for variance-based sensitivity
- Compute first-order and total-order indices
- Identify most influential parameters
- Generate Sobol index bar charts
"
```

### Commit 38: Add local sensitivity (Jacobian)
```bash
git add analysis/sensitivity_analysis.py
git commit -m "feat(validation): add local sensitivity via Jacobian computation

- Compute ∂(predicted_params)/∂(input_prices) using TensorFlow GradientTape
- Generate Jacobian heatmaps
- Identify most informative strikes/maturities
- Visualize sensitivity patterns
"
```

### Commit 39: Implement robustness tests
```bash
git add analysis/robustness_tests.py
git commit -m "test(robustness): add noise injection and OOD detection

- Test model with Gaussian noise (±1%, ±5%, ±10%)
- Measure accuracy degradation
- Implement Mahalanobis distance for OOD detection
- Flag samples far from training distribution
"
```

### Commit 40: Generate validation report
```bash
git add analysis/generate_validation_report.py outputs/reports/
git commit -m "docs: create comprehensive validation HTML report

- Aggregate all validation metrics
- Include plots: residuals, CV scores, sensitivity
- Add statistical test results
- Generate interactive HTML with Plotly
"
```

### Commit 41: Fix autocorrelation in residuals
```bash
git add models/calibration_net/train.py
git commit -m "fix(validation): correct residual autocorrelation

- Add time-based features (days to maturity)
- Implement sequence shuffling in training
- Reduce autocorrelation from 0.3 to 0.05
- Improve residual independence
"
```

### Commit 42: Add statistical significance testing
```bash
git add analysis/significance_testing.py
git commit -m "test(validation): implement statistical significance tests

- Perform t-tests on prediction errors
- Test null hypothesis: errors indistinguishable from zero
- Apply Bonferroni correction for multiple testing
- Report p-values and confidence intervals
"
```

### Commit 43: Generate all validation visualizations
```bash
git add outputs/figures/ analysis/plot_validation.py
git commit -m "docs: create complete validation visualization suite

- Residual diagnostic plots (Q-Q, histogram, vs fitted)
- Cross-validation box plots
- Forward-walking MAE evolution
- Sensitivity heatmaps and bar charts
- Robustness test results
"
```

### Commit 44: Add performance benchmarking
```bash
git add analysis/benchmark_performance.py outputs/tables/
git commit -m "test(performance): benchmark ML vs traditional calibration

- Compare neural network vs scipy.optimize (L-BFGS-B)
- Compare vs differential evolution and grid search
- Measure speed (ms) and accuracy (RMSE)
- Generate benchmark comparison plots
"
```

---

## Phase 5: API & Deployment (7-9 commits)

### Commit 45: Create FastAPI server
```bash
git add api/main.py api/__init__.py
git commit -m "feat(api): implement FastAPI calibration server

- Add /calibrate endpoint with Pydantic validation
- Implement OptionSurface and CalibrationResult schemas
- Load model and scaler at startup
- Add request/response examples in docstrings
"
```

### Commit 46: Add API error handling
```bash
git add api/main.py api/errors.py
git commit -m "feat(api): add comprehensive error handling

- Validate input dimensions (strikes × maturities)
- Handle model loading failures gracefully
- Return proper HTTP status codes (400, 500)
- Add custom exception classes
"
```

### Commit 47: Implement health checks
```bash
git add api/main.py
git commit -m "feat(api): add health and monitoring endpoints

- Add /health endpoint for container orchestration
- Add /models endpoint listing available models
- Add /metrics endpoint for Prometheus scraping
- Return service status and model metadata
"
```

### Commit 48: Add API tests
```bash
git add tests/test_api.py api/test_api.py
git commit -m "test(api): add integration tests for API endpoints

- Use FastAPI TestClient for testing
- Test /calibrate with valid and invalid inputs
- Test error handling (422, 500 codes)
- Test latency (assert < 100ms)
"
```

### Commit 49: Create Dockerfile
```bash
git add Dockerfile .dockerignore
git commit -m "build: add Dockerfile for containerized deployment

- Use python:3.9-slim base image
- Copy only necessary files (models/, api/)
- Expose port 8000
- Use uvicorn as ASGI server
- Optimize layer caching
"
```

### Commit 50: Add docker-compose
```bash
git add docker-compose.yml
git commit -m "build: add docker-compose for orchestration

- Define api service with build context
- Add environment variables for config
- Mount models/ as read-only volume
- Add healthcheck with curl
- Expose port 8000
"
```

### Commit 51: Generate API documentation
```bash
git add docs/api_reference.md outputs/figures/
git commit -m "docs: create API documentation and examples

- Document all endpoints with examples
- Add curl and Python client examples
- Generate API architecture diagram
- Add latency benchmarks plot
"
```

### Commit 52: Optimize model loading
```bash
git add api/main.py api/model_cache.py
git commit -m "perf(api): optimize model loading and caching

- Load models lazily on first request
- Implement LRU cache for model artifacts
- Add warmup endpoint for preloading
- Reduce cold start time from 5s to 500ms
"
```

### Commit 53: Add request logging
```bash
git add api/main.py api/logging_config.py
git commit -m "feat(api): implement structured request logging

- Add JSON-formatted logs to stdout
- Log request_id, latency, status_code
- Add correlation IDs for tracing
- Configure log levels via environment
"
```

---

## Phase 6: Documentation & Finalization (5-7 commits)

### Commit 54: Generate system architecture diagrams
```bash
git add docs/diagrams/ outputs/figures/
git commit -m "docs: create system architecture diagrams

- Generate architecture.png using diagrams library
- Add data pipeline flow diagram
- Add ML workflow diagram
- Add API architecture diagram
"
```

### Commit 55: Generate all benchmark plots
```bash
git add outputs/figures/ analysis/generate_all_plots.py
git commit -m "docs: generate complete benchmark and comparison plots

- ML vs traditional optimization benchmark
- Model comparison (MLP, CNN, ResNet, Ensemble)
- Speed-accuracy tradeoff Pareto frontier
- Model size vs accuracy scatter plot
"
```

### Commit 56: Create interactive dashboard
```bash
git add analysis/generate_dashboard.py outputs/figures/
git commit -m "docs: create interactive results dashboard

- Build Plotly dashboard with training curves
- Add interactive prediction scatter plots
- Add parameter distribution histograms
- Save as interactive_dashboard.html
"
```

### Commit 57: Add example Jupyter notebooks
```bash
git add notebooks/01_quickstart.ipynb notebooks/02_advanced_calibration.ipynb
git commit -m "docs: add example notebooks for common use cases

- 01_quickstart.ipynb: Basic calibration workflow
- 02_advanced_calibration.ipynb: Bayesian MCMC example
- Include visualizations and explanations
- Store in notebooks/ directory
"
```

### Commit 58: Generate executive summary
```bash
git add outputs/reports/executive_summary.pdf analysis/generate_summary.py
git commit -m "docs: create one-page executive summary

- Summarize problem, solution, results
- Include key metrics and visualizations
- Format for recruiters and stakeholders
- Generate PDF with matplotlib
"
```

### Commit 59: Final cleanup and organization
```bash
git add .
git commit -m "chore: final repository cleanup and organization

- Remove unused files and debug code
- Organize imports and add __init__.py files
- Update all docstrings for completeness
- Verify all tests pass
"
```

### Commit 60: Update all documentation cross-references
```bash
git add README.md docs/ CLAUDE.md
git commit -m "docs: update all documentation with final cross-references

- Link README to all new visualizations
- Update PLAN.md with completion status
- Add references to notebooks in README
- Update CLAUDE.md with final structure
"
```

---

## Bonus Commits (As Needed)

### Bug Fixes
```bash
git commit -m "fix(scope): brief description of bug fix

- Describe what was broken
- Explain root cause
- Describe solution
"
```

### Performance Improvements
```bash
git commit -m "perf(scope): brief description of optimization

- Measure before/after performance
- Explain optimization technique
- Quantify speedup
"
```

### Refactoring
```bash
git commit -m "refactor(scope): brief description of refactoring

- Explain motivation (readability, maintainability)
- No functional changes
- Preserve existing tests
"
```

---

## Commit Summary

**Total Estimated Commits**: 60+ atomic commits

**By Phase**:
- Phase 0 (Setup): 6 commits
- Phase 1 (Pricing): 9 commits
- Phase 2 (Neural Nets): 10 commits
- Phase 3 (Bayesian): 8 commits
- Phase 4 (Validation): 11 commits
- Phase 5 (API): 9 commits
- Phase 6 (Documentation): 7 commits

**Commit Types**:
- `feat`: 28 commits (new features)
- `docs`: 14 commits (documentation)
- `test`: 9 commits (testing)
- `perf`: 4 commits (performance)
- `fix`: 2 commits (bug fixes)
- `refactor`: 1 commit (code refactoring)
- `build`: 2 commits (build system)
- `chore`: 2 commits (maintenance)

---

## Push Strategy

### Batch Pushes by Phase

After completing each phase, push all commits:

```bash
# Phase 0
git push origin master  # Commits 1-6

# Phase 1
git push origin master  # Commits 7-15

# Phase 2
git push origin master  # Commits 16-25

# Phase 3
git push origin master  # Commits 26-33

# Phase 4
git push origin master  # Commits 34-44

# Phase 5
git push origin master  # Commits 45-53

# Phase 6
git push origin master  # Commits 54-60
```

### Daily Pushes (Alternative)

Push at end of each day to backup work:

```bash
git push origin master --tags
```

---

## Verification Checklist

Before each push, verify:
- ✅ All tests pass: `pytest`
- ✅ Code formatted: `black . && flake8 .`
- ✅ No merge conflicts
- ✅ Commit messages follow convention
- ✅ No secrets committed (API keys, credentials)

---

**Last Updated**: 2025-10-01
**Status**: Ready for implementation
