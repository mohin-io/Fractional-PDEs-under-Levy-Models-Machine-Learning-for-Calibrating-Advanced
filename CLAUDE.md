# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a machine learning framework for calibrating Lévy-based stochastic models (Variance Gamma, CGMY) in quantitative finance using deep learning. The system learns to map option price surfaces to model parameters, enabling near-instantaneous calibration compared to traditional optimization methods.

## Key Commands

### Setup
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Linting & Formatting
```bash
# Lint with flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics

# Format with Black
black .

# Check formatting without making changes
black --check .
```

### Data Pipeline & Training

The typical workflow is:

1. **Generate synthetic training data:**
   ```bash
   python models/generate_dataset.py --num_samples 100000
   python models/generate_dataset_cgmy.py --num_samples 100000
   ```
   - Generates synthetic option surfaces using Fourier pricing
   - Uses Sobol quasi-random sampling for uniform parameter space coverage
   - Outputs to `data/synthetic/training_data.parquet`
   - Configuration in [models/generate_dataset.py](models/generate_dataset.py): `NUM_SAMPLES`, `MODEL_NAME`, `PARAM_RANGES`

2. **Build features and targets:**
   ```bash
   python features/build_features.py
   ```
   - Separates features (price surfaces) from targets (model parameters)
   - Outputs to `data/processed/features.parquet` and `data/processed/targets.parquet`

3. **Train the calibration model:**
   ```bash
   python models/calibration_net/train.py --architecture mlp --epochs 50
   python models/calibration_net/train.py --architecture cnn --epochs 50
   python models/calibration_net/train.py --architecture resnet --epochs 50
   ```
   - Trains neural network on the processed data
   - Saves model to `models/calibration_net/mlp_calibration_model.h5`
   - Saves feature scaler to `models/calibration_net/scaler_X.pkl`

4. **Run predictions:**
   ```bash
   python models/calibration_net/predict.py
   ```

5. **Start production API:**
   ```bash
   # Local development
   uvicorn api.main:app --reload --port 8000

   # Docker deployment
   docker-compose up -d
   ```

## Architecture

### Core Workflow
```
Market Data → Fourier Pricing (Forward Problem) → Synthetic Dataset
                                                         ↓
                                                   Feature Engineering
                                                         ↓
                                                   Neural Network Training
                                                         ↓
Option Surface → Trained Model (Inverse Problem) → Lévy Parameters
```

### Key Components

**Lévy Models** ([models/pricing_engine/levy_models.py](models/pricing_engine/levy_models.py))
- `variance_gamma_char_func()`: Characteristic function for VG process (params: sigma, nu, theta)
- `cgmy_char_func()`: Characteristic function for CGMY process (params: C, G, M, Y)
- These define the stochastic processes used in option pricing

**Fourier Pricing** ([models/pricing_engine/fourier_pricer.py](models/pricing_engine/fourier_pricer.py))
- `carr_madan_pricer()`: Enhanced FFT-based European option pricing
  - CubicSpline interpolation for improved accuracy
  - Supports both call and put options via put-call parity
  - Dividend yield (q) parameter support
  - Increased FFT resolution (N=2^12, eta=0.1)
- `compute_greeks()`: Greeks computation via finite differences (Delta, Gamma, Theta, Rho)
- `price_surface()`: Generates full option price surface across strikes and maturities
- Solves the "forward problem": parameters → prices

**Calibration Network** ([models/calibration_net/](models/calibration_net/))
- `model.py`: Enhanced MLP with batch normalization and L2 regularization
  - Configurable architecture (default: 256→128→64)
  - `get_callbacks()` for early stopping, LR scheduling, checkpointing
- `architectures.py`: Advanced model architectures
  - **CNN**: Treats option surface as 2D image with Conv2D layers
  - **ResNet**: Residual blocks with skip connections for deep networks
  - **CalibrationEnsemble**: Combines multiple models (averaging, weighted, stacking)
- `train.py`: Optimized training pipeline
  - TensorFlow Dataset API with prefetching
  - Mixed precision training support
  - CLI arguments for architecture selection
  - Saves training history to JSON
- `predict.py`: Inference for parameter estimation
- Solves the "inverse problem": prices → parameters

**Data Generation**
- [models/generate_dataset.py](models/generate_dataset.py): Variance Gamma dataset
- [models/generate_dataset_cgmy.py](models/generate_dataset_cgmy.py): CGMY dataset
- [models/dataset_utils.py](models/dataset_utils.py): Shared utilities
  - `generate_sobol_samples()`: Quasi-random parameter sampling
  - `add_market_noise()`: Simulates bid-ask spreads and measurement noise
  - `generate_synthetic_dataset()`: Unified dataset creation
- CLI arguments: `--num_samples`, `--add_noise`, `--noise_level`
- Fixed grid: 20 strikes (80-120), 10 maturities (0.1-2.0 years)
- The grid dimensions MUST match between data generation and feature engineering

**Feature Engineering** ([features/build_features.py](features/build_features.py))
- Splits synthetic data into features (flattened price surfaces) and targets (model parameters)
- Features are columns matching `price_*` pattern
- Targets are model parameters: sigma, nu, theta (VG) or C, G, M, Y (CGMY)

**Model Analysis** ([analysis/](analysis/))
- `model_comparison.py`: Compare MLP, CNN, ResNet, Ensemble
  - Accuracy metrics (MSE, MAE, R²)
  - Inference speed benchmarking
  - Robustness testing with noise injection
- `residual_analysis.py`: Statistical diagnostics for model residuals
  - Normality tests (Shapiro-Wilk, KS, D'Agostino)
  - Q-Q plots, heteroscedasticity testing (Breusch-Pagan)
- `cross_validation.py`: K-fold cross-validation framework
- `sensitivity_analysis_enhanced.py`: Jacobian and Sobol sensitivity indices
- `robustness_tests.py`: Noise injection, OOD detection, missing data tests

**Bayesian Calibration** ([models/bayesian_calibration/](models/bayesian_calibration/))
- `mcmc.py`: Full Bayesian inference using TensorFlow Probability NUTS
  - **BayesianCalibrator** class with multi-chain MCMC
  - Informative priors: LogNormal(σ), Gamma(ν), Normal(θ)
  - Posterior summaries, credible intervals
  - CLI: `--model`, `--samples`, `--burnin`, `--chains`
- `uncertainty_propagation.py`: Uncertainty quantification
  - Prediction intervals for single options
  - Surface-wide uncertainty
  - Coverage probability testing
- `diagnostics.py`: Convergence diagnostics
  - R-hat, ESS, MCSE computation
  - Trace plots, posterior distributions, correlations
  - Full report generation

**Production API** ([api/](api/))
- `main.py`: FastAPI application with production-ready endpoints
  - `/calibrate`: Main calibration endpoint (~12-15ms latency)
  - `/health`: Health checks for container orchestration
  - `/models`: List available models
  - `/warmup`: Preload models for faster first request
  - `/cache`: Clear model cache
- `schemas.py`: Pydantic request/response validation
  - `OptionSurfaceRequest`: Input validation (prices, strikes, maturities)
  - `CalibrationResult`: Structured output with timing metrics
  - `HealthResponse`, `ModelInfoResponse`, `ErrorResponse`
- `errors.py`: Custom exception hierarchy
  - `CalibrationError`, `ModelNotLoadedError`, `InvalidInputDimensionError`
  - Centralized error handling with detailed messages
- `model_loader.py`: Singleton model cache with lazy loading
  - Caches TensorFlow models and StandardScalers
  - Warmup functionality for production deployments

**Deployment**
- `Dockerfile`: Multi-stage build with security best practices
  - Non-root user, health checks, resource limits
- `docker-compose.yml`: Production orchestration
  - Environment variables, volume mounts, networking
- `notebooks/`: Jupyter examples
  - `01_quickstart.ipynb`: Basic calibration workflow
  - `02_advanced_calibration.ipynb`: Bayesian MCMC with uncertainty

### Directory Structure

- `models/pricing_engine/`: Lévy models and Fourier-based option pricing (forward problem)
- `models/calibration_net/`: Neural network for parameter calibration (inverse problem)
- `models/bayesian_calibration/`: MCMC-based Bayesian calibration with uncertainty quantification
- `data/synthetic/`: Generated training data from pricing engine
- `data/processed/`: Features and targets ready for ML training
- `features/`: Feature engineering scripts
- `analysis/`: Statistical analysis (residual analysis, sensitivity, cross-validation, robustness)
- `api/`: Production FastAPI server with Docker deployment
- `notebooks/`: Jupyter notebook examples and tutorials
- `tests/`: pytest test suite
- `docs/`: Documentation (PLAN.md, ARCHITECTURE.md, api_reference.md)

## Development Conventions

### Commit Message Format
Follow [Conventional Commits](https://www.conventionalcommits.org/):
- Format: `<type>(<scope>): <description>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`
- Scopes: `api`, `pricing-engine`, `calibration-net`, `docs`, etc.
- Examples:
  - `feat(pricing-engine): add CGMY model characteristic function`
  - `fix(api): handle missing option_prices in request`

### Branch Naming
- Feature: `feature/<short-description>`
- Bugfix: `bugfix/<issue-number>-<short-description>`
- Hotfix: `hotfix/<short-description>`
- Release: `release/<version>`

### Code Style
- 2 spaces for indentation (per CONTRIBUTING.md, though Python typically uses 4)
- Max line length: 120 characters
- Use Black for formatting
- Use flake8 for linting

## Important Implementation Details

### Option Pricing Grid Consistency
The strike and maturity grids defined in [models/generate_dataset.py](models/generate_dataset.py) (`GRID_STRIKES`, `GRID_MATURITIES`) MUST match exactly with those expected during feature engineering and inference. Changing grid dimensions requires regenerating the entire dataset.

### Model Parameter Constraints
- **Variance Gamma**: sigma > 0, nu > 0, theta typically negative for equities
- **CGMY**: C > 0, G > 0, M > 0, Y < 2 (enforced in code)

### Scaler Management
The `StandardScaler` fitted during training MUST be saved and reused during inference. The current implementation saves it as `scaler_X.pkl`. Never fit a new scaler on production data.

### Testing Philosophy
Tests focus on:
- Characteristic functions return complex numbers
- Pricers return positive prices
- Surfaces have correct dimensionality
- See [tests/test_models.py](tests/test_models.py)
