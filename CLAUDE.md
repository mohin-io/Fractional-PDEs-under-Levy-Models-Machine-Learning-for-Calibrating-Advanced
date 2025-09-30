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
   python models/generate_dataset.py
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
   python models/calibration_net/train.py
   ```
   - Trains MLP neural network on the processed data
   - Saves model to `models/calibration_net/mlp_calibration_model.h5`
   - Saves feature scaler to `models/calibration_net/scaler_X.pkl`

4. **Run predictions:**
   ```bash
   python models/calibration_net/predict.py
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
- `carr_madan_pricer()`: FFT-based European option pricing using characteristic functions
- `price_surface()`: Generates full option price surface across strikes and maturities
- Solves the "forward problem": parameters → prices

**Calibration Network** ([models/calibration_net/](models/calibration_net/))
- `model.py`: MLP architecture (256→128→64 neurons with dropout)
- `train.py`: Training pipeline with StandardScaler preprocessing
- `predict.py`: Inference for parameter estimation
- Solves the "inverse problem": prices → parameters

**Data Generation** ([models/generate_dataset.py](models/generate_dataset.py))
- Uses Sobol sequences for quasi-random sampling of parameter space
- Prices options via Fourier methods to create (surface, params) pairs
- Fixed grid: 20 strikes (80-120), 10 maturities (0.1-2.0 years)
- The grid dimensions MUST match between data generation and feature engineering

**Feature Engineering** ([features/build_features.py](features/build_features.py))
- Splits synthetic data into features (flattened price surfaces) and targets (model parameters)
- Features are columns matching `price_*` pattern
- Targets are model parameters: sigma, nu, theta (VG) or C, G, M, Y (CGMY)

### Directory Structure

- `models/pricing_engine/`: Lévy models and Fourier-based option pricing (forward problem)
- `models/calibration_net/`: Neural network for parameter calibration (inverse problem)
- `models/bayesian_calibration/`: MCMC-based Bayesian calibration alternative
- `data/synthetic/`: Generated training data from pricing engine
- `data/processed/`: Features and targets ready for ML training
- `features/`: Feature engineering scripts
- `analysis/`: Statistical analysis (out-of-sample, sensitivity, forward walking)
- `tests/`: pytest test suite
- `api/`: FastAPI endpoints (currently empty, planned for deployment)

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

## CI/CD

GitHub Actions workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on push/PR to master:
1. Install dependencies (including flake8, black, pytest)
2. Run pytest test suite
3. Lint with flake8
4. Check formatting with Black

Ensure all checks pass before merging.
