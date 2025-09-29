# Project Mindmap: Machine Learning for Calibrating Advanced Asset Pricing Models

This mindmap provides a high-level overview of the project's structure, key components, and their relationships.

```mermaid
mindmap
  root((Levy Model Calibration Engine))
    Project Overview
      Problem: Standard models fail (fat tails, jumps)
      Solution: ML for faster, accurate calibration
      Models: Variance Gamma, CGMY
    Phase 1: Scaffolding & Docs
      Directory Structure
      Initial Documentation
        README.md
        CONTRIBUTING.md
        CODE_OF_CONDUCT.md
        SECURITY.md
        CHANGELOG.md
        Project Report
      GitHub Pages Setup
    Phase 2: Data & Feature Engineering
      Fractional PDE Solver (Pricing Engine)
        levy_models.py (Char Funcs)
        fourier_pricer.py (Carr-Madan FFT)
      Synthetic Data Generation
        generate_dataset.py
        Output: data/synthetic/training_data.parquet
      Feature Engineering
        build_features.py
        Output: data/processed/features.parquet, targets.parquet
    Phase 3: Model Development & Training
      Calibration Network (MLP)
        models/calibration_net/model.py (Architecture)
        models/calibration_net/train.py (Training Loop)
        models/calibration_net/predict.py (Inference)
        Saved Model: mlp_calibration_model.h5
        Saved Scaler: scaler_X.pkl
      Bayesian Calibration (Advanced/Future)
        models/bayesian_calibration/mcmc.py (MCMC Placeholder)
    Phase 4: Backtesting & Validation
      Backtesting Engine
        backtesting/engine.py (Event-driven)
        backtesting/strategy.py (Option Arbitrage)
      Robust Validation
        analysis/out_of_sample.py
        analysis/forward_walking.py
        analysis/significance_testing.py
        analysis/sensitivity_analysis.py
    Phase 5: Productionization & Deployment
      API (FastAPI)
        production/api/main.py
        Endpoints: /calibrate, /health
      CI/CD (GitHub Actions)
        .github/workflows/ci.yml (Build, Test, Lint, Format)
        .github/workflows/cd.yml (Deploy Docs, API)
    Phase 6: Final Touches
      Project Identity
        README.md Banner
        Consistent Naming (Branches, Commits, Releases)
      Documentation
        docs/README.md (GitHub Pages Index)
        API Reference
        System Diagram
        Design Principles
        Mindmap (this file)
```
