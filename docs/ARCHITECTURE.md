# System Architecture: Lévy Model Calibration Engine

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │   Jupyter      │  │   REST API     │  │   CLI Tool     │        │
│  │   Notebook     │  │  (FastAPI)     │  │   (Argparse)   │        │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘        │
└───────────┼──────────────────┼──────────────────┼─────────────────┘
            │                  │                  │
            └──────────────────┴──────────────────┘
                               │
    ┌──────────────────────────┴──────────────────────────┐
    │                                                       │
┌───▼────────────────────┐                   ┌────────────▼─────────┐
│  CALIBRATION ENGINE    │                   │   PRICING ENGINE     │
│                        │                   │                      │
│  ┌──────────────────┐  │                   │  ┌─────────────────┐ │
│  │ Neural Network   │  │◄──────────────────┤  │ Fourier Pricer  │ │
│  │  - MLP           │  │  Training Data    │  │  (Carr-Madan)   │ │
│  │  - CNN           │  │                   │  │                 │ │
│  │  - ResNet        │  │                   │  └─────────────────┘ │
│  │  - Ensemble      │  │                   │                      │
│  └──────────────────┘  │                   │  ┌─────────────────┐ │
│                        │                   │  │ PIDE Solver     │ │
│  ┌──────────────────┐  │                   │  │ (Optional)      │ │
│  │ Bayesian MCMC    │  │                   │  └─────────────────┘ │
│  │  - PyMC3         │  │                   │                      │
│  │  - NumPyro       │  │                   │  ┌─────────────────┐ │
│  │  - Variational   │  │                   │  │ Lévy Models     │ │
│  └──────────────────┘  │                   │  │ - VG Char Func  │ │
│                        │                   │  │ - CGMY Char Func│ │
└────────────────────────┘                   │  └─────────────────┘ │
            │                                └──────────────────────┘
            │                                           │
            │                                           │
    ┌───────▼────────────┐                   ┌─────────▼────────────┐
    │  VALIDATION SUITE  │                   │   DATA PIPELINE      │
    │                    │                   │                      │
    │  ┌──────────────┐  │                   │  ┌────────────────┐  │
    │  │ Out-of-Sample│  │                   │  │ Sobol Sampling │  │
    │  │ Testing      │  │                   │  │                │  │
    │  └──────────────┘  │                   │  └────────────────┘  │
    │                    │                   │                      │
    │  ┌──────────────┐  │                   │  ┌────────────────┐  │
    │  │ Forward      │  │                   │  │ Feature        │  │
    │  │ Walking      │  │                   │  │ Engineering    │  │
    │  └──────────────┘  │                   │  └────────────────┘  │
    │                    │                   │                      │
    │  ┌──────────────┐  │                   │  ┌────────────────┐  │
    │  │ Sensitivity  │  │                   │  │ Data Cleaning  │  │
    │  │ Analysis     │  │                   │  │                │  │
    │  └──────────────┘  │                   │  └────────────────┘  │
    └────────────────────┘                   └──────────────────────┘
```

## Component Details

### 1. Pricing Engine (`models/pricing_engine/`)

**Purpose**: Solve the forward problem (parameters → option prices)

**Components**:
- **`levy_models.py`**: Mathematical characteristic functions
  - Variance Gamma: φ(u; σ, ν, θ)
  - CGMY: φ(u; C, G, M, Y)

- **`fourier_pricer.py`**: Fast Fourier Transform pricing
  - Carr-Madan FFT method
  - Greeks via finite differences
  - Put-call parity for puts

- **`pide_solver.py`** (Future): Alternative PIDE-based pricing
  - Finite difference for diffusion
  - Quadrature for jump integral

**Key Algorithm: Carr-Madan FFT**
```
1. Define characteristic function φ(u) for log-price
2. Compute damped characteristic function: ψ(u) = φ(u - iα)
3. FFT of integrand: exp(-ru)ψ(u) / (α² + α - u² + iu(2α + 1))
4. Interpolate to desired strikes K
5. Return call prices
```

**Mathematical Foundation**:
- Option price as Fourier transform of payoff
- Exploits O(N log N) FFT complexity vs O(N²) integration
- Achieves 0.1% accuracy with N=2048 points

---

### 2. Data Pipeline (`data/`, `features/`, `models/generate_dataset.py`)

**Purpose**: Generate synthetic training data for supervised learning

**Workflow**:
```
Parameter Sampling → Pricing → Feature Extraction → Storage
```

**Steps**:

1. **Parameter Sampling** (`models/generate_dataset.py`)
   - Sobol quasi-random sequences for uniform coverage
   - Realistic ranges based on market observations
   - 100,000+ parameter sets per model

2. **Option Surface Generation**
   - Fixed grid: 20 strikes × 10 maturities = 200 prices
   - Call prices computed via Fourier pricer
   - Stored as Parquet for efficient I/O

3. **Feature Engineering** (`features/build_features.py`)
   - Flatten 2D surface → 1D feature vector
   - Separate features (prices) from targets (parameters)
   - StandardScaler normalization

4. **Data Augmentation** (Optional)
   - Add bid-ask spread noise
   - Simulate missing data
   - Generate stress scenarios

**Storage**:
```
data/
├── synthetic/
│   └── training_data.parquet  (params + price surfaces)
├── processed/
│   ├── features.parquet       (X: price vectors)
│   └── targets.parquet        (y: parameters)
└── raw/                       (real market data, future)
```

---

### 3. Calibration Engine (`models/calibration_net/`, `models/bayesian_calibration/`)

**Purpose**: Solve the inverse problem (option prices → parameters)

#### 3.1 Neural Network Calibration

**Architecture Evolution**:

**Baseline MLP**:
```
Input(200) → Dense(256, ReLU) → Dropout(0.2)
          → Dense(128, ReLU) → Dropout(0.2)
          → Dense(64, ReLU)
          → Output(3/4)  [VG: 3 params, CGMY: 4 params]
```

**CNN for Spatial Features**:
```
Input(200) → Reshape(20×10×1)
          → Conv2D(32, 3×3) → MaxPool(2×2)
          → Conv2D(64, 3×3)
          → Flatten → Dense(128) → Output(3/4)
```

**ResNet with Skip Connections**:
```
Input → Dense(256)
     → [ResBlock(256) × 3]  # Each block: Dense → BN → ReLU → Dense + Skip
     → Dense(128)
     → Output(3/4)
```

**Training**:
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=1e-3 with decay)
- Regularization: Dropout, L2, Early stopping
- Validation: 20% holdout + 5-fold CV

**Files**:
- `model.py`: Architecture definitions
- `train.py`: Training loop with callbacks
- `predict.py`: Inference engine
- `ensemble.py`: Multi-model aggregation

#### 3.2 Bayesian Calibration

**Purpose**: Quantify uncertainty in parameter estimates

**MCMC Approach** (`models/bayesian_calibration/mcmc.py`):

**Probabilistic Model**:
```
Priors:
  σ ~ LogNormal(log(0.2), 0.5)
  ν ~ Gamma(2, 2)
  θ ~ Normal(-0.2, 0.2)

Likelihood:
  P_obs ~ Normal(P_model(σ, ν, θ), ε)
  where P_model = fourier_pricer(params)

Posterior:
  P(σ, ν, θ | P_obs) ∝ P(P_obs | σ, ν, θ) × P(σ) × P(ν) × P(θ)
```

**Sampling**:
- Algorithm: NUTS (No-U-Turn Sampler)
- Chains: 4 independent chains
- Samples: 5000 per chain (after 2000 warmup)
- Diagnostics: R-hat < 1.01, ESS > 1000

**Variational Inference** (`models/bayesian_calibration/variational.py`):
- ADVI (Automatic Differentiation Variational Inference)
- Approximate posterior as Gaussian
- 100× faster than MCMC, slight accuracy tradeoff

**Outputs**:
- Posterior mean (point estimate)
- 95% credible intervals
- Parameter correlations
- Predictive uncertainty for new options

---

### 4. Validation Suite (`analysis/`)

**Purpose**: Rigorous testing beyond standard train/test split

#### 4.1 Out-of-Sample Testing (`out_of_sample.py`)
- Random 20% holdout
- Scatter plots: actual vs predicted
- Residual diagnostics: Q-Q plots, normality tests
- Per-parameter metrics: R², MAE, RMSE

#### 4.2 Forward-Walking Validation (`forward_walking.py`)
- Simulates temporal deployment
- Expanding window: train [0:10k] → test [10k:12k], etc.
- Detects model drift over time
- Plots MAE evolution

#### 4.3 Sensitivity Analysis (`sensitivity_analysis.py`)
- **Global**: Sobol variance decomposition
  - First-order indices: individual param importance
  - Total-order indices: param + interactions
- **Local**: Jacobian ∂(pred_params)/∂(input_prices)
  - Identifies most informative strikes/maturities

#### 4.4 Statistical Significance (`significance_testing.py`)
- t-tests on prediction errors
- Null hypothesis: errors indistinguishable from zero
- Bonferroni correction for multiple tests

#### 4.5 Robustness Testing (`robustness_tests.py`)
- Adversarial noise injection
- Out-of-distribution detection
- Stress testing edge cases

---

### 5. Production API (`api/`)

**Purpose**: Serve calibration as a microservice

**Technology Stack**:
- **Framework**: FastAPI (async, auto docs)
- **Server**: Uvicorn (ASGI)
- **Containerization**: Docker + docker-compose
- **Monitoring**: Prometheus + Grafana

**Endpoints**:

**POST /calibrate**
```json
Request:
{
  "spot_price": 100.0,
  "risk_free_rate": 0.05,
  "strikes": [90, 95, 100, 105, 110],
  "maturities": [0.25, 0.5, 1.0],
  "prices": [[...], [...], [...]],
  "model_type": "VarianceGamma"
}

Response:
{
  "model_type": "VarianceGamma",
  "parameters": {"sigma": 0.23, "nu": 0.41, "theta": -0.15},
  "calibration_time_ms": 2.3,
  "fit_quality": {"rmse": 0.08, "relative_error": 0.012}
}
```

**GET /health**
```json
{"status": "healthy", "model_loaded": true}
```

**GET /models**
```json
{
  "models": ["VarianceGamma", "CGMY"],
  "input_dimensions": {"VarianceGamma": 200}
}
```

**Features**:
- Request validation via Pydantic
- Error handling with HTTP codes
- Rate limiting
- Logging (JSON structured logs)
- Health checks for orchestration

**Deployment**:
```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - MODEL_PATH=/app/models/mlp.h5
    healthcheck:
      test: ["CMD", "curl", "http://localhost:8000/health"]
      interval: 30s
```

---

## Data Flow Diagram

### Training Phase
```
┌─────────────────┐
│ Parameter Space │
│   (Sobol)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fourier Pricer  │  Forward Problem
│  (100k calls)   │  Params → Prices
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Eng.   │  Flatten surfaces
│  (Parquet I/O)  │  Normalize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Neural Network │  Inverse Problem
│   Training      │  Prices → Params
│  (50 epochs)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Registry │  Save .h5, .pkl
│  (Artifacts)    │
└─────────────────┘
```

### Inference Phase
```
┌─────────────────┐
│ Market Option   │
│   Prices        │  (from exchange)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  Scale, reshape
│  (StandardScaler)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Predict  │  Neural network
│  (< 10ms)       │  forward pass
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Calibrated     │  σ, ν, θ
│  Parameters     │  (or C,G,M,Y)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation     │  Re-price & check
│  (Optional)     │  RMSE < threshold
└─────────────────┘
```

---

## Technology Stack

### Core Libraries
| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Numerical | NumPy | 1.24+ | Array operations |
| Numerical | SciPy | 1.10+ | FFT, optimization |
| Data | Pandas | 2.0+ | DataFrames, Parquet I/O |
| ML Framework | TensorFlow | 2.12+ | Neural networks |
| Bayesian | PyMC3 | 5.0+ | MCMC sampling |
| Bayesian | NumPyro | 0.12+ | JAX-based inference |
| Viz | Matplotlib | 3.7+ | Static plots |
| Viz | Seaborn | 0.12+ | Statistical plots |
| Viz | Plotly | 5.14+ | Interactive dashboards |
| API | FastAPI | 0.95+ | REST endpoints |
| Server | Uvicorn | 0.22+ | ASGI server |
| Testing | Pytest | 7.3+ | Unit tests |

### Development Tools
- **Linting**: flake8, black
- **Type Checking**: mypy (optional)
- **Profiling**: cProfile, memory_profiler
- **Version Control**: Git + GitHub
- **Containerization**: Docker 20.10+

---

## Performance Characteristics

### Pricing Engine
- **Fourier Pricer**: 0.5ms per option (single strike/maturity)
- **Surface Generation**: 100ms per surface (200 options)
- **Dataset Generation**: ~3 hours for 100k samples (parallelizable)

### Neural Network
- **Training**: ~10 minutes (100k samples, 50 epochs, GPU)
- **Inference**: 2-5ms per calibration (batch size 1)
- **Memory**: ~500MB (loaded model + scaler)

### Bayesian MCMC
- **Sampling**: 5-15 minutes per calibration (5k samples × 4 chains)
- **Variational**: 30-60 seconds per calibration
- **Memory**: ~2GB (PyMC3 model + traces)

### API
- **Latency**: p50=3ms, p95=8ms, p99=15ms
- **Throughput**: ~200 req/s (single instance, CPU)
- **Startup**: <5 seconds (model loading)

---

## Scalability Considerations

### Horizontal Scaling
- **Stateless API**: Multiple replicas behind load balancer
- **Model Serving**: TensorFlow Serving or TorchServe
- **Caching**: Redis for frequently calibrated surfaces

### Vertical Scaling
- **GPU Acceleration**: TensorFlow GPU for batch inference
- **JAX/NumPyro**: GPU-accelerated MCMC (10x speedup)
- **Vectorization**: Batch multiple calibrations

### Data Scaling
- **Distributed Training**: Horovod for multi-GPU
- **Data Sharding**: Parquet partitioning by model type
- **Incremental Learning**: Fine-tune on new market data

---

## Security & Reliability

### Input Validation
- Schema validation (Pydantic)
- Range checks (positive prices, valid strikes)
- Size limits (max 1000 options per surface)

### Error Handling
- Graceful degradation (fallback to simpler model)
- Circuit breakers for external dependencies
- Timeout handling (max 30s per calibration)

### Monitoring
- **Metrics**: Request rate, latency, error rate
- **Alerts**: Model drift, anomalous predictions
- **Logging**: Structured JSON logs to stdout

### Model Governance
- **Versioning**: Track training data, hyperparams, metrics
- **Rollback**: Blue-green deployment for safe updates
- **A/B Testing**: Compare model versions in production

---

## Future Enhancements

1. **Multi-Asset Calibration**: Joint calibration for correlated assets
2. **Real-Time Market Data**: Integration with Bloomberg/Reuters
3. **Greeks from Calibrated Models**: Delta, Gamma, Vega surfaces
4. **Model Explainability**: SHAP values, attention visualization
5. **Active Learning**: Prioritize informative samples for retraining
6. **Transfer Learning**: Pre-train on one model, fine-tune on another

---

**Last Updated**: 2025-10-01
**Author**: Mohin Hasin
**Contact**: mohinhasin999@gmail.com
