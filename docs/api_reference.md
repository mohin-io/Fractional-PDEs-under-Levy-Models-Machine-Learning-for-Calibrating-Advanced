# API Reference

Complete reference for the Lévy Model Calibration API.

## Base URL

```
http://localhost:8000
```

For production deployments, replace with your actual domain.

---

## Endpoints

### `GET /`

Root endpoint returning API information.

**Response:**
```json
{
  "message": "Lévy Model Calibration API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### `GET /health`

Health check endpoint for monitoring and container orchestration.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Model not loaded or service degraded

**Use Cases:**
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Monitoring dashboards

---

### `GET /models`

Get information about available calibration models.

**Response:**
```json
{
  "available_models": ["VarianceGamma", "CGMY"],
  "default_model": "VarianceGamma",
  "expected_input_dimension": 200
}
```

**Field Descriptions:**
- `available_models`: List of supported Lévy models
- `default_model`: Model used if not specified in request
- `expected_input_dimension`: Required number of option prices (strikes × maturities)

---

### `POST /calibrate`

**Main endpoint**: Calibrate Lévy model parameters from option price surface.

#### Request Body

```json
{
  "option_prices": [20.5, 15.3, 10.8, 25.2, ...],
  "strikes": [80, 90, 100, 80, ...],
  "maturities": [0.5, 0.5, 0.5, 1.0, ...],
  "model_name": "VarianceGamma",
  "spot_price": 100.0,
  "risk_free_rate": 0.05
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `option_prices` | `List[float]` | ✅ Yes | Flattened option price surface (length = strikes × maturities) |
| `strikes` | `List[float]` | ❌ No | Strike prices (for validation, optional) |
| `maturities` | `List[float]` | ❌ No | Time to maturity in years (for validation, optional) |
| `model_name` | `string` | ❌ No | `"VarianceGamma"` or `"CGMY"` (default: `"VarianceGamma"`) |
| `spot_price` | `float` | ❌ No | Current spot price S₀ (default: 100.0) |
| `risk_free_rate` | `float` | ❌ No | Risk-free rate r (default: 0.05) |

**Constraints:**
- All `option_prices` must be ≥ 0
- `spot_price` must be > 0
- `option_prices.length` must equal 200 (default grid: 20 strikes × 10 maturities)

#### Response

```json
{
  "model_name": "VarianceGamma",
  "parameters": {
    "sigma": 0.215,
    "nu": 0.342,
    "theta": -0.145
  },
  "inference_time_ms": 12.5,
  "input_dimension": 200,
  "success": true
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `string` | Model type used for calibration |
| `parameters` | `Dict[string, float]` | Calibrated model parameters |
| `inference_time_ms` | `float` | Inference latency in milliseconds |
| `input_dimension` | `int` | Number of input prices processed |
| `success` | `bool` | Whether calibration succeeded |

**Parameter Interpretation:**

**Variance Gamma Model:**
- `sigma`: Volatility of the underlying Brownian motion (σ > 0)
- `nu`: Variance rate of the time change (ν > 0)
- `theta`: Drift parameter (θ, typically negative for equities)

**CGMY Model:**
- `C`: Overall level of jump activity (C > 0)
- `G`: Decay rate of right tail (G > 0)
- `M`: Decay rate of left tail (M > 0)
- `Y`: Fine structure parameter (Y < 2)

#### Error Responses

**422 Unprocessable Entity** - Invalid input:
```json
{
  "error": "InvalidInputDimensionError",
  "message": "Invalid input dimension. Expected 200, received 150",
  "details": {
    "expected": 200,
    "received": 150
  }
}
```

**503 Service Unavailable** - Model not loaded:
```json
{
  "error": "ModelNotLoadedError",
  "message": "Model file not found: models/calibration_net/mlp_calibration_model.h5",
  "details": {
    "model_name": "VarianceGamma",
    "expected_path": "..."
  }
}
```

---

### `POST /warmup`

Preload all models into memory for faster subsequent requests.

**Response:**
```json
{
  "status": "success",
  "message": "Models warmed up successfully",
  "models_loaded": ["VarianceGamma", "CGMY"]
}
```

**Use Cases:**
- Reducing cold start latency after deployment
- Pre-loading models during container startup

---

### `DELETE /cache`

Clear the model cache, forcing reload on next request.

**Response:**
```json
{
  "status": "success",
  "message": "Model cache cleared"
}
```

**Use Cases:**
- Development/debugging
- Forcing model refresh after update

---

## Usage Examples

### cURL

**Basic calibration:**
```bash
curl -X POST "http://localhost:8000/calibrate" \
  -H "Content-Type: application/json" \
  -d '{
    "option_prices": [20.5, 15.3, 10.8, 8.2, 6.1, 25.2, 18.9, 12.4, 9.1, 6.8, ...],
    "model_name": "VarianceGamma",
    "spot_price": 100.0,
    "risk_free_rate": 0.05
  }'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

### Python

**Using `requests`:**

```python
import requests
import numpy as np

# Generate sample option prices (200 values)
np.random.seed(42)
option_prices = np.random.uniform(5, 25, 200).tolist()

# API request
response = requests.post(
    "http://localhost:8000/calibrate",
    json={
        "option_prices": option_prices,
        "model_name": "VarianceGamma",
        "spot_price": 100.0,
        "risk_free_rate": 0.05
    }
)

# Parse response
if response.status_code == 200:
    result = response.json()
    print(f"Model: {result['model_name']}")
    print(f"Parameters: {result['parameters']}")
    print(f"Inference time: {result['inference_time_ms']}ms")
else:
    print(f"Error {response.status_code}: {response.json()}")
```

**Using `httpx` (async):**

```python
import httpx
import asyncio

async def calibrate_async():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/calibrate",
            json={
                "option_prices": [20.5, 15.3, ...],  # 200 prices
                "model_name": "CGMY"
            }
        )
        return response.json()

result = asyncio.run(calibrate_async())
print(result)
```

---

### JavaScript (Node.js)

```javascript
const axios = require('axios');

async function calibrate() {
  try {
    const response = await axios.post('http://localhost:8000/calibrate', {
      option_prices: [20.5, 15.3, 10.8, ...], // 200 prices
      model_name: 'VarianceGamma',
      spot_price: 100.0,
      risk_free_rate: 0.05
    });

    console.log('Model:', response.data.model_name);
    console.log('Parameters:', response.data.parameters);
    console.log('Inference time:', response.data.inference_time_ms, 'ms');
  } catch (error) {
    console.error('Error:', error.response.data);
  }
}

calibrate();
```

---

## Interactive Documentation

The API provides interactive documentation via Swagger UI and ReDoc:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
  - Test endpoints directly in the browser
  - View request/response schemas
  - See example payloads

- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
  - Alternative documentation format
  - Better for reading and sharing

---

## Performance Benchmarks

Measured on Intel i7-10700K CPU @ 3.80GHz (no GPU):

| Model | Inference Time | Throughput |
|-------|----------------|------------|
| Variance Gamma | ~12-15 ms | ~70 req/s |
| CGMY | ~12-15 ms | ~70 req/s |

**Latency breakdown:**
- Model inference: ~5-8 ms
- Input preprocessing: ~2-3 ms
- Overhead (serialization, etc.): ~3-4 ms

**Optimization tips:**
- Use `/warmup` endpoint to preload models
- Batch multiple calibrations in parallel
- Deploy with GPU for 5-10× speedup
- Use connection pooling for high-throughput scenarios

---

## Deployment

### Docker

**Build image:**
```bash
docker build -t levy-calibration-api:latest .
```

**Run container:**
```bash
docker run -d \
  --name levy-api \
  -p 8000:8000 \
  -v $(pwd)/models/calibration_net:/app/models/calibration_net:ro \
  levy-calibration-api:latest
```

### Docker Compose

```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f api
```

Stop:
```bash
docker-compose down
```

---

## Error Handling

The API uses standard HTTP status codes:

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 422 | Unprocessable Entity | Invalid input (wrong dimension, negative prices, etc.) |
| 500 | Internal Server Error | Unexpected error during processing |
| 503 | Service Unavailable | Model not loaded or system not ready |

All error responses follow this schema:
```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "key": "value"
  }
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production deployments, consider:

- **NGINX/Traefik**: Reverse proxy with rate limiting
- **API Gateway**: AWS API Gateway, Kong, etc.
- **FastAPI middleware**: Custom rate limiting middleware

---

## Security Considerations

**Current implementation** (suitable for internal/development use):
- CORS allows all origins (`allow_origins=["*"]`)
- No authentication/authorization
- Models run as non-root user in container

**Production recommendations**:
1. **Authentication**: Add API keys or OAuth2
2. **CORS**: Restrict to specific origins
3. **HTTPS**: Use TLS certificates
4. **Input validation**: Add size limits, sanitization
5. **Monitoring**: Log all requests, add anomaly detection

---

**Version**: 1.0.0
**Last Updated**: 2025-10-01
