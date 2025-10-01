"""
FastAPI application for Lévy Model Calibration Engine.

Provides REST API endpoints for calibrating Variance Gamma and CGMY models
from option price surfaces.

Example usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""

import time
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__
from api.schemas import (
    OptionSurfaceRequest,
    CalibrationResult,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from api.errors import (
    CalibrationError,
    ModelNotLoadedError,
    InvalidInputDimensionError,
    calibration_error_handler,
    model_not_loaded_error_handler,
    generic_exception_handler,
    raise_for_invalid_dimension
)
from api.model_loader import model_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI application.

    Handles model preloading during startup.
    """
    logger.info("Starting up API server...")

    # Optionally warmup models (can be slow, so disabled by default)
    # Uncomment the next line to preload models at startup
    # model_cache.warmup()

    logger.info("API server ready")
    yield
    logger.info("Shutting down API server...")


# Initialize FastAPI app
app = FastAPI(
    title="Lévy Model Calibration API",
    description=(
        "REST API for calibrating Lévy-based stochastic models (Variance Gamma, CGMY) "
        "from option price surfaces using deep learning."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(CalibrationError, calibration_error_handler)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)


@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    """
    Root endpoint.

    Returns:
        Welcome message with API information
    """
    return {
        "message": "Lévy Model Calibration API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for container orchestration.

    Returns:
        Health status and model availability
    """
    # Try to load default model to verify system health
    model_loaded = False
    try:
        model_cache.load_model("VarianceGamma")
        model_loaded = True
    except Exception as e:
        logger.warning(f"Health check: Model not loaded - {str(e)}")

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=__version__,
        model_loaded=model_loaded
    )


@app.get("/models", response_model=ModelInfoResponse, tags=["General"])
async def get_model_info() -> ModelInfoResponse:
    """
    Get information about available calibration models.

    Returns:
        Available models and expected input dimensions
    """
    return ModelInfoResponse(
        available_models=["VarianceGamma", "CGMY"],
        default_model="VarianceGamma",
        expected_input_dimension=200  # 20 strikes × 10 maturities
    )


@app.post(
    "/calibrate",
    response_model=CalibrationResult,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        503: {"model": ErrorResponse, "description": "Model Not Available"}
    },
    tags=["Calibration"]
)
async def calibrate(request: OptionSurfaceRequest) -> CalibrationResult:
    """
    Calibrate Lévy model parameters from option price surface.

    This endpoint takes a flattened option price surface and returns
    calibrated model parameters using a pre-trained neural network.

    Args:
        request: Option surface data with prices and metadata

    Returns:
        Calibrated model parameters and inference metrics

    Raises:
        InvalidInputDimensionError: If input dimension doesn't match expected
        ModelNotLoadedError: If model files cannot be loaded

    Example:
        ```bash
        curl -X POST "http://localhost:8000/calibrate" \\
             -H "Content-Type: application/json" \\
             -d '{
                   "option_prices": [20.5, 15.3, 10.8, ...],
                   "model_name": "VarianceGamma",
                   "spot_price": 100.0,
                   "risk_free_rate": 0.05
                 }'
        ```
    """
    start_time = time.perf_counter()

    try:
        # Load model and scaler
        model, scaler = model_cache.load_model_and_scaler(request.model_name)

        # Validate input dimension
        input_prices = np.array(request.option_prices).reshape(1, -1)
        expected_dim = model.input_shape[1]
        received_dim = input_prices.shape[1]

        raise_for_invalid_dimension(received_dim, expected_dim)

        # Preprocess input
        X_scaled = scaler.transform(input_prices)

        # Predict parameters
        predictions = model.predict(X_scaled, verbose=0)

        # Build parameter dictionary
        if request.model_name == "VarianceGamma":
            param_names = ["sigma", "nu", "theta"]
        elif request.model_name == "CGMY":
            param_names = ["C", "G", "M", "Y"]
        else:
            raise CalibrationError(
                message=f"Unknown model: {request.model_name}",
                details={"model_name": request.model_name}
            )

        parameters = {
            name: float(predictions[0, i])
            for i, name in enumerate(param_names)
        }

        # Calculate inference time
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        logger.info(
            f"Calibration successful: {request.model_name}, "
            f"time={inference_time_ms:.2f}ms, params={parameters}"
        )

        return CalibrationResult(
            model_name=request.model_name,
            parameters=parameters,
            inference_time_ms=round(inference_time_ms, 2),
            input_dimension=received_dim,
            success=True
        )

    except (InvalidInputDimensionError, ModelNotLoadedError):
        # Re-raise custom exceptions (handled by exception handlers)
        raise

    except Exception as e:
        logger.exception("Unexpected error during calibration")
        raise CalibrationError(
            message=f"Calibration failed: {str(e)}",
            details={"error_type": e.__class__.__name__}
        )


@app.post("/warmup", tags=["Admin"])
async def warmup_models():
    """
    Preload all models into memory for faster subsequent requests.

    Useful for reducing cold start latency after deployment.

    Returns:
        Status of warmup operation
    """
    try:
        model_cache.warmup()
        return {
            "status": "success",
            "message": "Models warmed up successfully",
            "models_loaded": ["VarianceGamma", "CGMY"]
        }
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        return {
            "status": "partial",
            "message": f"Warmup completed with errors: {str(e)}"
        }


@app.delete("/cache", tags=["Admin"])
async def clear_cache():
    """
    Clear the model cache.

    Useful for forcing model reload during development.

    Returns:
        Status of cache clear operation
    """
    model_cache.clear_cache()
    return {
        "status": "success",
        "message": "Model cache cleared"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
