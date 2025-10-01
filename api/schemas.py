"""
Pydantic schemas for API request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import numpy as np


class OptionSurfaceRequest(BaseModel):
    """
    Request schema for calibration endpoint.

    Expects flattened option price surface along with metadata.
    """

    option_prices: List[float] = Field(
        ...,
        description="Flattened option price surface (strikes × maturities)",
        min_items=1
    )

    strikes: Optional[List[float]] = Field(
        None,
        description="Strike prices (optional, for validation)"
    )

    maturities: Optional[List[float]] = Field(
        None,
        description="Time to maturity in years (optional, for validation)"
    )

    model_name: str = Field(
        "VarianceGamma",
        description="Model type: 'VarianceGamma' or 'CGMY'"
    )

    spot_price: float = Field(
        100.0,
        gt=0,
        description="Current spot price (S0)"
    )

    risk_free_rate: float = Field(
        0.05,
        description="Risk-free interest rate"
    )

    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name is supported."""
        allowed = ['VarianceGamma', 'CGMY']
        if v not in allowed:
            raise ValueError(f"model_name must be one of {allowed}")
        return v

    @validator('option_prices')
    def validate_prices(cls, v):
        """Validate all prices are non-negative."""
        if any(p < 0 for p in v):
            raise ValueError("All option prices must be non-negative")
        return v

    class Config:
        schema_extra = {
            "example": {
                "option_prices": [20.5, 15.3, 10.8, 25.2, 18.9, 12.4],
                "strikes": [80, 90, 100, 80, 90, 100],
                "maturities": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
                "model_name": "VarianceGamma",
                "spot_price": 100.0,
                "risk_free_rate": 0.05
            }
        }


class CalibrationResult(BaseModel):
    """
    Response schema for calibration endpoint.

    Returns calibrated model parameters and diagnostics.
    """

    model_name: str = Field(..., description="Model type used for calibration")

    parameters: Dict[str, float] = Field(
        ...,
        description="Calibrated model parameters"
    )

    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )

    input_dimension: int = Field(
        ...,
        description="Number of input prices used"
    )

    success: bool = Field(
        True,
        description="Whether calibration succeeded"
    )

    class Config:
        schema_extra = {
            "example": {
                "model_name": "VarianceGamma",
                "parameters": {
                    "sigma": 0.215,
                    "nu": 0.342,
                    "theta": -0.145
                },
                "inference_time_ms": 12.5,
                "input_dimension": 200,
                "success": True
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response."""

    available_models: List[str] = Field(
        ...,
        description="List of available calibration models"
    )

    default_model: str = Field(
        ...,
        description="Default model used if not specified"
    )

    expected_input_dimension: int = Field(
        ...,
        description="Expected number of input prices (strikes × maturities)"
    )

    class Config:
        schema_extra = {
            "example": {
                "available_models": ["VarianceGamma", "CGMY"],
                "default_model": "VarianceGamma",
                "expected_input_dimension": 200
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input dimension",
                "details": {
                    "expected": 200,
                    "received": 150
                }
            }
        }
