"""
Custom exception classes and error handlers for the API.
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Base exception for calibration errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedError(CalibrationError):
    """Raised when model files cannot be loaded."""

    pass


class InvalidInputDimensionError(CalibrationError):
    """Raised when input dimension doesn't match expected."""

    pass


class PricingEngineError(CalibrationError):
    """Raised when pricing engine fails."""

    pass


class ScalerNotFoundError(CalibrationError):
    """Raised when feature scaler is not found."""

    pass


async def calibration_error_handler(request: Request, exc: CalibrationError):
    """
    Handle custom calibration errors.

    Args:
        request: The FastAPI request
        exc: The calibration exception

    Returns:
        JSONResponse with error details
    """
    logger.error(f"Calibration error: {exc.message}", extra={"details": exc.details})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


async def model_not_loaded_error_handler(request: Request, exc: ModelNotLoadedError):
    """
    Handle model loading errors.

    Args:
        request: The FastAPI request
        exc: The model not loaded exception

    Returns:
        JSONResponse with error details
    """
    logger.error(f"Model loading error: {exc.message}", extra={"details": exc.details})

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "ModelNotLoadedError",
            "message": exc.message,
            "details": exc.details
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions.

    Args:
        request: The FastAPI request
        exc: The exception

    Returns:
        JSONResponse with error details
    """
    logger.exception("Unexpected error occurred", exc_info=exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"exception_type": exc.__class__.__name__}
        }
    )


def raise_for_invalid_dimension(received: int, expected: int):
    """
    Raise exception if input dimension doesn't match expected.

    Args:
        received: Received input dimension
        expected: Expected input dimension

    Raises:
        InvalidInputDimensionError: If dimensions don't match
    """
    if received != expected:
        raise InvalidInputDimensionError(
            message=f"Invalid input dimension. Expected {expected}, received {received}",
            details={"expected": expected, "received": received}
        )
