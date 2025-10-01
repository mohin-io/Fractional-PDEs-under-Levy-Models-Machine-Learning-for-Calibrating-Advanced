"""
Model loading and caching utilities.
"""

import os
import pickle
import logging
from typing import Optional, Tuple, Any
from pathlib import Path
import tensorflow as tf
from api.errors import ModelNotLoadedError, ScalerNotFoundError

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Singleton class for loading and caching calibration models.

    Implements lazy loading: models are loaded on first request
    and cached in memory for subsequent requests.
    """

    _instance: Optional['ModelCache'] = None
    _models: dict = {}
    _scalers: dict = {}

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model cache."""
        self.base_path = Path(__file__).parent.parent / "models" / "calibration_net"
        logger.info(f"ModelCache initialized with base path: {self.base_path}")

    def load_model(self, model_name: str = "VarianceGamma") -> tf.keras.Model:
        """
        Load calibration model from disk.

        Args:
            model_name: Model type ('VarianceGamma' or 'CGMY')

        Returns:
            Loaded Keras model

        Raises:
            ModelNotLoadedError: If model file not found or loading fails
        """
        # Return cached model if available
        if model_name in self._models:
            logger.debug(f"Returning cached model: {model_name}")
            return self._models[model_name]

        # Determine model path
        if model_name == "VarianceGamma":
            model_path = self.base_path / "mlp_calibration_model.h5"
        elif model_name == "CGMY":
            model_path = self.base_path / "cgmy_calibration_model.h5"
        else:
            raise ModelNotLoadedError(
                message=f"Unknown model name: {model_name}",
                details={"model_name": model_name}
            )

        # Check if file exists
        if not model_path.exists():
            raise ModelNotLoadedError(
                message=f"Model file not found: {model_path}",
                details={
                    "model_name": model_name,
                    "expected_path": str(model_path)
                }
            )

        # Load model
        try:
            logger.info(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(str(model_path))
            self._models[model_name] = model
            logger.info(f"Model loaded successfully: {model_name}")
            return model
        except Exception as e:
            raise ModelNotLoadedError(
                message=f"Failed to load model: {str(e)}",
                details={
                    "model_name": model_name,
                    "path": str(model_path),
                    "error": str(e)
                }
            )

    def load_scaler(self, model_name: str = "VarianceGamma") -> Any:
        """
        Load feature scaler from disk.

        Args:
            model_name: Model type ('VarianceGamma' or 'CGMY')

        Returns:
            Loaded StandardScaler object

        Raises:
            ScalerNotFoundError: If scaler file not found or loading fails
        """
        # Return cached scaler if available
        if model_name in self._scalers:
            logger.debug(f"Returning cached scaler: {model_name}")
            return self._scalers[model_name]

        # Determine scaler path
        if model_name == "VarianceGamma":
            scaler_path = self.base_path / "scaler_X.pkl"
        elif model_name == "CGMY":
            scaler_path = self.base_path / "scaler_X_cgmy.pkl"
        else:
            raise ScalerNotFoundError(
                message=f"Unknown model name: {model_name}",
                details={"model_name": model_name}
            )

        # Check if file exists
        if not scaler_path.exists():
            raise ScalerNotFoundError(
                message=f"Scaler file not found: {scaler_path}",
                details={
                    "model_name": model_name,
                    "expected_path": str(scaler_path)
                }
            )

        # Load scaler
        try:
            logger.info(f"Loading scaler from: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            self._scalers[model_name] = scaler
            logger.info(f"Scaler loaded successfully: {model_name}")
            return scaler
        except Exception as e:
            raise ScalerNotFoundError(
                message=f"Failed to load scaler: {str(e)}",
                details={
                    "model_name": model_name,
                    "path": str(scaler_path),
                    "error": str(e)
                }
            )

    def load_model_and_scaler(self, model_name: str = "VarianceGamma") -> Tuple[tf.keras.Model, Any]:
        """
        Load both model and scaler.

        Args:
            model_name: Model type ('VarianceGamma' or 'CGMY')

        Returns:
            Tuple of (model, scaler)
        """
        model = self.load_model(model_name)
        scaler = self.load_scaler(model_name)
        return model, scaler

    def is_model_loaded(self, model_name: str = "VarianceGamma") -> bool:
        """
        Check if model is already loaded in cache.

        Args:
            model_name: Model type

        Returns:
            True if model is cached, False otherwise
        """
        return model_name in self._models and model_name in self._scalers

    def clear_cache(self):
        """Clear all cached models and scalers."""
        logger.info("Clearing model cache")
        self._models.clear()
        self._scalers.clear()

    def warmup(self):
        """
        Preload all available models for faster first request.

        Call this during application startup.
        """
        logger.info("Warming up model cache")
        for model_name in ["VarianceGamma", "CGMY"]:
            try:
                self.load_model_and_scaler(model_name)
                logger.info(f"Warmup successful: {model_name}")
            except (ModelNotLoadedError, ScalerNotFoundError) as e:
                logger.warning(f"Warmup skipped for {model_name}: {e.message}")


# Global singleton instance
model_cache = ModelCache()
