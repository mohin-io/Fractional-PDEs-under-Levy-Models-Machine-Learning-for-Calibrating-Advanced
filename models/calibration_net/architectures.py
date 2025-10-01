"""
Advanced neural network architectures for Lévy model calibration.

This module provides various deep learning architectures beyond the baseline MLP:
- CNN: Convolutional Neural Network treating option surfaces as 2D images
- ResNet: Residual Network with skip connections for deep architectures
- Ensemble: Combining multiple models for improved robustness
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(input_shape, output_dim, num_strikes=20, num_maturities=10,
                   learning_rate=0.001):
    """
    Build CNN model treating option surface as 2D image.

    The option price surface has natural spatial structure:
    - Strikes dimension: moneyness patterns
    - Maturities dimension: term structure patterns

    CNNs can learn these spatial patterns through convolutional filters.

    Args:
        input_shape (tuple): Flattened input shape (e.g., (200,) for 20x10 grid).
        output_dim (int): Number of model parameters to predict.
        num_strikes (int): Number of strike prices (default: 20).
        num_maturities (int): Number of maturities (default: 10).
        learning_rate (float): Learning rate (default: 0.001).

    Returns:
        keras.Model: Compiled CNN model.
    """
    inputs = layers.Input(shape=input_shape)

    # Reshape flattened input to 2D surface with single channel
    x = layers.Reshape((num_strikes, num_maturities, 1))(inputs)

    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)

    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)

    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    outputs = layers.Dense(output_dim, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_Calibrator')

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def build_resnet_block(x, filters, kernel_size=3, dropout_rate=0.2):
    """
    Build a residual block with skip connection.

    Args:
        x: Input tensor.
        filters (int): Number of filters.
        kernel_size (int): Kernel size for convolutions.
        dropout_rate (float): Dropout rate.

    Returns:
        Tensor: Output of residual block.
    """
    # Save input for skip connection
    shortcut = x

    # First layer
    x = layers.Dense(filters, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second layer
    x = layers.Dense(filters, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    # Skip connection (add shortcut to output)
    # If dimensions don't match, project shortcut
    if shortcut.shape[-1] != filters:
        shortcut = layers.Dense(filters)(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    return x


def build_resnet_model(input_shape, output_dim, num_blocks=3, filters_list=[256, 128, 64],
                      learning_rate=0.001):
    """
    Build ResNet model with residual blocks.

    Residual connections help train deeper networks by:
    - Preventing vanishing gradients
    - Allowing gradient flow through skip connections
    - Learning residual mappings instead of direct mappings

    Args:
        input_shape (tuple): Input shape.
        output_dim (int): Number of output parameters.
        num_blocks (int): Number of residual blocks (default: 3).
        filters_list (list): Number of filters per block (default: [256, 128, 64]).
        learning_rate (float): Learning rate (default: 0.001).

    Returns:
        keras.Model: Compiled ResNet model.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial dense layer
    x = layers.Dense(filters_list[0], activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Residual blocks
    for i, filters in enumerate(filters_list):
        x = build_resnet_block(x, filters, dropout_rate=0.2)

    # Output layer
    outputs = layers.Dense(output_dim, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='ResNet_Calibrator')

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


class CalibrationEnsemble:
    """
    Ensemble of multiple calibration models for improved robustness.

    Combines predictions from multiple models using:
    - Simple averaging
    - Weighted averaging (by validation performance)
    - Stacking with meta-learner
    """

    def __init__(self, models=None, weights=None, aggregation='average'):
        """
        Initialize ensemble.

        Args:
            models (list): List of trained Keras models.
            weights (list): Optional weights for weighted averaging.
            aggregation (str): Aggregation method - 'average', 'weighted', or 'stacking'.
        """
        self.models = models if models is not None else []
        self.weights = weights
        self.aggregation = aggregation
        self.meta_learner = None

    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        if self.weights is None:
            self.weights = []
        self.weights.append(weight)

    def predict(self, X):
        """
        Make predictions using ensemble.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Ensemble predictions.
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        # Get predictions from all models
        predictions = np.array([model.predict(X, verbose=0) for model in self.models])

        if self.aggregation == 'average':
            # Simple averaging
            return np.mean(predictions, axis=0)

        elif self.aggregation == 'weighted':
            # Weighted averaging
            if self.weights is None:
                raise ValueError("Weights not provided for weighted aggregation")
            weights = np.array(self.weights) / np.sum(self.weights)
            weights = weights.reshape(-1, 1, 1)  # Broadcast over samples and features
            return np.sum(predictions * weights, axis=0)

        elif self.aggregation == 'stacking':
            # Stacking with meta-learner
            if self.meta_learner is None:
                raise ValueError("Meta-learner not trained for stacking")
            # Flatten predictions as features for meta-learner
            stacked_features = predictions.reshape(len(self.models), -1).T
            return self.meta_learner.predict(stacked_features, verbose=0)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def train_meta_learner(self, X_val, y_val):
        """
        Train meta-learner for stacking ensemble.

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.
        """
        # Get base model predictions on validation set
        predictions = np.array([model.predict(X_val, verbose=0) for model in self.models])
        stacked_features = predictions.reshape(len(self.models), -1).T

        # Train simple linear meta-learner
        self.meta_learner = keras.Sequential([
            layers.Input(shape=(stacked_features.shape[1],)),
            layers.Dense(y_val.shape[1], activation='linear')
        ])
        self.meta_learner.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.meta_learner.fit(stacked_features, y_val, epochs=50, verbose=0)

    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble performance.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            dict: Evaluation metrics.
        """
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))

        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }


if __name__ == "__main__":
    # Example usage
    input_dim = 200  # 20 strikes × 10 maturities
    output_dim = 3   # sigma, nu, theta

    print("Building CNN model...")
    cnn_model = build_cnn_model((input_dim,), output_dim)
    cnn_model.summary()

    print("\nBuilding ResNet model...")
    resnet_model = build_resnet_model((input_dim,), output_dim)
    resnet_model.summary()

    print("\nCreating ensemble...")
    ensemble = CalibrationEnsemble(aggregation='average')
    ensemble.add_model(cnn_model)
    ensemble.add_model(resnet_model)
    print(f"Ensemble contains {len(ensemble.models)} models")
