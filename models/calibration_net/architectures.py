"""
Advanced Neural Network Architectures for Lévy Model Calibration

This module implements state-of-the-art deep learning architectures for the
inverse problem of parameter calibration from option price surfaces.

Architectures:
1. CNN (Convolutional Neural Network): Treats option surfaces as 2D images
2. Transformer: Self-attention mechanism for importance weighting
3. ResNet: Deep residual networks with skip connections
4. Ensemble: Combines multiple models for robustness

Mathematical Problem:
Given option price surface P(K,T), find parameters θ such that:
    P_market(K,T) ≈ P_model(K,T; θ)

This is an ill-posed inverse problem solved via supervised learning.

Author: Mohin Hasin (mohinhasin999@gmail.com)
Project: Fractional PDEs and Lévy Processes: An ML Approach
Repository: https://github.com/mohin-io/Fractional-PDEs-under-Levy-Models-Machine-Learning-for-Calibrating-Advanced
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional, Dict, Tuple


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


def build_transformer_model(input_shape, output_dim, d_model=256, num_heads=8,
                           num_blocks=4, ff_dim=1024, dropout=0.1, learning_rate=0.001):
    """
    Build Transformer model with multi-head self-attention.

    The Transformer architecture allows the model to focus on the most
    important strikes and maturities for parameter estimation through
    the attention mechanism.

    Architecture:
        Input (n_features,)
        → Embedding(d_model)
        → Positional Encoding
        → [Transformer Block (Multi-Head Attention + FFN)] × num_blocks
        → Global Average Pooling
        → Dense(256) → Dense(128) → Dense(output_dim)

    Args:
        input_shape (tuple): Input shape (flattened surface)
        output_dim (int): Number of parameters to predict
        d_model (int): Embedding dimension (default: 256)
        num_heads (int): Number of attention heads (default: 8)
        num_blocks (int): Number of transformer blocks (default: 4)
        ff_dim (int): Feed-forward dimension (default: 1024)
        dropout (float): Dropout rate (default: 0.1)
        learning_rate (float): Learning rate (default: 0.001)

    Returns:
        keras.Model: Compiled Transformer model
    """
    inputs = layers.Input(shape=input_shape)

    # Embedding to d_model dimension
    x = layers.Dense(d_model)(inputs)
    x = layers.Reshape((-1, d_model))(x)  # (batch, seq_len, d_model)

    # Add positional encoding
    seq_len = input_shape[0]
    pos_encoding = _get_positional_encoding(seq_len, d_model)
    x = x + pos_encoding

    # Transformer blocks
    for i in range(num_blocks):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f'mha_{i}'
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ffn_output = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ], name=f'ffn_{i}')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Transformer_Calibrator')

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def _get_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """
    Generate sinusoidal positional encoding for Transformer.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length
        d_model: Embedding dimension

    Returns:
        Positional encoding tensor of shape (1, seq_len, d_model)
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angle_rads = positions * angle_rates

    # Apply sin to even indices, cos to odd indices
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)


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
    print("=" * 70)
    print("Advanced Calibration Architectures - Examples")
    print("=" * 70)

    input_dim = 200  # 20 strikes × 10 maturities
    output_dim = 3   # sigma, nu, theta

    print("\n1. CNN Model")
    print("-" * 70)
    cnn_model = build_cnn_model((input_dim,), output_dim, num_strikes=20, num_maturities=10)
    print(f"Built: {cnn_model.name}")
    print(f"Parameters: {cnn_model.count_params():,}")

    print("\n2. Transformer Model")
    print("-" * 70)
    transformer_model = build_transformer_model((input_dim,), output_dim, d_model=256, num_heads=8)
    print(f"Built: {transformer_model.name}")
    print(f"Parameters: {transformer_model.count_params():,}")

    print("\n3. ResNet Model")
    print("-" * 70)
    resnet_model = build_resnet_model((input_dim,), output_dim, filters_list=[256, 128, 64])
    print(f"Built: {resnet_model.name}")
    print(f"Parameters: {resnet_model.count_params():,}")

    print("\n4. Ensemble")
    print("-" * 70)
    ensemble = CalibrationEnsemble(aggregation='average')
    ensemble.add_model(cnn_model)
    ensemble.add_model(transformer_model)
    ensemble.add_model(resnet_model)
    print(f"Ensemble contains {len(ensemble.models)} models")
    print(f"Aggregation method: {ensemble.aggregation}")

    # Test prediction
    print("\n5. Test Predictions")
    print("-" * 70)
    X_test = np.random.randn(5, input_dim)
    ensemble_pred = ensemble.predict(X_test)
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {ensemble_pred.shape}")
    print(f"Sample predictions:\n{ensemble_pred}")

    print("\n" + "=" * 70)
