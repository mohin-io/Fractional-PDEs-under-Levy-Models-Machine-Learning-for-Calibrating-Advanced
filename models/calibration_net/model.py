import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def build_mlp_model(input_shape, output_dim, use_batch_norm=True, dropout_rate=0.3,
                    hidden_units=[256, 128, 64], learning_rate=0.001):
    """
    Builds an enhanced Multi-Layer Perceptron (MLP) model for calibration.

    Args:
        input_shape (tuple): Shape of the input features (e.g., (num_strikes * num_maturities,)).
        output_dim (int): Dimension of the output (number of Levy model parameters).
        use_batch_norm (bool): Whether to use batch normalization (default: True).
        dropout_rate (float): Dropout rate for regularization (default: 0.3).
        hidden_units (list): List of hidden layer sizes (default: [256, 128, 64]).
        learning_rate (float): Learning rate for Adam optimizer (default: 0.001).

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Hidden layers
    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(units, activation="relu",
                              kernel_regularizer=keras.regularizers.l2(1e-4)))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(output_dim, activation="linear"))

    # Compile with learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


def get_callbacks(model_path='models/calibration_net/best_model.h5', patience=10):
    """
    Get training callbacks for model optimization.

    Args:
        model_path (str): Path to save best model checkpoint.
        patience (int): Patience for early stopping (default: 10).

    Returns:
        list: List of Keras callbacks.
    """
    callbacks = [
        # Early stopping when validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


if __name__ == "__main__":
    # Example Usage
    input_dim = 20 * 10  # Example: 20 strikes * 10 maturities
    output_dim = 3  # Example: sigma, nu, theta for Variance Gamma

    mlp_model = build_mlp_model(input_shape=(input_dim,), output_dim=output_dim)
    mlp_model.summary()
