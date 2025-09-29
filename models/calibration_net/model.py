import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_mlp_model(input_shape, output_dim):
    """
    Builds a simple Multi-Layer Perceptron (MLP) model for calibration.

    Args:
        input_shape (tuple): Shape of the input features (e.g., (num_strikes * num_maturities,)).
        output_dim (int): Dimension of the output (number of Levy model parameters).

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='linear') # Linear activation for regression
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':
    # Example Usage
    input_dim = 20 * 10 # Example: 20 strikes * 10 maturities
    output_dim = 3      # Example: sigma, nu, theta for Variance Gamma

    mlp_model = build_mlp_model(input_shape=(input_dim,), output_dim=output_dim)
    mlp_model.summary()
