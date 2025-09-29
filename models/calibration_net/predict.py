import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from models.calibration_net.model import build_mlp_model # Needed to load custom model architecture

# --- Configuration ---
MODEL_PATH = 'models/calibration_net/mlp_calibration_model.h5'
# Assuming we need the scaler used during training to preprocess new input data
# In a real scenario, this scaler would also be saved and loaded.
# For now, we'll re-fit a scaler on dummy data or assume a pre-fitted one.
# For demonstration, we'll assume a scaler is available or re-fit on some data.
# A more robust solution would save/load the scaler alongside the model.

def predict_parameters(option_surface_data, scaler_X=None, target_cols=None):
    """
    Loads a trained model and predicts Levy model parameters from option surface data.

    Args:
        option_surface_data (np.ndarray): A 2D array where each row is a flattened
                                          option price surface for which to predict parameters.
        scaler_X (StandardScaler, optional): Pre-fitted StandardScaler for features.
                                             If None, a new one will be fitted (not recommended for production).
        target_cols (list, optional): List of target parameter names (e.g., ['sigma', 'nu', 'theta']).
                                      Used for returning predictions in a DataFrame.

    Returns:
        pd.DataFrame: Predicted Levy model parameters.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

    print(f"Loading model from {MODEL_PATH}...")
    # When loading a model with custom layers or functions, they need to be provided
    # In this case, build_mlp_model is used to define the architecture, so it's needed for loading.
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'build_mlp_model': build_mlp_model})

    # Preprocess input data using the same scaler used during training
    if scaler_X is None:
        print("Warning: No scaler provided. Fitting a new scaler on dummy data. This is not recommended for production.")
        # In a real application, you would load the pre-fitted scaler here.
        # For demonstration, we'll create a dummy scaler.
        dummy_data = np.random.rand(100, option_surface_data.shape[1])
        scaler_X = StandardScaler()
        scaler_X.fit(dummy_data)
    
    option_surface_scaled = scaler_X.transform(option_surface_data)

    print("Making predictions...")
    predictions = model.predict(option_surface_scaled)

    if target_cols:
        return pd.DataFrame(predictions, columns=target_cols)
    return pd.DataFrame(predictions)

if __name__ == '__main__':
    # Example Usage:
    # This example assumes you have a trained model and a scaler.
    # For a real test, you would need to:
    # 1. Run train.py to generate a model.
    # 2. Save the scaler used in train.py and load it here.
    
    # Dummy option surface data (e.g., from a single option surface)
    # This should match the input_shape used during training
    dummy_option_surface = np.random.rand(1, 200) # 1 sample, 20 strikes * 10 maturities

    # Dummy scaler (in a real scenario, load the saved scaler)
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(np.random.rand(100, 200)) # Fit on some data with same feature dimensions

    # Dummy target columns (should match the order used during training)
    dummy_target_cols = ['sigma', 'nu', 'theta']

    try:
        predicted_params = predict_parameters(
            dummy_option_surface,
            scaler_X=dummy_scaler,
            target_cols=dummy_target_cols
        )
        print("\nPredicted Levy Model Parameters:")
        print(predicted_params)
    except FileNotFoundError as e:
        print(e)
        print("Please run 'train.py' first to generate the model.")
