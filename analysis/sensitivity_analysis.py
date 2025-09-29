import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from models.calibration_net.model import build_mlp_model
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FEATURES_FILE = 'data/processed/features.parquet'
TARGETS_FILE = 'data/processed/targets.parquet'
MODEL_PATH = 'models/calibration_net/mlp_calibration_model.h5'

def run_sensitivity_analysis(param_to_vary='sigma', num_steps=10, range_factor=0.2):
    """
    Performs sensitivity analysis on the model's predictions by varying one input parameter
    (e.g., a specific option price) and observing the change in predicted Levy parameters.

    Args:
        param_to_vary (str): The name of the feature column to vary (e.g., 'price_0_0').
                             This should be one of the feature columns in features.parquet.
        num_steps (int): Number of steps to vary the parameter.
        range_factor (float): Factor to determine the range of variation around the mean.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

    print("Loading features and targets...")
    features_df = pd.read_parquet(FEATURES_FILE)
    targets_df = pd.read_parquet(TARGETS_FILE)

    X = features_df.values
    y = targets_df.values
    feature_cols = features_df.columns.tolist()
    target_cols = targets_df.columns.tolist()

    # Find the index of the parameter to vary
    try:
        param_idx = feature_cols.index(param_to_vary)
    except ValueError:
        print(f"Warning: Feature '{param_to_vary}' not found. Using 'price_0_0' as default.")
        param_to_vary = 'price_0_0'
        param_idx = feature_cols.index(param_to_vary)


    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'build_mlp_model': build_mlp_model})

    # Select a base input (e.g., the mean of the scaled features)
    base_input = np.mean(X_scaled, axis=0)

    # Determine the range for variation
    mean_val = base_input[param_idx]
    std_val = np.std(X_scaled[:, param_idx])
    variation_range = np.linspace(mean_val - range_factor * std_val,
                                  mean_val + range_factor * std_val,
                                  num_steps)

    predicted_params_over_range = []
    for val in variation_range:
        input_to_predict = np.copy(base_input)
        input_to_predict[param_idx] = val
        # Reshape for model prediction (batch size of 1)
        prediction = model.predict(input_to_predict.reshape(1, -1), verbose=0)[0]
        predicted_params_over_range.append(prediction)

    predicted_params_over_range = np.array(predicted_params_over_range)

    print(f"\n--- Sensitivity Analysis for varying '{param_to_vary}' ---")
    print(f"Varying '{param_to_vary}' from {scaler_X.inverse_transform(base_input.reshape(1,-1))[0, param_idx] * (1 - range_factor):.2f} to {scaler_X.inverse_transform(base_input.reshape(1,-1))[0, param_idx] * (1 + range_factor):.2f}")

    # Plotting the sensitivity
    plt.figure(figsize=(12, 7))
    for i, target_param in enumerate(target_cols):
        plt.plot(variation_range, predicted_params_over_range[:, i], label=f'Predicted {target_param}')
    
    plt.title(f'Sensitivity of Predicted Parameters to {param_to_vary}')
    plt.xlabel(f'Scaled Value of {param_to_vary}')
    plt.ylabel('Predicted Parameter Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Example: Vary the first option price feature
    run_sensitivity_analysis(param_to_vary='price_0_0')
