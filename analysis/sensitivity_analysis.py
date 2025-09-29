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
    # Inverse transform the scaled values for interpretation
    original_base_input = scaler_X.inverse_transform(base_input.reshape(1, -1))[0]
    original_variation_range = scaler_X.inverse_transform(np.array([base_input[:param_idx].tolist() + [val] + base_input[param_idx+1:].tolist() for val in variation_range]))[:, param_idx]

    print(f"Varying '{param_to_vary}' from {original_variation_range.min():.4f} to {original_variation_range.max():.4f} (Original Scale)")

    # Plotting the sensitivity
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid") # Apply seaborn style
    for i, target_param in enumerate(target_cols):
        sns.lineplot(x=original_variation_range, y=predicted_params_over_range[:, i], label=f'Predicted {target_param}', marker='o')
    
    plt.title(f'Sensitivity of Predicted Parameters to {param_to_vary}', fontsize=16)
    plt.xlabel(f'Value of {param_to_vary} (Original Scale)', fontsize=12)
    plt.ylabel('Predicted Parameter Value', fontsize=12)
    plt.legend(title='Levy Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\n--- Interpretation of Sensitivity Analysis Plot ---")
    print(f"This plot illustrates how the predicted Levy model parameters change as a specific input feature ('{param_to_vary}') is varied, while other features are held constant at their average values.")
    print("Each line represents a different Levy model parameter. The slope and direction of these lines indicate the sensitivity:")
    print("  - A steep slope suggests high sensitivity: a small change in the input feature leads to a large change in the predicted parameter.")
    print("  - A flat line suggests low sensitivity: the predicted parameter is relatively robust to changes in this input feature.")
    print("  - The direction (positive or negative slope) shows whether the predicted parameter increases or decreases with the input feature.")
    print("This analysis is crucial for understanding the model's behavior, identifying influential input features, and assessing the stability of the calibration process. For instance, if a parameter is highly sensitive to a noisy input, it might lead to unstable calibrations in real-world scenarios.")
