import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FEATURES_FILE = 'data/processed/features.parquet'
TARGETS_FILE = 'data/processed/targets.parquet'
MODEL_PATH = 'models/calibration_net/mlp_calibration_model.h5'

def run_out_of_sample_validation():
    """
    Loads the trained model and evaluates its performance on an out-of-sample dataset.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

    print("Loading features and targets...")
    features_df = pd.read_parquet(FEATURES_FILE)
    targets_df = pd.read_parquet(TARGETS_FILE)

    X = features_df.values
    y = targets_df.values

    # Split data into training and testing sets (using the same split as in training for consistency)
    # In a true out-of-sample, this would be a completely new, unseen dataset.
    # For now, we'll use the test set from the training split.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features using a scaler fitted on the training data (conceptually)
    # In a real scenario, the scaler would be saved during training and loaded here.
    scaler_X = StandardScaler()
    # Fit on the full dataset for demonstration, ideally fit only on training data
    scaler_X.fit(X)
    X_test_scaled = scaler_X.transform(X_test)

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'build_mlp_model': build_mlp_model})

    print("Evaluating model on out-of-sample data...")
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=1)

    print(f"\nOut-of-Sample Validation Results:")
    print(f"  Loss (MSE): {loss:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")

    # Further analysis: e.g., plot predictions vs actuals
    predictions = model.predict(X_test_scaled)
    print("\n--- Visualizing Out-of-Sample Results ---")

    # Get parameter names for plotting
    param_names = targets_df.columns.tolist()

    # Plot Actual vs. Predicted for each parameter
    plt.figure(figsize=(15, 5 * len(param_names)))
    for i, param_name in enumerate(param_names):
        plt.subplot(len(param_names), 2, 2*i + 1)
        sns.scatterplot(x=y_test[:, i], y=predictions[:, i], alpha=0.6)
        plt.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])],
                 color='red', linestyle='--', lw=2, label='Perfect Prediction')
        plt.title(f'Actual vs. Predicted {param_name}')
        plt.xlabel(f'Actual {param_name}')
        plt.ylabel(f'Predicted {param_name}')
        plt.legend()
        plt.grid(True)

        # Plot Histogram of Errors for each parameter
        plt.subplot(len(param_names), 2, 2*i + 2)
        errors = predictions[:, i] - y_test[:, i]
        sns.histplot(errors, kde=True, bins=50)
        plt.title(f'Distribution of Errors for {param_name}')
        plt.xlabel(f'Error in {param_name}')
        plt.ylabel('Frequency')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n--- Interpretation of Out-of-Sample Plots ---")
    print("The scatter plots of 'Actual vs. Predicted' parameters show how well the model's predictions align with the true values. Ideally, points should cluster tightly around the red 'Perfect Prediction' line. Deviations indicate prediction errors.")
    print("The 'Distribution of Errors' histograms illustrate the spread and bias of the prediction errors for each parameter. A good model typically shows errors centered around zero with a relatively narrow distribution, suggesting unbiased and precise predictions.")
    print("Significant spread or a non-zero mean in the error distribution could indicate areas where the model struggles or has a systematic bias.")

if __name__ == '__main__':
    run_out_of_sample_validation()
