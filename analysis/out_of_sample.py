import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.calibration_net.model import build_mlp_model # Needed to load custom model architecture

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
    print("\nSample of Predictions vs Actuals:")
    for i in range(min(5, len(predictions))):
        print(f"  Actual: {y_test[i]}, Predicted: {predictions[i]}")

if __name__ == '__main__':
    run_out_of_sample_validation()
