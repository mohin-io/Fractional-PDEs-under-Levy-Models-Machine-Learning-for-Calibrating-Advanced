import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.calibration_net.model import build_mlp_model
import tensorflow as tf
import joblib

# --- Configuration ---
FEATURES_FILE = 'data/processed/features.parquet'
TARGETS_FILE = 'data/processed/targets.parquet'
MODEL_SAVE_PATH = 'models/calibration_net/mlp_calibration_model.h5'
SCALER_SAVE_PATH = 'models/calibration_net/scaler_X.pkl'

def train_model():
    """
    Loads processed features and targets, trains the MLP model, and saves it.
    """
    print("Loading features and targets...")
    features_df = pd.read_parquet(FEATURES_FILE)
    targets_df = pd.read_parquet(TARGETS_FILE)

    X = features_df.values
    y = targets_df.values

    # Standardize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Determine input and output dimensions
    input_shape = (X_train.shape[1],)
    output_dim = y_train.shape[1]

    # Build and compile the model
    model = build_mlp_model(input_shape, output_dim)
    model.summary()

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50, # Can be increased for better performance
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    joblib.dump(scaler_X, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
