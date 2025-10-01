import pandas as pd
import numpy as np
import os
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.calibration_net.model import build_mlp_model, get_callbacks
from models.calibration_net.architectures import build_cnn_model, build_resnet_model
import tensorflow as tf
import joblib

# --- Configuration ---
FEATURES_FILE = "data/processed/features.parquet"
TARGETS_FILE = "data/processed/targets.parquet"
MODEL_SAVE_PATH = "models/calibration_net/mlp_calibration_model.h5"
SCALER_SAVE_PATH = "models/calibration_net/scaler_X.pkl"
HISTORY_SAVE_PATH = "models/calibration_net/training_history.json"


def create_tf_dataset(X, y, batch_size=32, shuffle=True, prefetch=True):
    """
    Create optimized TensorFlow Dataset for efficient training.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Targets.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle.
        prefetch (bool): Whether to prefetch batches.

    Returns:
        tf.data.Dataset: Optimized dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size)

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def train_model(architecture='mlp', epochs=50, batch_size=32, learning_rate=0.001,
                use_mixed_precision=False, save_history=True):
    """
    Loads processed features and targets, trains the model, and saves it.

    Args:
        architecture (str): Model architecture - 'mlp', 'cnn', or 'resnet'.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        use_mixed_precision (bool): Enable mixed precision training for GPU speedup.
        save_history (bool): Save training history to JSON.
    """
    # Enable mixed precision if requested
    if use_mixed_precision:
        print("Enabling mixed precision training...")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    print("Loading features and targets...")
    features_df = pd.read_parquet(FEATURES_FILE)
    targets_df = pd.read_parquet(TARGETS_FILE)

    X = features_df.values
    y = targets_df.values

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Standardize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Determine input and output dimensions
    input_shape = (X_train.shape[1],)
    output_dim = y_train.shape[1]

    # Build model based on architecture choice
    print(f"Building {architecture.upper()} model...")
    if architecture == 'mlp':
        model = build_mlp_model(input_shape, output_dim, learning_rate=learning_rate)
    elif architecture == 'cnn':
        model = build_cnn_model(input_shape, output_dim, learning_rate=learning_rate)
    elif architecture == 'resnet':
        model = build_resnet_model(input_shape, output_dim, learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.summary()

    # Get callbacks
    model_path = MODEL_SAVE_PATH.replace('.h5', f'_{architecture}.h5')
    callbacks = get_callbacks(model_path=model_path, patience=15)

    # Create TensorFlow datasets for efficient training
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

    print(f"\nTraining {architecture.upper()} model for {epochs} epochs...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate the model on test set
    print("\nEvaluating on test set...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test RMSE: {np.sqrt(loss):.6f}")

    # Per-parameter MAE
    y_pred = model.predict(X_test, verbose=0)
    param_names = targets_df.columns.tolist()
    print("\nPer-parameter MAE:")
    for i, param_name in enumerate(param_names):
        param_mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
        print(f"  {param_name}: {param_mae:.6f}")

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save scaler
    scaler_path = SCALER_SAVE_PATH.replace('.pkl', f'_{architecture}.pkl')
    joblib.dump(scaler_X, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Save training history
    if save_history:
        history_path = HISTORY_SAVE_PATH.replace('.json', f'_{architecture}.json')
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']],
            'test_loss': float(loss),
            'test_mae': float(mae),
            'architecture': architecture,
            'epochs_trained': len(history.history['loss']),
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"Training history saved to {history_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train calibration model")
    parser.add_argument('--architecture', type=str, default='mlp',
                       choices=['mlp', 'cnn', 'resnet'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training')

    args = parser.parse_args()

    train_model(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_mixed_precision=args.mixed_precision
    )
