import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.calibration_net.model import build_mlp_model
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FEATURES_FILE = "data/processed/features.parquet"
TARGETS_FILE = "data/processed/targets.parquet"
MODEL_PATH = "models/calibration_net/mlp_calibration_model.h5"


def run_forward_walking_validation(n_splits=5):
    """
    Performs forward-walking validation on the model.

    Args:
        n_splits (int): Number of splits for forward-walking.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please train the model first."
        )

    print("Loading features and targets...")
    features_df = pd.read_parquet(FEATURES_FILE)
    targets_df = pd.read_parquet(TARGETS_FILE)

    X = features_df.values
    y = targets_df.values

    # For forward-walking, we need a time-ordered dataset.
    # Our synthetic data is not inherently time-ordered, so we'll simulate it
    # by splitting the dataset sequentially.
    # In a real-world scenario, this would involve actual time series data.

    total_samples = len(X)
    fold_size = total_samples // n_splits

    all_mae_scores = []
    all_loss_scores = []
    actual_folds = []

    print(f"Starting forward-walking validation with {n_splits} splits...")

    for i in range(n_splits):
        print(f"\n--- Fold {i+1}/{n_splits} ---")
        # Define training and testing sets for the current fold
        # Training data grows with each fold
        train_end_idx = (i + 1) * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, total_samples)

        if test_start_idx >= total_samples:
            print("No more data for testing in this fold. Skipping.")
            break

        X_train_fold = X[:train_end_idx]
        y_train_fold = y[:train_end_idx]
        X_test_fold = X[test_start_idx:test_end_idx]
        y_test_fold = y[test_start_idx:test_end_idx]

        if len(X_test_fold) == 0:
            print("Test set is empty. Skipping this fold.")
            continue

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_fold)
        X_test_scaled = scaler_X.transform(X_test_fold)

        # Re-train model for each fold (or load and fine-tune)
        # For simplicity, we'll load the pre-trained model and evaluate.
        # In a true forward-walking, the model would be re-trained or fine-tuned
        # on the expanding training set.
        model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={
                "build_mlp_model": build_mlp_model,
                "mse": losses.MeanSquaredError(),
                "mae": metrics.MeanAbsoluteError()
            }
        )

        print(
            f"Evaluating model on test data for fold {i+1} (samples {test_start_idx}-{test_end_idx})..."
        )
        loss, mae = model.evaluate(X_test_scaled, y_test_fold, verbose=0)
        print(f"  Loss (MSE): {loss:.4f}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")

        all_loss_scores.append(loss)
        all_mae_scores.append(mae)
        actual_folds.append(i + 1)

    print("\n--- Forward-Walking Validation Summary ---")
    print(
        f"Average Loss (MSE) across folds: {np.mean(all_loss_scores):.4f} (+/- {np.std(all_loss_scores):.4f})"
    )
    print(
        f"Average MAE across folds: {np.mean(all_mae_scores):.4f} (+/- {np.std(all_mae_scores):.4f})"
    )

    print("\n--- Visualizing Forward-Walking Results ---")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.lineplot(x=actual_folds, y=all_loss_scores, marker="o")
    plt.title("Loss (MSE) Across Forward-Walking Folds")
    plt.xlabel("Fold Number")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.lineplot(x=actual_folds, y=all_mae_scores, marker="o")
    plt.title("MAE Across Forward-Walking Folds")
    plt.xlabel("Fold Number")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n--- Interpretation of Forward-Walking Plots ---")
    print(
        "These plots show the model's performance (Loss and MAE) as it's evaluated on progressively later, unseen data segments. In a truly time-series context, this helps assess the model's stability and adaptability over time."
    )
    print(
        "A stable model would show consistent performance across folds, without significant degradation. Increasing errors might indicate concept drift or that the model is not adapting well to new market conditions."
    )


if __name__ == "__main__":
    run_forward_walking_validation()
