import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.calibration_net.model import build_mlp_model
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FEATURES_FILE = 'data/processed/features.parquet'
TARGETS_FILE = 'data/processed/targets.parquet'
MODEL_PATH = 'models/calibration_net/mlp_calibration_model.h5'

def run_significance_testing():
    """
    Performs statistical significance tests on the model's predictions.
    This example focuses on comparing predicted vs. actual errors.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

    print("Loading features and targets...")
    features_df = pd.read_parquet(FEATURES_FILE)
    targets_df = pd.read_parquet(TARGETS_FILE)

    X = features_df.values
    y = targets_df.values

    # Use the test set for evaluation
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_X.fit(X) # Fit on full data for demonstration
    X_test_scaled = scaler_X.transform(X_test)

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'build_mlp_model': build_mlp_model})

    print("Making predictions on test data...")
    predictions = model.predict(X_test_scaled)

    # Calculate errors
    errors = predictions - y_test

    print("\n--- Statistical Significance Testing ---")

    # Example 1: Paired t-test on absolute errors (comparing to zero error)
    # This tests if the mean absolute error is significantly different from zero.
    # A more appropriate test might compare errors of two different models.
    abs_errors = np.abs(errors)
    for i, param_name in enumerate(targets_df.columns):
        t_stat, p_value = stats.ttest_1samp(abs_errors[:, i], 0)
        print(f"\nParameter: {param_name}")
        print(f"  Paired t-test (Abs Errors vs 0): t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("  Conclusion: Mean absolute error is significantly different from zero (p < 0.05).")
        else:
            print("  Conclusion: Mean absolute error is NOT significantly different from zero (p >= 0.05).")

    # Example 2: Distribution of errors (visual inspection or normality tests)
    # For a truly unbiased model, errors should ideally be centered around zero and normally distributed.
    print("\nError Distribution (Mean and Std Dev):")
    for i, param_name in enumerate(targets_df.columns):
        print(f"  {param_name}: Mean Error = {np.mean(errors[:, i]):.4f}, Std Dev Error = {np.std(errors[:, i]):.4f}")

    print("\n--- Visualizing Error Distributions ---")

    param_names = targets_df.columns.tolist()

    plt.figure(figsize=(15, 5 * len(param_names)))
    for i, param_name in enumerate(param_names):
        # Histogram of Errors
        plt.subplot(len(param_names), 2, 2*i + 1)
        sns.histplot(errors[:, i], kde=True, bins=50)
        plt.title(f'Histogram of Errors for {param_name}')
        plt.xlabel(f'Error in {param_name}')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Q-Q Plot of Errors
        plt.subplot(len(param_names), 2, 2*i + 2)
        stats.probplot(errors[:, i], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of Errors for {param_name}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n--- Interpretation of Error Distribution Plots ---")
    print("The histograms of errors show the frequency distribution of prediction errors for each parameter. Ideally, these should be centered around zero and resemble a normal (bell-shaped) distribution, indicating unbiased predictions.")
    print("The Q-Q (Quantile-Quantile) plots compare the distribution of errors against a theoretical normal distribution. If the errors are normally distributed, the points should fall approximately along the 45-degree red line. Deviations from this line suggest non-normality, which could imply issues like heteroscedasticity or a biased model.")
    print("Both plots help in visually assessing the quality and statistical properties of the model's errors, which is crucial for understanding its reliability.")

    # Further tests could include:
    # - Comparing errors of our ML model against a benchmark model (e.g., traditional optimization)
    # - Bootstrapping to get confidence intervals for performance metrics
    # - Permutation tests for comparing model performance

if __name__ == '__main__':
    run_significance_testing()
