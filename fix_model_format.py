"""
Re-save models in Keras 3+ compatible format.
"""
import sys
sys.path.insert(0, '.')

import tensorflow as tf
import joblib
from pathlib import Path

print("Fixing model formats for deployment...")

# Load and re-save MLP model
mlp_path = Path("models/calibration_net/mlp_calibration_model.h5")
if mlp_path.exists():
    print(f"\n1. Loading MLP model from: {mlp_path}")
    try:
        # Load the old model without metrics
        model = tf.keras.models.load_model(str(mlp_path), compile=False)
        print(f"   Model loaded successfully (without compilation)")

        # Save in new format (.keras)
        new_path = mlp_path.with_suffix('.keras')
        model.save(str(new_path))
        print(f"   Saved to new format: {new_path}")

        # Also save as .h5 but properly
        model.save(str(mlp_path))
        print(f"   Re-saved as: {mlp_path}")

    except Exception as e:
        print(f"   Error: {e}")

print("\nModel format fix complete!")
print("\nNow you can deploy the API successfully.")
