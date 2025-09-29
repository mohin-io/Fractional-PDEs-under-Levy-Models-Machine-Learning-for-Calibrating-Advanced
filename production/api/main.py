from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import joblib # To save/load the scaler

# Import the prediction function and model architecture
from models.calibration_net.predict import predict_parameters
from models.calibration_net.model import build_mlp_model

# --- Configuration ---
MODEL_PATH = 'models/calibration_net/mlp_calibration_model.h5'
SCALER_PATH = 'models/calibration_net/scaler_X.pkl' # Path to save/load the scaler

app = FastAPI(
    title="Levy Model Calibration API",
    description="API for calibrating Levy model parameters from option price surfaces using a trained ML model.",
    version="0.1.0",
)

# Load model and scaler globally to avoid re-loading on each request
model = None
scaler_X = None
target_parameter_names = None # To be loaded from training data or defined

@app.on_event("startup")
async def load_resources():
    """
    Load the trained model and scaler when the FastAPI application starts up.
    """
    global model, scaler_X, target_parameter_names

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler not found at {SCALER_PATH}. Please ensure it's saved during training.")

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'build_mlp_model': build_mlp_model})
    print(f"Loading scaler from {SCALER_PATH}...")
    scaler_X = joblib.load(SCALER_PATH)

    # In a real scenario, target_parameter_names would also be saved/loaded
    # For now, we'll infer from the training data or define a default.
    # This assumes the order of parameters is consistent.
    # For demonstration, let's assume VG parameters if the model was trained on them.
    # A more robust solution would save the target_cols from build_features.py
    # For now, we'll use a dummy list.
    target_parameter_names = ['sigma', 'nu', 'theta'] # Example for VG model

    print("API resources loaded successfully.")

class OptionSurfaceInput(BaseModel):
    """
    Input schema for the option surface data.
    """
    option_prices: list[float]
    # Add other relevant market data if needed, e.g., strikes, maturities, S0, r
    # For simplicity, we assume the input `option_prices` is already flattened
    # and corresponds to the fixed grid used during model training.

@app.post("/calibrate", summary="Calibrate Levy model parameters from an option price surface")
async def calibrate_parameters(input_data: OptionSurfaceInput):
    """
    Receives a flattened option price surface and returns the calibrated
    Levy model parameters.
    """
    if model is None or scaler_X is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded. API is not ready.")

    option_surface_array = np.array(input_data.option_prices).reshape(1, -1) # Reshape for single prediction

    # Validate input shape against scaler's expected features
    if option_surface_array.shape[1] != scaler_X.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input option_prices dimension mismatch. Expected {scaler_X.n_features_in_} features, got {option_surface_array.shape[1]}."
        )

    # Preprocess the input data
    option_surface_scaled = scaler_X.transform(option_surface_array)

    # Make prediction
    predicted_params = model.predict(option_surface_scaled)[0] # Get the first (and only) prediction

    # Return as a dictionary with parameter names
    if target_parameter_names and len(target_parameter_names) == len(predicted_params):
        return dict(zip(target_parameter_names, predicted_params))
    else:
        # Fallback if target_parameter_names is not correctly set
        return {"calibrated_parameters": predicted_params.tolist()}

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Checks the health of the API and ensures resources are loaded.
    """
    status = {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler_X is not None,
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
    }
    return status

if __name__ == '__main__':
    import uvicorn
    # To run this:
    # 1. Ensure you have a trained model at MODEL_PATH and a scaler at SCALER_PATH.
    #    You'll need to modify train.py to save the scaler.
    # 2. Run: uvicorn main:app --reload --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
