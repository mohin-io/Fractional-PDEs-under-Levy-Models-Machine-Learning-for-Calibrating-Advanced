# API Reference

This section provides a detailed reference for the API endpoints and their functionalities.

## `/calibrate` Endpoint

**Method:** `POST`

**Summary:** Calibrate Levy model parameters from an option price surface.

**Description:** Receives a flattened option price surface and returns the calibrated Levy model parameters using the trained ML model.

**Request Body:**

*   **`option_prices`** (list of float, required):
    A list of option prices representing a flattened option price surface. The length of this list must match the expected input dimension of the trained model (e.g., `num_strikes * num_maturities`).

**Responses:**

*   **`200 OK`**:
    Successful calibration. Returns a JSON object with the predicted Levy model parameters.
    
    **Example:**
    ```json
    {
      "sigma": 0.25,
      "nu": 0.45,
      "theta": -0.12
    }
    ```

*   **`400 Bad Request`**:
    Input option_prices dimension mismatch or invalid data.
    
    **Example:**
    ```json
    {
      "detail": "Input option_prices dimension mismatch. Expected 200 features, got 150."
    }
    ```

*   **`503 Service Unavailable`**:
    Model or scaler not loaded. The API is not ready to process requests.
    
    **Example:**
    ```json
    {
      "detail": "Model or scaler not loaded. API is not ready."
    }
    ```

## `/health` Endpoint

**Method:** `GET`

**Summary:** Health check endpoint.

**Description:** Checks the health of the API and ensures that the necessary resources (model, scaler) are loaded.

**Responses:**

*   **`200 OK`**:
    API is operational and resources are loaded.
    
    **Example:**
    ```json
    {
      "status": "ok",
      "model_loaded": true,
      "scaler_loaded": true,
      "tensorflow_version": "2.20.0",
      "keras_version": "3.11.3"
    }
    ```
