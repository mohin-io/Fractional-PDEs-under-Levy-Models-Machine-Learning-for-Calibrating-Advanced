# Project Guideline: Fractional PDEs and Lévy Processes: An ML Approach

## 1. Executive Summary

This report details the development of a novel machine learning-based engine for calibrating advanced asset pricing models, specifically Lévy models (Variance Gamma and CGMY), to market option data. Traditional calibration methods are computationally intensive and struggle with the complexity of these models, creating a significant bottleneck in quantitative finance. Our solution leverages deep neural networks to provide near-instantaneous parameter calibration and incorporates robust validation techniques, including out-of-sample, forward-walking, and statistical significance testing, alongside sensitivity analysis. The project delivers a modular, scalable, and production-ready framework designed to enhance the speed, accuracy, and interpretability of financial modeling.

## 2. Introduction

The financial markets are characterized by phenomena such as "fat tails" and jumps in asset prices, which are not adequately captured by classical models like Black-Scholes. Lévy processes offer a more realistic framework by incorporating these features. However, calibrating these models—determining their parameters from observed market option prices—is a challenging inverse problem. This project addresses this challenge by proposing and implementing a machine learning approach that transforms the inverse problem into a regression task, significantly accelerating the calibration process.

## 3. Methodology

### 3.1. Project Architecture

The project follows a modular architecture, organized into distinct phases:

*   **Data Generation:** Synthetic option price surfaces are generated using a Fourier-based pricing engine for various Lévy model parameters.
*   **Feature Engineering:** Raw option price surfaces are processed into features and corresponding Lévy parameters as targets.

*   **Model Development:** A deep Multi-Layer Perceptron (MLP) is trained to learn the mapping from option price surfaces to model parameters. Bayesian methods are also explored for uncertainty quantification.
*   **Backtesting & Validation:** Comprehensive validation includes out-of-sample testing, forward-walking analysis, statistical significance tests, and sensitivity analysis.
*   **Productionization:** A FastAPI-based API exposes the calibration engine for real-time predictions, supported by a CI/CD pipeline.

### 3.2. Data Generation

A synthetic dataset of 100,000 (option price surface, Lévy model parameters) pairs was generated.
*   **Lévy Models:** Variance Gamma and CGMY models were used.
*   **Parameter Sampling:** Sobol sequences ensured uniform coverage of realistic parameter ranges.
*   **Pricing Engine:** A custom Fourier-based Carr-Madan pricer was implemented to efficiently compute option prices for given parameters.

### 3.3. Calibration Network

A deep MLP was designed and trained using TensorFlow/Keras.
*   **Input:** Flattened option price surface (features).
*   **Output:** Lévy model parameters (targets).
*   **Training:** The model was trained using Mean Squared Error (MSE) loss and Adam optimizer.
*   **Scalers:** `StandardScaler` was used for feature preprocessing, and saved alongside the model for consistent inference.

### 3.4. Validation Strategy

Robust validation is critical for financial models:
*   **Out-of-Sample Validation:** Evaluates model performance on unseen data, assessing generalization capabilities.
*   **Forward-Walking Validation:** Simulates a time-series evaluation, assessing model stability and adaptability over time.
*   **Statistical Significance Testing:** Uses t-tests to analyze prediction errors, ensuring they are unbiased and statistically insignificant from zero.
*   **Sensitivity Analysis:** Examines how predicted parameters change with variations in input features, identifying influential factors and model robustness.

## 4. Results and Interpretations

### 4.1. Out-of-Sample Performance

The out-of-sample validation demonstrated the model's ability to generalize to unseen option price surfaces. Scatter plots of actual vs. predicted parameters showed a strong correlation, with points clustering around the perfect prediction line. Error distributions were generally centered around zero, indicating minimal bias.

### 4.2. Forward-Walking Stability

Forward-walking validation revealed the model's stability over simulated time. The average Mean Absolute Error (MAE) and Loss (MSE) remained consistent across folds, suggesting the model's robustness to evolving data patterns (within the synthetic data context).

### 4.3. Error Significance

Statistical significance tests (e.g., paired t-tests on absolute errors) confirmed that the mean absolute errors for individual parameters were statistically significant from zero, as expected for a regression task. Further analysis of error distributions (histograms and Q-Q plots) provided insights into the normality and homoscedasticity of the errors, crucial for assessing model reliability.

### 4.4. Sensitivity Insights

Sensitivity analysis plots illustrated the responsiveness of predicted Lévy parameters to changes in specific input option prices. This analysis helps identify which market observations have the most significant impact on calibrated parameters, offering valuable insights into the model's interpretability and potential vulnerabilities to noisy inputs.

## 5. Production Readiness

The project includes a production-ready FastAPI API for real-time calibration requests. A GitHub Actions CI/CD pipeline ensures code quality, automated testing, and streamlined deployment processes. This infrastructure supports continuous development and reliable operation in a production environment.

## 6. Future Work and Roadmap

*   **Real-World Data Integration:** Incorporate actual market option data for training and validation, moving beyond synthetic datasets.
*   **Advanced Lévy Models:** Extend the framework to include more complex Lévy models or stochastic volatility models.
*   **Uncertainty Quantification:** Fully implement and integrate Bayesian calibration methods (MCMC/Variational Inference) to provide robust uncertainty estimates for calibrated parameters.
*   **Performance Optimization:** Explore GPU acceleration for the pricing engine and ML inference, and distributed training for larger datasets.
*   **User Interface:** Develop a web-based user interface for interactive calibration and visualization.
*   **Risk Management Integration:** Integrate calibrated parameters directly into risk management systems for VaR/CVaR calculations and stress testing.

## 7. Conclusion

This project successfully establishes a robust, machine learning-driven framework for the rapid and accurate calibration of Lévy models. By addressing the computational challenges of traditional methods, it provides a powerful tool for quantitative analysts and traders, enabling more dynamic and informed decision-making in complex financial markets. The modular design and comprehensive validation ensure its reliability and extensibility for future enhancements.

<!-- This is a test comment to trigger GitHub Actions -->