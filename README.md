# Levy Model Calibration Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/travis/com/mohin-io/levy-model-calibration.svg)](https://travis-ci.com/mohin-io/levy-model-calibration)
[![Code Coverage](https://img.shields.io/codecov/c/github/mohin-io/levy-model-calibration.svg)](https://codecov.io/gh/mohin-io/levy-model-calibration)

A state-of-the-art deep learning framework for calibrating Lévy-based stochastic models in quantitative finance. This engine provides a high-performance, research-ready platform for academics and practitioners to accelerate financial modeling and derivatives pricing.

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About The Project

The `levy-model-calibration` project is designed to address the computationally intensive task of model calibration in modern quantitative finance. Traditional methods are often slow and struggle with complex models. This project leverages neural networks to learn the mapping from market option prices to Lévy model parameters, enabling near-instantaneous calibration.

Key features:
- **Speed:** Calibrate complex models like CGMY and Variance Gamma in milliseconds.
- **Accuracy:** Achieve high accuracy with deep learning-based calibration.
- **Flexibility:** Easily extend the framework with new models and datasets.
- **Data-Driven:** Includes tools for data acquisition, cleaning, and synthetic data generation.

## Getting Started

### Prerequisites

- Python 3.9+
- Pip and Virtualenv

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/mohin-io/levy-model-calibration.git
    cd levy-model-calibration
    ```
2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the calibration engine, you can use the following command:

```sh
python -m models.calibration_net.run --config <path_to_config>
```

For more examples and detailed usage, please refer to the [documentation](https://github.com/mohin-io/levy-model-calibration/wiki).

## Project Structure

```
.
├── analysis/               # Scripts for analyzing model results
├── api/                    # API for accessing the models
├── calibration/            # Model calibration scripts
├── data/                   # Data acquisition, cleaning, and generation
│   ├── acquisition.py
│   ├── cleaning.py
│   ├── processed/
│   ├── raw/
│   └── synthetic/
├── features/               # Feature engineering
├── models/                 # Core models and pricing engines
│   ├── bayesian_calibration/
│   ├── calibration_net/
│   └── pricing_engine/
│       ├── fourier_pricer.py
│       └── levy_models.py
├── production/             # Deployment scripts
├── research/               # Experimental models
└── validation/             # Model validation scripts
```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m '''Add some AmazingFeature''')
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Mohin Hasin - mohinhasin999@gmail.com

Project Link: [https://github.com/mohin-io/levy-model-calibration](https://github.com/mohin-io/levy-model-calibration)