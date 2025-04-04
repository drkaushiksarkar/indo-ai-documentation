# Climate Smart Indonesia Multi-Disease Prediction System

![License](https://img.shields.io/badge/license-MIT-blue)
![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)

A comprehensive system for predicting climate-sensitive disease outbreaks and scenarios in Indonesia using disease surveillance data, climate data, and advanced neural network models.

## Overview

This project leverages climate data encoded to predict climate-sensitive disease outbreaks across Indonesia. By combining traditional epidemiological indicators with latent climate representations, our models demonstrate enhanced prediction capabilities for disease incidence rates.

### Key Features

- Multi-region dengue outbreak prediction for Indonesia
- Integration of latent climate vector representations
- Advanced neural network architectures with attention mechanisms
- Bayesian optimization for hyperparameter tuning
- Comprehensive evaluation and visualization tools
- MLflow experiment tracking
- Time series forecasting with lagged variables

## Models

The project implements multiple model architectures:

1. **Simple LSTM Model**: Baseline approach using LSTM layers for sequence modeling
2. **Hybrid LSTM-GRU-Attention Model**: Advanced approach combining multiple recurrent architectures with attention mechanisms
3. **Conditional GAN Model**: Adversarial approach with generator and discriminator networks for improved predictions

## Dataset

The dataset used in this project includes:

- Monthly dengue incidence rates per 100,000 population in Indonesia
- Population density and urbanization metrics
- VAE-encoded climate data containing latent representations of weather patterns
- Data spanning from 1955 to present day

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.4+
- PyTorch 1.9+ (for VAE models)
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/NormanMul/Climate-Smart-Indonesia---Documentation.git
cd Climate-Smart-Indonesia---Documentation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Train the hybrid model
python -m training.train --model hybrid --data data/processed/merged_monthly_dengue_climate_padded_data.csv

# Make predictions with a trained model
python -m evaluation.predict --model models/saved_models/hybrid_model.h5 --data data/processed/test_data.csv
```

## Documentation

Full documentation is available in the following sections:

- [Project Overview and Models](dengue-prediction-documentation.md)
- [Model API Documentation](model-api-docs.md)
- [Model Implementation Code Examples](code-examples.md)
- [Deployment Guide](deployment-guide.md)
- [BMKG Integration Guide](docs/bmkg-integration-guide.md)

## Project Structure

```
Climate-Smart-Indonesia/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── bmkg/              # BMKG data storage
│   └── README.md
├── models/
│   ├── saved_models/
│   ├── __init__.py
│   ├── simple_lstm.py
│   ├── hybrid_model.py
│   └── gan_model.py
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py
│   └── feature_engineering.py
├── bmkg_integration/      # BMKG integration module
│   ├── __init__.py
│   ├── api.py
│   ├── wms.py
│   ├── netcdf_utils.py
│   ├── cli.py
│   └── README.md
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── hyperparameter_tuning.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
├── api/
│   ├── __init__.py
│   └── app.py
├── dashboard/
│   ├── assets/
│   └── app.py
├── docs/
│   ├── dengue-prediction-documentation.md
│   ├── model-api-docs.md
│   ├── code-examples.md
│   ├── deployment-guide.md
│   └── bmkg-integration-guide.md
├── requirements.txt
├── setup.py
└── README.md
```

## Usage

### Training Models

```bash
# Train the simple LSTM model
python -m training.train --model simple_lstm --data data/processed/merged_monthly_dengue_climate_padded_data.csv --output models/saved_models/simple_lstm.h5

# Train the hybrid model with custom parameters
python -m training.train --model hybrid --data data/processed/merged_monthly_dengue_climate_padded_data.csv --lstm-units 128 --attention-heads 8 --dropout 0.3 --output models/saved_models/hybrid_model.h5

# Train the GAN model
python -m training.train --model gan --data data/processed/merged_monthly_dengue_climate_padded_data.csv --output models/saved_models/gan_generator.h5
```

### Making Predictions

```bash
# Make predictions using a trained model
python -m evaluation.predict --model models/saved_models/hybrid_model.h5 --data data/processed/test_data.csv --output predictions.csv
```
### BMKG Data Processing

```bash
# Fetch weather forecast data and convert to NetCDF
python -m bmkg_integration.cli fetch-weather --adm4 31.74.06.1002 --output data/bmkg/weather.nc

# Get rainfall map data
python -m bmkg_integration.cli fetch-rainfall --output data/bmkg/rainfall.nc

# Process data for a specific region and date range
python -m bmkg_integration.cli process-region --region jakarta --start 2023-01-01 --end 2023-12-31 --output data/bmkg/jakarta_2023.nc
```

### Running the API

```bash
# Start the prediction API
python -m api.app

# In a separate terminal, make a prediction request
curl -X POST -F "file=@data/processed/test_data.csv" http://localhost:5000/predict

### Running the API

```bash
# Start the prediction API
python -m api.app

# In a separate terminal, make a prediction request
curl -X POST -F "file=@data/processed/test_data.csv" http://localhost:5000/predict
```

### Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboard/app.py
```

## Results

Our models demonstrate strong predictive performance for dengue incidence across Indonesia:

| Model                    | Test RMSE | Test R² | Test SMAPE | Durbin-Watson |
|--------------------------|-----------|---------|------------|---------------|
| Simple LSTM              | 0.9912    | 0.42    | 24.37%     | 1.86          |
| Hybrid LSTM-GRU-Attention| 0.8733    | 0.53    | 19.45%     | 1.94          |
| Conditional GAN          | 0.8412    | 0.56    | 18.72%     | 1.97          |



## Contributors
- Dr. Kaushik Sarkar
- Naufal Prawironegoro


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The ClimateSmart Indonesia initiative was launched in 2023 with funding support from the philanthropy of the President of the UAE through the Reaching The Last Mile.
- The distributed AI and EDGE infrastructure is developed with funding support from the Patrick J. McGovern Foundation.
- Special thanks to KORIKA, the Ministry of Health, Indonesia and BMKG for data and strategic inputs.
- Powered by TensorFlow, PyTorch, and MLflow

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Train the hybrid model
python -m training.train --model hybrid --data data/processed/merged_monthly_dengue_climate_padded_data.csv

# Make predictions with a trained model
python -m evaluation.predict --model models/saved_models/hybrid_model.h5 --data data/processed/test_data.csv
```


## Deployment

See the [Deployment Guide](docs/deployment-guide.md) for detailed instructions on deploying the models in various environments.

### Docker Deployment

```bash
# Build and run with Docker
docker build -t dengue-prediction .
docker run -p 5000:5000 dengue-prediction

# Using docker-compose
docker-compose up -d
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Project developed as part of research on climate-sensitive disease prediction
- Special thanks to contributors and institutions providing the climate and dengue incidence data
- Powered by TensorFlow, PyTorch, and MLflow
