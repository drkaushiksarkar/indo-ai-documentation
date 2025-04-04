---
layout: default
---

# Climate Smart Indonesia - Documentation

A comprehensive system for predicting dengue fever outbreaks in Indonesia using climate data and neural network models.

## Documentation

- [Project Overview and Models](dengue-prediction-documentation.html)
- [Model API Documentation](model-api-docs.html)
- [Model Implementation Code Examples](code-examples.html)
- [Deployment Guide](deployment-guide.html)

## Overview

This project leverages climate data encoded through Variational Autoencoders (VAEs) to predict dengue fever outbreaks across Indonesia. By combining traditional epidemiological indicators with latent climate representations, our models demonstrate enhanced prediction capabilities for disease incidence rates.

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

## Results

Our models demonstrate strong predictive performance for dengue incidence across Indonesia:

| Model                    | Test RMSE | Test R² | Test SMAPE | Durbin-Watson |
|--------------------------|-----------|---------|------------|---------------|
| Simple LSTM              | 0.9912    | 0.42    | 24.37%     | 1.86          |
| Hybrid LSTM-GRU-Attention| 0.8733    | 0.53    | 19.45%     | 1.94          |
| Conditional GAN          | 0.8412    | 0.56    | 18.72%     | 1.97          |

## Contributors

- Naufal Prawironegoro
- Dr. Kaushik Sharkar

## Getting Started

For detailed instructions on installation, usage, and deployment, see our [documentation](dengue-prediction-documentation.html).
