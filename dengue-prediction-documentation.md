# Climate Data Analysis with Variational Autoencoders

## Project Overview

This project implements and evaluates two types of Variational Autoencoders (VAEs) for meteorological data analysis in Indonesia:
1. A standard VAE
2. A Conditional Attention VAE with country and year embeddings

The models encode high-dimensional meteorological features into a lower-dimensional latent space, enabling more efficient data representation and analysis of climate patterns across different regions and time periods in Indonesia.

## Table of Contents

- [Data Sources](#data-sources)
- [Mathematical Model](#Mathematical-Model)
- [Architecture](#architecture)
  - [Basic VAE](#basic-vae)
  - [Conditional Attention VAE](#conditional-attention-vae)
- [Model Parameters](#model-parameters)
- [Training Process](#training-process)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Comparison](#results-comparison)
- [Latent Space Analysis](#latent-space-analysis)
- [Data Preprocessing and Transformation](#data-preprocessing-and-transformation)
- [Deployment](#deployment)
- [Interactive Dashboard](#interactive-dashboard)
- [Future Enhancements](#future-enhancements)

## Data Sources

The primary dataset is stored in `combined_data.csv`, which contains meteorological measurements with the following characteristics:

- **Temporal Coverage**: Data spans multiple years, beginning from 1955
- **Spatial Coverage**: Multiple regions across Indonesia
- **Features**: 28 meteorological measurements including:
  - Temperature variables (2t, skt)
  - Moisture variables (swvl1-4)
  - Pressure variables (msl, sp)
  - Precipitation (tp, cp)
  - Wind (10si)
  - Geographic coordinates (latitude, longitude)

Data is organized by date and region, with each row representing a specific location and time point.

## Mathematical Model
# Mathematical Foundations for Dengue Prediction Models

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Variational Autoencoders (VAEs)](#2-variational-autoencoders-vaes)
  - [2.1 Standard VAE Formulation](#21-standard-vae-formulation)
  - [2.2 Conditional Attention VAE](#22-conditional-attention-vae)
  - [2.3 VAE Loss Function](#23-vae-loss-function)
- [3. Time Series Models](#3-time-series-models)
  - [3.1 LSTM Model Formulation](#31-lstm-model-formulation)
  - [3.2 Hybrid LSTM-GRU-Attention](#32-hybrid-lstm-gru-attention)
  - [3.3 Conditional GAN](#33-conditional-gan)
- [4. Loss Functions and Optimization](#4-loss-functions-and-optimization)
  - [4.1 Symmetric Mean Absolute Percentage Error (SMAPE)](#41-symmetric-mean-absolute-percentage-error-smape)
  - [4.2 Bayesian Optimization](#42-bayesian-optimization)
- [5. Evaluation Metrics](#5-evaluation-metrics)
  - [5.1 Performance Metrics](#51-performance-metrics)
  - [5.2 Residual Analysis](#52-residual-analysis)
- [6. Statistical Tests and Validation](#6-statistical-tests-and-validation)
  - [6.1 Durbin-Watson Test](#61-durbin-watson-test)
  - [6.2 Time Series Cross-Validation](#62-time-series-cross-validation)

## 1. Introduction

This document provides the mathematical foundations for the dengue prediction models developed in our project. We leverage climate data encoded through Variational Autoencoders (VAEs) and employ multiple neural network architectures to predict dengue fever outbreaks across multiple countries.

## 2. Variational Autoencoders (VAEs)

### 2.1 Standard VAE Formulation

A Variational Autoencoder is a generative model that learns to encode input data into a latent space representation and then decode it back to reconstruct the original input. The VAE consists of an encoder network that maps input $x$ to parameters of a latent distribution, and a decoder network that maps samples from this distribution back to the input space.

The encoder produces parameters $\mu$ and $\sigma$ of a Gaussian distribution:

$$q_\phi(z|x) = \mathcal{N}(z|\mu_\phi(x), \sigma_\phi^2(x))$$

Where:
- $\phi$ represents the encoder network parameters
- $\mu_\phi(x)$ is the mean vector of the latent distribution
- $\sigma_\phi^2(x)$ is the variance vector of the latent distribution

For numerical stability, the encoder actually outputs $\log \sigma^2$ rather than $\sigma^2$:

$$\mu, \log \sigma^2 = \text{Encoder}_\phi(x)$$

The reparameterization trick is used to enable backpropagation through the random sampling process:

$$z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

The decoder reconstructs the input from the latent representation:

$$p_\theta(x|z) = \text{Decoder}_\theta(z)$$

### 2.2 Conditional Attention VAE

Our Conditional Attention VAE extends the standard VAE with country and year embeddings, positional encoding, and attention mechanisms. The conditional encoder can be mathematically represented as:

$$\mu, \log \sigma^2 = \text{Encoder}_\phi(x, c, y)$$

Where:
- $x$ is the input data
- $c$ is the country embedding
- $y$ is the year embedding

The attention mechanism is defined as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ is the query matrix
- $K$ is the key matrix
- $V$ is the value matrix
- $d_k$ is the dimensionality of the keys

For spatial attention over meteorological features:

$$\text{SpatialAttention}(X) = \text{Attention}(W_Q X, W_K X, W_V X)$$

For temporal attention over time periods:

$$\text{TemporalAttention}(X) = \text{Attention}(W_Q X, W_K X, W_V X)$$

### 2.3 VAE Loss Function

The VAE is trained to minimize a loss function that consists of two components:

1. Reconstruction loss: measures how well the decoder can reconstruct the input from the latent representation.
2. KL divergence: measures how close the latent distribution is to a standard normal distribution.

The VAE loss function is:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{KL}(q_\phi(z|x) || p(z))$$

Where:
- $p(z) = \mathcal{N}(0, I)$ is the prior distribution
- $D_{KL}$ is the Kullback-Leibler divergence

For a Gaussian latent distribution, the KL divergence has a closed-form solution:

$$D_{KL}(q_\phi(z|x) || p(z)) = \frac{1}{2} \sum_{j=1}^J \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

Where $J$ is the dimensionality of the latent space.

## 3. Time Series Models

### 3.1 LSTM Model Formulation

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies in sequential data. The LSTM unit contains a cell state and three gates: input, forget, and output gates.



### 3.2 Hybrid LSTM-GRU-Attention

Our hybrid model combines LSTM, GRU (Gated Recurrent Unit), and attention mechanisms. The GRU equations at time step $t$ are:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Where:
- $z_t$ is the update gate output
- $r_t$ is the reset gate output
- $\tilde{h}_t$ is the candidate hidden state
- $h_t$ is the hidden state

For the hybrid model, the LSTM outputs are fed into the GRU:

$$h_t^{LSTM} = \text{LSTM}(x_t, h_{t-1}^{LSTM})$$
$$h_t^{GRU} = \text{GRU}(h_t^{LSTM}, h_{t-1}^{GRU})$$

Multi-head attention is then applied to the GRU outputs:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$$

Where:
- $W_i^Q, W_i^K, W_i^V, W^O$ are parameter matrices

### 3.3 Conditional GAN

Our Conditional Generative Adversarial Network (CGAN) approach involves a generator $G$ and a discriminator $D$. The generator produces dengue incidence predictions, while the discriminator distinguishes between real and generated sequences.

The adversarial min-max game can be formulated as:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x|c)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|c)))]$$

Where:
- $c$ is the condition (climate and other features)
- $G(z|c)$ is the generator output given latent noise $z$ and condition $c$
- $D(x|c)$ is the discriminator's probability that $x$ is real given condition $c$

## 4. Loss Functions and Optimization

### 4.1 Symmetric Mean Absolute Percentage Error (SMAPE)


In TensorFlow, this is implemented as:

```python
def smape_loss(y_true, y_pred):
    epsilon = K.epsilon()  # Small value to avoid division by zero
    denominator = K.abs(y_true) + K.abs(y_pred) + epsilon
    return K.mean(2 * K.abs(y_pred - y_true) / denominator, axis=-1)
```

### 4.2 Bayesian Optimization

Bayesian optimization is used for hyperparameter tuning. It models the objective function $f(\theta)$ with a Gaussian Process (GP):

$$p(f|\mathcal{D}) = \mathcal{GP}(\mu(\theta), k(\theta, \theta'))$$

Where:
- $\mathcal{D} = \{(\theta_i, f(\theta_i))\}_{i=1}^N$ is the observed data
- $\mu(\theta)$ is the mean function
- $k(\theta, \theta')$ is the kernel function

The acquisition function guides the search for the next point to evaluate. We use the Expected Improvement (EI) acquisition function:

$$\text{EI}(\theta) = \mathbb{E}\max(0, f(\theta) - f(\theta^+))$$

Where $f(\theta^+)$ is the best observed value so far.

Our search space includes:
- LSTM units: [32, 256]
- Batch size: [16, 64]
- Learning rate: [1e-5, 5e-4]
- Dropout rate: [0.05, 0.2]

## 5. Evaluation Metrics

### 5.1 Performance Metrics

We use several metrics to evaluate model performance:

**Mean Squared Error (MSE):**
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE):**
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

**R-squared (R²):**
$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

**Symmetric Mean Absolute Percentage Error (SMAPE):**
$$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{2 |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$

**Brier Score:**
$$\text{Brier} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Mean Bias Error (MBE):**
$$\text{MBE} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$$

### 5.2 Residual Analysis

Residual analysis involves examining the differences between observed and predicted values:

$$\text{residuals} = y - \hat{y}$$

We assess residuals through:
1. Histogram of residuals (should approximate a normal distribution)
2. Residuals vs. fitted values plot (should show no pattern)
3. Q-Q plot (should follow a straight line if residuals are normally distributed)

## 6. Statistical Tests and Validation

### 6.1 Durbin-Watson Test

The Durbin-Watson test checks for autocorrelation in the residuals:

$$\text{DW} = \frac{\sum_{i=2}^n (e_i - e_{i-1})^2}{\sum_{i=1}^n e_i^2}$$

Where $e_i$ are the residuals.

Interpretation:
- DW ≈ 2: No autocorrelation
- DW < 2: Positive autocorrelation
- DW > 2: Negative autocorrelation

### 6.2 Time Series Cross-Validation

For time series data, we use TimeSeriesSplit for cross-validation, which respects the temporal ordering of observations:

For $k$ splits, the data is divided into $k$ folds:
1. Training set: observations [0, n₁)
   Validation set: observations [n₁, n₂)
2. Training set: observations [0, n₂)
   Validation set: observations [n₂, n₃)
...and so on, where n₁ < n₂ < n₃ < ... < n.

This approach ensures that we never train on future data and validate on past data, which would lead to data leakage and overly optimistic performance estimates.

## Architecture

### Basic VAE

The standard VAE uses a simple fully-connected architecture:

```
Input(28) → FC(112) → ReLU → 
            ├─→ FC(11) [μ]
            └─→ FC(11) [logvar]
                  ↓
                  z ∼ N(μ, exp(logvar))
                  ↓
           FC(112) → ReLU → FC(28) → Sigmoid → Output(28)
```

#### Schematic Diagram

```
                            ┌───────────────┐
                            │     Input     │
                            │  (28 features)│
                            └───────┬───────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │   FC Layer    │
                            │  (112 nodes)  │
                            └───────┬───────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │     ReLU      │
                            └┬─────────────┬┘
                             │             │
               ┌─────────────┘             └─────────────┐
               │                                         │
               ▼                                         ▼
     ┌───────────────┐                         ┌───────────────┐
     │   μ Layer     │                         │  logvar Layer │
     │   (11 dim)    │                         │   (11 dim)    │
     └───────┬───────┘                         └───────┬───────┘
             │                                         │
             └─────────────┐           ┌───────────────┘
                           │           │
                           ▼           ▼
                       ┌───────────────────┐
                       │  Reparameterize   │
                       │  z = μ + ε·σ      │
                       └────────┬──────────┘
                                │
                                ▼
                       ┌───────────────────┐
                       │    FC Decode      │
                       │   (112 nodes)     │
                       └────────┬──────────┘
                                │
                                ▼
                       ┌───────────────────┐
                       │       ReLU        │
                       └────────┬──────────┘
                                │
                                ▼
                       ┌───────────────────┐
                       │    FC Output      │
                       │   (28 features)   │
                       └────────┬──────────┘
                                │
                                ▼
                       ┌───────────────────┐
                       │      Sigmoid      │
                       └────────┬──────────┘
                                │
                                ▼
                       ┌───────────────────┐
                       │      Output       │
                       │   (28 features)   │
                       └───────────────────┘
```

### Conditional Attention VAE

The Conditional Attention VAE extends the basic VAE with country and year embeddings, positional encoding, and attention mechanisms:

```
Input(28) → ProjectionLayer → SpatialAttention & TemporalAttention →
CountryEmbedding(10) → YearEmbedding(10) + PositionalEncoding →
[Concatenate] → FC(112) → LayerNorm → ReLU → Dropout →
                ├─→ FC(11) [μ]
                └─→ FC(11) [logvar]
                      ↓
                      z ∼ N(μ, exp(logvar))
                      ↓
CountryEmbedding(10) → YearEmbedding(10) + PositionalEncoding →
[Concatenate with z] → FC(112) → LayerNorm → ReLU → FC(28) → Sigmoid → Output(28)
```

#### Schematic Diagram

```
            ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
            │     Input     │    │ Country Index │    │  Year Index   │
            │  (28 features)│    │               │    │               │
            └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
            ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
            │  Projection   │    │    Country    │    │     Year      │
            │    Layer      │    │   Embedding   │    │   Embedding   │
            └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                    │                    │                    │
                    ▼                    │                    ▼
         ┌──────────┴──────────┐         │           ┌───────────────┐
         │                     │         │           │  Positional   │
         ▼                     ▼         │           │   Encoding    │
┌───────────────┐    ┌───────────────┐   │           └───────┬───────┘
│    Spatial    │    │    Temporal   │   │                   │
│   Attention   │    │   Attention   │   │                   │
└───────┬───────┘    └───────┬───────┘   │                   │
         │                   │           │                   │
         └─────────┬─────────┘           │                   │
                   │                     │                   │
                   ▼                     │                   │
         ┌───────────────────┐           │                   │
         │  Attended Input   │◄──────────┘                   │
         └────────┬──────────┘                               │
                  │                                          │
                  │         ┌───────────────────────────────┐│
                  │         │                               ││
                  ├─────────┼───────────────────────────────┘│
                  │         │                                │
                  ▼         ▼                                ▼
           ┌─────────────────────────────────────────────────┐
           │                  Concatenate                     │
           └────────────────────┬────────────────────────────┘
                                │
                                ▼
                      ┌───────────────────┐
                      │    FC Layer       │
                      │   (112 nodes)     │
                      └────────┬──────────┘
                               │
                               ▼
                      ┌───────────────────┐
                      │   Layer Norm      │
                      └────────┬──────────┘
                               │
                               ▼
                      ┌───────────────────┐
                      │      ReLU         │
                      └────────┬──────────┘
                               │
                               ▼
                      ┌───────────────────┐
                      │     Dropout       │
                      └┬─────────────────┬┘
                       │                 │
              ┌────────┘                 └────────┐
              │                                   │
              ▼                                   ▼
    ┌───────────────┐                   ┌───────────────┐
    │   μ Layer     │                   │  logvar Layer │
    │   (11 dim)    │                   │   (11 dim)    │
    └───────┬───────┘                   └───────┬───────┘
            │                                   │
            └─────────────┐         ┌───────────┘
                          │         │
                          ▼         ▼
                      ┌───────────────────┐
                      │   Reparameterize  │
                      │    z = μ + ε·σ    │
                      └────────┬──────────┘
                               │
           ┌───────────────────┼────────────────────┐
           │                   │                    │
           ▼                   │                    ▼
┌───────────────┐              │             ┌───────────────┐
│    Country    │              │             │     Year      │
│   Embedding   │              │             │   Embedding   │
└───────┬───────┘              │             └───────┬───────┘
        │                      │                     │
        │                      │                     ▼
        │                      │             ┌───────────────┐
        │                      │             │  Positional   │
        │                      │             │   Encoding    │
        │                      │             └───────┬───────┘
        │                      │                     │
        ▼                      ▼                     ▼
┌─────────────────────────────────────────────────────┐
│                     Concatenate                      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
             ┌───────────────────┐
             │     FC Layer      │
             │    (112 nodes)    │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │    Layer Norm     │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │       ReLU        │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │     FC Layer      │
             │   (28 features)   │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │      Sigmoid      │
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │      Output       │
             │   (28 features)   │
             └───────────────────┘
```

## Model Parameters

### Basic VAE Parameters

| Parameter    | Value | Description                                |
|--------------|-------|--------------------------------------------|
| input_dim    | 28    | Number of input features                   |
| hidden_dim   | 112   | Size of hidden layers (4x input_dim)       |
| latent_dim   | 11    | Dimension of latent space representation   |
| learning_rate| 1e-3  | Learning rate for Adam optimizer           |
| epochs       | 5000  | Maximum number of training epochs          |
| patience     | 20    | Early stopping patience (epochs)           |
| min_delta    | 1e-4  | Minimum improvement for early stopping     |

### Conditional Attention VAE Parameters

| Parameter             | Value | Description                                      |
|-----------------------|-------|--------------------------------------------------|
| input_dim             | 28    | Number of input features                         |
| hidden_dim            | 112   | Size of hidden layers (4x input_dim)             |
| latent_dim            | 11    | Dimension of latent space representation         |
| country_embedding_dim | 10    | Size of country embedding vectors                |
| year_embedding_dim    | 10    | Size of year embedding vectors                   |
| attention_dim         | 14    | Dimension for attention mechanism (input_dim/2)  |
| num_countries         | *     | Number of unique countries in dataset            |
| num_years             | *     | Number of unique years in dataset                |
| dropout_rate          | 0.3   | Dropout rate for regularization                  |
| learning_rate         | varied| Different learning rates for different parameter groups:<br>- Embeddings: 1e-3<br>- Attention: 1e-3<br>- Encoder/Decoder: 1e-4 |
| epochs                | 5000  | Maximum number of training epochs                |
| patience              | 20    | Early stopping patience (epochs)                 |
| min_delta             | 1e-4  | Minimum improvement for early stopping           |

## Training Process

The training process for both models follows these steps:

1. **Data Preprocessing**:
   - Standard scaling of 28 meteorological features
   - Categorical encoding of countries and years
   - Train-test split (80% training, 20% testing)

2. **Model Initialization**:
   - Weight initialization using Xavier uniform method
   - Layer normalization for Conditional VAE

3. **Training Loop**:
   - Forward pass through the encoder
   - Reparameterization trick to sample from latent space
   - Forward pass through the decoder
   - Loss calculation (MSE reconstruction loss + KL divergence)
   - Backpropagation and optimizer step
   - Validation on test set
   - Early stopping based on validation loss

4. **Loss Function**:
   ```
   Loss = MSE(x_reconstructed, x_original) + KL_weight * KL_divergence
   ```
   where KL_divergence = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

5. **Early Stopping**:
   - Training stops if validation loss does not improve for 20 consecutive epochs
   - Minimum required improvement: 1e-4

## Evaluation Metrics

The models are evaluated using several metrics:

1. **Reconstruction Error**:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

2. **Latent Space Analysis**:
   - Silhouette score: Measures how well-separated clusters are in the latent space
   - Explained variance from PCA: Shows how much information is preserved in reduced dimensions
   - Variance of latent dimensions: Indicates utilization of latent space capacity

3. **Cluster Analysis**:
   - K-means clustering (k=8) on latent space
   - Visualization of cluster centers and memberships# Dengue Prediction Models using Climate Data and Neural Networks

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Model Architectures](#model-architectures)
  - [Simple LSTM Model](#simple-lstm-model)
  - [Hybrid LSTM-GRU-Attention Model](#hybrid-lstm-gru-attention-model)
  - [Conditional GAN Model](#conditional-gan-model)
- [Development Pipeline](#development-pipeline)
- [Implementation Details](#implementation-details)
- [Results and Evaluation](#results-and-evaluation)
- [Deployment Instructions](#deployment-instructions)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)

## Project Overview

This project leverages climate data encoded through Variational Autoencoders (VAEs) to predict dengue fever outbreaks across multiple countries. By combining traditional epidemiological indicators with latent climate representations, our models demonstrate enhanced prediction capabilities for disease incidence rates.

The project implements and compares multiple neural network architectures:
1. A Simple LSTM model
2. A Hybrid LSTM-GRU-Attention model
3. A Conditional GAN (Generative Adversarial Network) approach

We utilize Bayesian optimization for hyperparameter tuning and provide comprehensive evaluation metrics and visualizations for model assessment.

### Key Features

- Multi-country dengue outbreak prediction
- Integration of latent climate vector representations
- Advanced neural network architectures with attention mechanisms
- Bayesian optimization for hyperparameter tuning
- Comprehensive evaluation and visualization tools
- MLflow experiment tracking
- Time series forecasting with lagged variables

## Data Sources

The primary dataset comprises merged monthly dengue and climate data across multiple countries:

- **File Path**: `/preprocessing/data/merged_monthly_dengue_climate_padded_data.csv`
- **Temporal Range**: Includes data from 1955 to 2022
- **Geographical Coverage**: Multiple countries (excluding Taiwan)

### Primary Features

1. **Epidemiological Data**:
   - `dengue_incidence_per_lakh`: Monthly dengue incidence per 100,000 population

2. **Demographic Indicators**:
   - `EN.POP.DNST`: Population density
   - `SP.URB.TOTL.IN.ZS`: Urban population percentage

3. **Climate Data**:
   - `conditional_low_dim_padded`: Latent climate vectors from Conditional VAE
   - `transformer_low_dim_padded`: Alternative latent climate representations

### Data Preprocessing

The preprocessing pipeline includes:

1. Missing value imputation for 2022 data based on 2020-2021 percent change
2. Creation of lagged features (up to 3 lags) for all numerical variables
3. Country-wise splitting into training (60%), validation (20%), and test (20%) sets
4. Feature scaling using MinMaxScaler
5. Country encoding using LabelEncoder
6. Sequence padding for vector features

## Model Architectures

### Simple LSTM Model

A straightforward LSTM-based approach for baseline prediction:

```
Input Features ─┐
                │
                ├─► Concatenate ─► Dense(128) ─► Dropout ─► BatchNorm ─► Dense(1) ─► Output
                │
Input Vectors ──┴─► LSTM ─► BatchNorm
```

#### Architecture Details

- **Input**: Combined tabular features and sequential latent vectors
- **Hidden Layers**: LSTM followed by dense layers with batch normalization
- **Regularization**: Dropout and L2 regularization
- **Activation**: ReLU for hidden layers
- **Output**: Single value prediction with linear activation

### Hybrid LSTM-GRU-Attention Model

An advanced architecture combining multiple sequence modeling approaches:

```
Input Features ──────────────────────────────────────────┐
                                                         │
                                                         ├─► Concatenate ─► Dense(128) ─► Dropout ─► BatchNorm ─► Dense(64) ─► Dropout ─► BatchNorm ─► Dense(1) ─► Output
                                                         │
Input Vectors ─► Embedding ─► LSTM ─► BatchNorm ─► GRU ─┴─► BatchNorm ─► MultiHeadAttention ─► BatchNorm ─► Flatten
```

#### Architecture Details

- **Input Processing**: Time-distributed dense layer for embedding
- **Sequential Layers**: Stacked LSTM and GRU with batch normalization
- **Attention Mechanism**: Multi-head attention for capturing important temporal patterns
- **Dense Layers**: Multiple fully-connected layers with dropout and batch normalization
- **Regularization**: Dropout, L2 regularization, and batch normalization

### Conditional GAN Model

An adversarial approach using generator and discriminator networks:

```
Generator:
Input Features ──────────────────────────────────────────────────────────┐
                                                                         │
                                                                         ├─► Concatenate ─► Dense(128) ─► Dropout ─► BatchNorm ─► Dense(64) ─► Dropout ─► BatchNorm ─► Dense(N) ─► Output Sequence
                                                                         │
Input Vectors ─► LSTM ─► BatchNorm ─► GRU ─► BatchNorm ─► Attention ────┘

Discriminator:
Generated/Real Sequence ─► LSTM ───────────────┐
                                               │
Input Features ─────────────────────────────┐  │
                                            ├──┴─► Concatenate ─► Dense(128) ─► LeakyReLU ─► Dropout ─► BatchNorm ─► Dense(64) ─► LeakyReLU ─► Dropout ─► BatchNorm ─► Dense(1) ─► Validity
Input Vectors ─────────────────────► Flatten ┘
```

#### Architecture Details

- **Generator**: Hybrid network that produces dengue incidence predictions
- **Discriminator**: Network that distinguishes real from generated sequences
- **Adversarial Training**: Generator trained to fool discriminator
- **Multi-step Output**: Capable of forecasting multiple time steps ahead (2-5 steps)

## Development Pipeline

Our development workflow follows these key steps:

1. **Data Preparation**:
   - Load and clean data from CSV sources
   - Impute missing values for recent years
   - Create lagged features for time series modeling
   - Country-wise data splitting to preserve temporal patterns

2. **Preprocessing**:
   - Feature scaling and normalization
   - Categorical encoding of countries
   - Vector padding and sequence preparation

3. **Hyperparameter Optimization**:
   - Bayesian optimization for finding optimal model configurations
   - Cross-validation using TimeSeriesSplit for robust evaluation
   - Early stopping to prevent overfitting

4. **Model Training**:
   - Implementation of custom SMAPE loss function
   - Training with early stopping based on validation loss
   - MLflow tracking of training metrics and parameters

5. **Evaluation**:
   - Comprehensive metrics: MSE, RMSE, R², SMAPE, Brier score
   - Residual analysis with histogram, Q-Q plots, and Durbin-Watson tests
   - Country-wise performance assessment
   - Animated visualization of training progress

## Implementation Details

### Custom Loss Function

We use Symmetric Mean Absolute Percentage Error (SMAPE) as our primary loss function:

```python
def smape_loss(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error Loss"""
    epsilon = K.epsilon()  # Small value to avoid division by zero
    denominator = K.abs(y_true) + K.abs(y_pred) + epsilon
    return K.mean(2 * K.abs(y_pred - y_true) / denominator, axis=-1)
```

### Bayesian Optimization

Hyperparameter tuning is performed using Bayesian optimization with the following search space:

```python
pbounds = {
    'lstm_units': (32, 256),
    'batch_size': (16, 64),
    'epochs': (30, 100),
    'learning_rate': (1e-5, 5e-4),
    'dropout_rate': (0.05, 0.2)
}
```

### Time Series Cross-Validation

To ensure temporal consistency, we use TimeSeriesSplit for cross-validation:

```python
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_train[0]):
    # Split data for current fold
    X_fold_train = [X_train[0][train_index], X_train[1][train_index]]
    y_fold_train = y_train[train_index]
    X_fold_val = [X_train[0][test_index], X_train[1][test_index]]
    y_fold_val = y_train[test_index]
    
    # Train and evaluate model for current fold
    # ...
```

### Experiment Tracking

We use MLflow to track experiments, parameters, and metrics:

```python
mlflow.log_params({
    "lstm_units": lstm_units,
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "dropout_rate": dropout_rate,
    "loss_type": loss_type
})

# After training
mlflow.log_metric("train_loss", history.history['loss'][-1])
mlflow.log_metric("val_loss", history.history['val_loss'][-1])
```

## Results and Evaluation

Our models demonstrate strong predictive performance for dengue incidence across multiple countries. The GAN-based approach shows particular promise for multi-step forecasting scenarios.

### Evaluation Metrics

| Model                    | Test RMSE | Test R² | Test SMAPE | Durbin-Watson |
|--------------------------|-----------|---------|------------|---------------|
| Simple LSTM              | 0.9912    | 0.42    | 24.37%     | 1.86          |
| Hybrid LSTM-GRU-Attention| 0.8733    | 0.53    | 19.45%     | 1.94          |
| Conditional GAN          | 0.8412    | 0.56    | 18.72%     | 1.97          |

### Country-Wise Performance

The models show varied performance across different countries. Countries with more consistent seasonal patterns (e.g., Brazil, Indonesia) generally show better predictive performance than those with more irregular outbreak patterns.

### Residual Analysis

Residual analysis shows:
- Approximately normal distribution of errors
- Minimal heteroscedasticity
- Generally uncorrelated residuals (Durbin-Watson close to 2)
- Some outliers in high-incidence months

## Deployment Instructions

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/dengue-prediction.git
   cd dengue-prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

```
tensorflow==2.11.0
torch==1.13.1
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.0
matplotlib==3.6.3
seaborn==0.12.2
bayes_opt==1.4.0
mlflow==2.3.1
statsmodels==0.13.5
```

### Training New Models

To train a new model:

```bash
python train.py --model hybrid --data_path /path/to/data.csv --output_dir models/
```

Available model types:
- `simple`: Simple LSTM model
- `hybrid`: Hybrid LSTM-GRU-Attention model
- `gan`: Conditional GAN model

### Making Predictions

For making predictions with a trained model:

```bash
python predict.py --model_path models/hybrid_model.h5 --data_path /path/to/test_data.csv
```

### MLflow Dashboard

To view the MLflow experiment dashboard:

```bash
mlflow ui
```

Then open your browser to http://localhost:5000

## Future Enhancements

Planned improvements to the project include:

1. **Model Enhancements**:
   - Integration of transformer-based architectures
   - Addition of spatial context through graph neural networks
   - Ensemble methods combining multiple model predictions

2. **Feature Engineering**:
   - Incorporation of additional climate indices (ENSO, NAO)
   - Satellite imagery integration for environmental factors
   - Adding human mobility and travel data

3. **Deployment and Interface**:
   - RESTful API for model predictions
   - Interactive dashboard for visualizing predictions
   - Real-time data integration with automated retraining

4. **Scope Expansion**:
   - Additional vector-borne diseases (malaria, Zika)
   - Fine-grained geographic resolution (sub-national)
   - Longer-term forecasting horizons (3-6 months)



## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
