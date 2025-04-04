# Climate Data Analysis with Variational Autoencoders

## Project Overview

This project implements and evaluates two types of Variational Autoencoders (VAEs) for meteorological data analysis:
1. A standard VAE
2. A Conditional Attention VAE with country and year embeddings

The models encode high-dimensional meteorological features into a lower-dimensional latent space, enabling more efficient data representation and analysis of climate patterns across different countries and time periods.

## Table of Contents

- [Data Sources](#data-sources)
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
- [Future Enhancements](#future-enhancements)

## Data Sources

The primary dataset is stored in `combined_data.csv`, which contains meteorological measurements with the following characteristics:

- **Temporal Coverage**: Data spans multiple years, beginning from 1955
- **Spatial Coverage**: Multiple countries represented in the dataset
- **Features**: 28 meteorological measurements including:
  - Temperature variables (2t, skt)
  - Moisture variables (swvl1-4)
  - Pressure variables (msl, sp)
  - Precipitation (tp, cp)
  - Wind (10si)
  - Geographic coordinates (latitude, longitude)

Data is organized by date and country, with each row representing a specific location and time point.

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
   - Visualization of cluster centers and memberships
   - Pairwise distances between points in latent space

## Results Comparison

| Metric                  | Basic VAE      | Conditional Attention VAE |
|-------------------------|----------------|---------------------------|
| Test MSE                | 0.9826         | 0.7626                    |
| Test RMSE               | 0.9912         | 0.8733                    |
| Silhouette Score        | 0.0459         | 0.1156                    |
| Latent Space Variance   | 0.0002         | 0.0007                    |
| Early Stopping Epoch    | 311            | *                         |

The Conditional Attention VAE outperforms the Basic VAE across all metrics:
- 22.4% lower reconstruction error (MSE)
- 152% higher silhouette score (better cluster separation)
- 250% higher latent space variance (better utilization of latent dimensions)

## Latent Space Analysis

The latent space analysis reveals:

1. **Cluster Separation**:
   - Conditional VAE produces more distinct, separable clusters
   - Higher silhouette scores indicate better-defined structure

2. **Density Distribution**:
   - Basic VAE creates a more uniform but less informative latent space
   - Conditional VAE creates more concentrated regions corresponding to specific climate patterns

3. **Correlation with Original Features**:
   - Conditional VAE's latent dimensions show stronger correlations with original climate features
   - This indicates better preservation of important climate relationships

4. **PCA Explained Variance**:
   - Conditional VAE's latent space retains more information in fewer dimensions

## Data Preprocessing and Transformation

### Input Data Processing

1. **Feature Scaling**:
   - StandardScaler applied to 28 meteorological features
   - Ensures all features contribute equally to model training

2. **Categorical Encoding**:
   - Countries converted to integer indices (0 to num_countries-1)
   - Years converted to integer indices (0 to num_years-1)

3. **Train-Test Split**:
   - 80% training data, 20% testing data
   - Random state 42 for reproducibility

### Output Data Processing

1. **Latent Vector Extraction**:
   - Trained model extracts latent representation (11-dimensional vectors)
   - Each vector encodes climate pattern for a specific country and time

2. **Vector String Formatting**:
   - Vectors stored as semicolon-delimited strings
   - Format: "0.123;-0.456;0.789;..."

3. **Aggregation by Date and Country**:
   - Multiple measurements for same (date, country) pair are aggregated
   - Results saved in `conditional_vector_climate_data.csv`

4. **Vector Padding**:
   - Ensures consistent vector dimensions across all entries
   - Padding with zeros when necessary
   - Final format saved in `conditional_vector_climate_data_padded.csv`

## Deployment

### Environment Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)

### Model Export

The trained models can be exported as:

```python
# Save models
torch.save(trained_vae.state_dict(), 'models/basic_vae.pth')
torch.save(trained_conditional_vae.state_dict(), 'models/conditional_attention_vae.pth')

# Save scaler for preprocessing new data
import joblib
joblib.dump(scaler, 'models/feature_scaler.pkl')
```

### Model Loading and Inference

```python
# Load models
vae_model = VAE(input_dim, hidden_dim, latent_dim)
vae_model.load_state_dict(torch.load('models/basic_vae.pth'))

conditional_vae = ConditionalAttentionVAE(
    input_dim, hidden_dim, latent_dim,
    country_embedding_dim, year_embedding_dim,
    num_countries, num_years
)
conditional_vae.load_state_dict(torch.load('models/conditional_attention_vae.pth'))

# Load scaler
scaler = joblib.load('models/feature_scaler.pkl')

# Inference
def encode_climate_data(model, features, country_idx=None, year_idx=None):
    """Encode climate data into latent space representation"""
    # Scale features
    scaled_features = scaler.transform(features)
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    
    # Convert categorical indices if using conditional model
    if country_idx is not None and year_idx is not None:
        country_tensor = torch.tensor(country_idx, dtype=torch.long)
        year_tensor = torch.tensor(year_idx, dtype=torch.long)
        
        # Get latent representation
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(features_tensor, country_tensor, year_tensor)
        
        return mu.numpy()
    else:
        # Basic VAE encoding
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(features_tensor)
        
        return mu.numpy()
```

## Interactive Dashboard

The project includes an interactive web dashboard built with Streamlit for visualizing climate data, disease outbreaks, and their correlations. This dashboard leverages the latent representations from the VAE models to enhance analysis and visualization.

### Dashboard Features

1. **Interactive Hotspot Map**:
   - Visualizes disease outbreaks as hotspots on a satellite map of Sierra Leone
   - Color-coded markers based on hotspot intensity
   - Size of markers corresponds to number of cases
   - Hover data includes cases, temperature, rainfall, and humidity

2. **Global Filters**:
   - Date range selection (start and end dates)
   - Disease type selection
   - Feature engineering method selection (None, PCA, Autoencoder)

3. **Trend Analysis**:
   - Time series visualizations showing correlation between:
     - Disease cases
     - Temperature
     - Rainfall
     - Humidity
   - Interactive line charts with multi-variable display

4. **Anomaly Detection**:
   - Z-score based anomaly detection for disease outbreaks
   - Visual highlight of anomalous data points
   - Filtering capabilities for anomaly thresholds

### Technical Implementation

The dashboard is implemented using:
- **Streamlit**: Main framework for the interactive web application
- **Plotly Express**: Advanced interactive visualizations
- **Pandas/NumPy**: Data manipulation and analysis
- **Mapbox**: Geographic visualization with satellite imagery
- **TensorFlow/Keras**: For implementing the autoencoder feature engineering option

### Dashboard Deployment

To run the dashboard locally:

```bash
# Install required packages
pip install streamlit pandas numpy plotly tensorflow statsmodels

# Run the dashboard
streamlit run dashboard.py
```

The dashboard will be accessible at http://localhost:8501 by default.

### Integration with VAE Models

The dashboard can be integrated with the trained VAE models by:

1. **Loading Latent Representations**:
   ```python
   # Load the latent vectors from the processed data files
   latent_df = pd.read_csv('preprocessing/data/conditional_vector_climate_data_padded.csv')
   
   # Parse the vector strings back to arrays
   latent_df["latent_vector"] = latent_df["conditional_low_dim_padded"].apply(
       lambda x: np.array(eval(x)) if isinstance(x, str) else x
   )
   ```

2. **Using VAE for Feature Engineering**:
   ```python
   # In the transform_weather_features function:
   elif method == "VAE":
       # Load the trained VAE model
       vae_model = ConditionalAttentionVAE(...)
       vae_model.load_state_dict(torch.load('models/conditional_attention_vae.pth'))
       
       # Convert to tensors
       X_train_tensor = torch.tensor(X_train[weather_cols].values, dtype=torch.float32)
       X_test_tensor = torch.tensor(X_test[weather_cols].values, dtype=torch.float32)
       
       # Get latent representations
       with torch.no_grad():
           _, train_mu, _ = vae_model.encode(X_train_tensor, country_train_tensor, year_train_tensor)
           _, test_mu, _ = vae_model.encode(X_test_tensor, country_test_tensor, year_test_tensor)
       
       # Convert to DataFrames
       train_latent = pd.DataFrame(
           train_mu.numpy(), 
           columns=[f'VAE_{i+1}' for i in range(train_mu.shape[1])],
           index=X_train.index
       )
       test_latent = pd.DataFrame(
           test_mu.numpy(), 
           columns=[f'VAE_{i+1}' for i in range(test_mu.shape[1])],
           index=X_test.index
       )
       
       # Combine with original features (excluding weather)
       X_train = X_train.drop(columns=weather_cols)
       X_test = X_test.drop(columns=weather_cols)
       X_train = pd.concat([X_train, train_latent], axis=1)
       X_test = pd.concat([X_test, test_latent], axis=1)
       
       return X_train, X_test
   ```

3. **Enhanced Anomaly Detection**:
   ```python
   # Enhanced anomaly detection using latent representations
   def detect_anomalies_vae(df, latent_df, threshold=3.0):
       # Match dates and locations with latent vectors
       merged_df = pd.merge(
           df, 
           latent_df[['date', 'country', 'latent_vector']], 
           on=['date', 'country']
       )
       
       # Calculate Mahalanobis distance in latent space
       mean_vector = np.mean(np.stack(merged_df['latent_vector'].values), axis=0)
       cov_matrix = np.cov(np.stack(merged_df['latent_vector'].values).T)
       inv_cov = np.linalg.inv(cov_matrix)
       
       # Calculate distance for each point
       merged_df['mahalanobis'] = merged_df['latent_vector'].apply(
           lambda x: np.sqrt((x - mean_vector).T.dot(inv_cov).dot(x - mean_vector))
       )
       
       # Mark anomalies
       merged_df['vae_anomaly'] = (merged_df['mahalanobis'] > threshold).astype(int)
       
       return merged_df
   ```

## Future Enhancements

1. **Model Architecture Improvements**:
   - Convolutional layers to better capture spatial patterns
   - Transformer-based attention for more complex temporal dependencies
   - Beta-VAE implementation for better disentanglement of latent factors

2. **Training Enhancements**:
   - Cyclical annealing of KL weight
   - Learning rate scheduling
   - Higher-dimensional latent space with regularization

3. **Additional Features**:
   - Integration with climate indices (ENSO, NAO, etc.)
   - Incorporation of satellite imagery data
   - Long-term forecast capabilities

4. **Dashboard Enhancements**:
   - Real-time data integration
   - Predictive analytics for disease outbreak forecasting
   - Comparative analysis between different regions
   - Advanced machine learning model integration

5. **Interpretability**:
   - Feature importance analysis in latent space
   - Climate pattern classification and labeling
   - Counterfactual analysis for climate scenarios
