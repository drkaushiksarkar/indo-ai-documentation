# Streamlit Dashboard Documentation

## Overview

The Streamlit Dashboard is an interactive web application designed to visualize the relationship between climate data (processed through our VAE models) and disease outbreaks in Sierra Leone. It provides real-time filtering, advanced visualization, and anomaly detection capabilities.

## Key Features

### 1. Interactive Hotspot Map
![Hotspot Map](https://via.placeholder.com/800x400?text=Hotspot+Map+Screenshot)

- **Technology**: Plotly Scatter_Mapbox with Mapbox integration
- **Visualization**: Geospatial visualization of disease hotspots
- **Interactivity**: 
  - Select specific dates with a slider
  - Hover over points to see detailed information
  - Adjust hotspot threshold in the sidebar
- **Map Styles**: Satellite-streets view for context-rich visualization

### 2. Trends and Climate Overview
![Trends Overview](https://via.placeholder.com/800x400?text=Trends+Overview+Screenshot)

- **Technology**: Plotly Line Charts
- **Visualization**: Multi-variable time series
- **Metrics**:
  - Disease cases
  - Temperature
  - Rainfall
  - Humidity
- **Analysis**: Visual correlation between climate variables and disease prevalence

### 3. Outbreak Anomaly Detection
![Anomaly Detection](https://via.placeholder.com/800x400?text=Anomaly+Detection+Screenshot)

- **Technology**: Z-score statistical analysis
- **Visualization**: Scatter plot of anomalous events
- **Methodology**: 
  - Calculates z-scores for each location and disease
  - Flags points with z-score > 2 as anomalies
  - Color-coded by location ID

### 4. Feature Engineering Options

The dashboard offers different methods for dimensionality reduction and feature engineering:

1. **None**: Uses raw features
2. **PCA**: Principal Component Analysis
   - Transforms climate variables into principal components
   - User-adjustable number of components
3. **Autoencoder**: Neural network-based dimensionality reduction
   - Uses TensorFlow/Keras autoencoder
   - Customizable encoding dimension

## Technical Implementation

### File Structure

```
dashboard/
├── app.py                # Main Streamlit application
├── utils/
│   ├── data_loader.py    # Functions to load and process data
│   ├── visualization.py  # Custom visualization functions
│   └── anomaly.py        # Anomaly detection algorithms
└── assets/
    ├── styles.css        # Custom CSS styling
    └── logo.png          # Application logo
```

### Key Functions

#### 1. `create_dummy_data()`

Generates synthetic data for Sierra Leone, including:
- Disease cases data across 7 diseases
- Weather data (temperature, rainfall, humidity)
- Geographic coordinates
- Time series data spanning 2020

```python
def create_dummy_data():
    # Date range for 2020
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
    n_dates = len(dates)

    # 20 random locations within Sierra Leone
    n_locations = 20
    lats = np.random.uniform(7.5, 10, n_locations)
    lons = np.random.uniform(-13, -10, n_locations)
    
    # Generate data for each date, location, and disease
    # ...
```

#### 2. `transform_weather_features()`

Performs dimensionality reduction on weather features:

```python
def transform_weather_features(X_train, X_test, weather_cols, method, n_components=3):
    if method == "PCA":
        # PCA implementation
        # ...
    elif method == "Autoencoder":
        # Autoencoder implementation
        # ...
    else:
        return X_train, X_test
```

#### 3. `detect_anomalies()`

Identifies statistical anomalies in the disease data:

```python
def detect_anomalies(df):
    df = df.copy()
    # Calculate z-scores for each location and disease group
    df['zscore'] = df.groupby(['location_id', 'disease'])['cases'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    # Flag points with absolute z-score > 2 as anomalies
    df['anomaly'] = (df['zscore'].abs() > 2).astype(int)
    return df
```

## Integration with VAE Models

The dashboard is designed to be integrated with the trained VAE models. Here's how to connect the VAE models to the dashboard:

### 1. Load VAE Model

```python
import torch
from models.conditional_vae import ConditionalAttentionVAE

# Load model parameters
input_dim = 28
hidden_dim = 112
latent_dim = 11
country_embedding_dim = 10
year_embedding_dim = 10
num_countries = len(df['country'].unique())
num_years = len(df['date'].dt.year.unique())

# Initialize model
vae_model = ConditionalAttentionVAE(
    input_dim, hidden_dim, latent_dim,
    country_embedding_dim, year_embedding_dim,
    num_countries, num_years
)

# Load trained weights
vae_model.load_state_dict(torch.load('models/conditional_attention_vae.pth'))
vae_model.eval()
```

### 2. Use VAE for Feature Encoding

Extend the `transform_weather_features()` function to include VAE-based encoding:

```python
elif method == "VAE":
    # Prepare data for VAE
    train_weather = X_train[weather_cols].values
    test_weather = X_test[weather_cols].values
    
    # Convert to tensors
    train_tensor = torch.tensor(train_weather, dtype=torch.float32)
    test_tensor = torch.tensor(test_weather, dtype=torch.float32)
    
    # Get country and year indices
    country_train = torch.tensor(X_train['country_idx'].values, dtype=torch.long)
    country_test = torch.tensor(X_test['country_idx'].values, dtype=torch.long)
    year_train = torch.tensor(X_train['year_idx'].values, dtype=torch.long)
    year_test = torch.tensor(X_test['year_idx'].values, dtype=torch.long)
    
    # Generate latent representations
    with torch.no_grad():
        mu_train, _ = vae_model.encode(train_tensor, country_train, year_train)
        mu_test, _ = vae_model.encode(test_tensor, country_test, year_test)
    
    # Convert to DataFrames
    vae_cols = [f'VAE_{i+1}' for i in range(latent_dim)]
    X_train_vae = pd.DataFrame(mu_train.numpy(), columns=vae_cols, index=X_train.index)
    X_test_vae = pd.DataFrame(mu_test.numpy(), columns=vae_cols, index=X_test.index)
    
    # Replace original features with VAE features
    X_train = X_train.drop(columns=weather_cols)
    X_test = X_test.drop(columns=weather_cols)
    X_train = pd.concat([X_train, X_train_vae], axis=1)
    X_test = pd.concat([X_test, X_test_vae], axis=1)
    
    return X_train, X_test
```

### 3. Enhanced Anomaly Detection

Add a VAE-based anomaly detection method that leverages the latent space:

```python
def detect_anomalies_vae(df, vae_model, threshold=3.0):
    # Prepare input features
    features = df[spatial_features].values
    features_tensor = torch.tensor(features, dtype=torch.float32)
    country_tensor = torch.tensor(df['country_idx'].values, dtype=torch.long)
    year_tensor = torch.tensor(df['year_idx'].values, dtype=torch.long)
    
    # Get latent representation
    with torch.no_grad():
        mu, _ = vae_model.encode(features_tensor, country_tensor, year_tensor)
    
    # Calculate reconstruction error
    with torch.no_grad():
        recon, _, _ = vae_model(features_tensor, country_tensor, year_tensor)
    
    # Mean squared error for each sample
    mse = ((recon - features_tensor) ** 2).mean(dim=1).numpy()
    
    # Flag anomalies based on reconstruction error
    df_result = df.copy()
    df_result['recon_error'] = mse
    df_result['vae_anomaly'] = (mse > threshold).astype(int)
    
    return df_result
```

## User Interface and Interaction

### Sidebar Controls

The dashboard provides a comprehensive set of controls in the sidebar:

1. **Global Filters**:
   - Date range selection (start date and end date)
   - Disease selection dropdown
   - Feature engineering method selection
   - Number of components slider (for PCA and Autoencoder)

2. **Map Controls**:
   - Hotspot threshold slider
   - Date selection slider

### Main Dashboard Sections

The dashboard is organized into the following sections:

1. **Header**: Title and brief description
2. **Interactive Map**: Visualization of disease hotspots
3. **Trends & Climate Overview**: Time series of disease cases and weather variables
4. **Outbreak Anomaly Detection**: Visualization of statistical anomalies

## Customization and Styling

The dashboard includes custom CSS styling for an enhanced visual experience:

```css
<style>
body {
    background: linear-gradient(135deg, #e0eafc, #cfdef3);
}
.main {
    background-color: transparent;
    font-family: 'Roboto', sans-serif;
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #2e7bcf, #1a4e8a);
    color: #ffffff;
}
h1, h2, h3 {
    color: #1a4e8a;
    font-family: 'Roboto', sans-serif;
}
.stButton>button {
    background-color: #1a4e8a;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
}
</style>
```

## Deployment Instructions

### Local Deployment

To run the dashboard locally:

1. **Install Requirements**:
```bash
pip install streamlit pandas numpy plotly tensorflow statsmodels
```

2. **Run the Application**:
```bash
streamlit run app.py
```

3. **Access the Dashboard**:
Open your browser and navigate to http://localhost:8501

### Cloud Deployment

To deploy to Streamlit Sharing:

1. **Create a GitHub Repository**:
   - Push your dashboard code to a GitHub repository
   - Include a `requirements.txt` file with dependencies

2. **Deploy on Streamlit Sharing**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Log in with GitHub
   - Select your repository
   - Configure deployment settings
   - Click "Deploy"

## Future Enhancements

### Planned Dashboard Features

1. **Predictive Analytics**:
   - Disease outbreak forecasting
   - Risk assessment based on climate patterns

2. **Advanced Visualizations**:
   - 3D visualization of latent space
   - Animated time series maps

3. **Machine Learning Integration**:
   - Active learning for anomaly threshold tuning
   - Real-time model retraining option

4. **Data Expansion**:
   - Integration with real climate datasets
   - Support for additional countries and regions

### Known Limitations

1. **Data Limitations**:
   - Currently using synthetic data
   - Limited to Sierra Leone region

2. **Performance Considerations**:
   - Autoencoder feature engineering can be slow
   - Large datasets may cause performance issues

3. **Visualization Constraints**:
   - Limited to 2D visualizations of high-dimensional data
   - Fixed map center and zoom level
