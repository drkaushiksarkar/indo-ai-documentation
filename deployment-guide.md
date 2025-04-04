# Deployment Guide

This guide provides detailed instructions for deploying the Dengue Prediction models in various environments.

## Table of Contents
- [Local Deployment](#local-deployment)
- [Docker Containerization](#docker-containerization)
- [Cloud Deployment](#cloud-deployment)
  - [AWS Deployment](#aws-deployment)
  - [Azure Deployment](#azure-deployment)
  - [Google Cloud Deployment](#google-cloud-deployment)
- [Model Serving API](#model-serving-api)
- [Streamlit Dashboard](#streamlit-dashboard)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

## Local Deployment

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/dengue-prediction.git
   cd dengue-prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Project Structure

Ensure your project follows this structure for proper deployment:

```
dengue-prediction/
├── data/
│   ├── raw/
│   ├── processed/
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
├── requirements.txt
├── setup.py
└── README.md
```

### Running Locally

To train a model:

```bash
# Train the basic LSTM model
python -m training.train --model simple_lstm --data data/processed/merged_monthly_dengue_climate_padded_data.csv --output models/saved_models/simple_lstm.h5

# Train the hybrid model
python -m training.train --model hybrid --data data/processed/merged_monthly_dengue_climate_padded_data.csv --output models/saved_models/hybrid_model.h5

# Train the GAN model
python -m training.train --model gan --data data/processed/merged_monthly_dengue_climate_padded_data.csv --output models/saved_models/gan_generator.h5
```

To make predictions using a trained model:

```bash
python -m evaluation.predict --model models/saved_models/hybrid_model.h5 --data data/processed/test_data.csv --output predictions.csv
```

## Docker Containerization

### Dockerfile

Create a Dockerfile for the project:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command
CMD ["python", "-m", "api.app"]
```

### Docker Compose

For a more complex setup, create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  dengue-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models/saved_models:/app/models/saved_models
    command: python -m api.app
    environment:
      - MODEL_PATH=models/saved_models/hybrid_model.h5
      - PORT=5000

  dengue-dashboard:
    build: 
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - dengue-api
    environment:
      - API_URL=http://dengue-api:5000
```

### Building and Running Docker Containers

```bash
# Build and run the API container
docker build -t dengue-prediction-api .
docker run -p 5000:5000 dengue-prediction-api

# Using docker-compose for the complete setup
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment

#### Using AWS Elastic Beanstalk

1. **Install the EB CLI**:
   ```bash
   pip install awsebcli
   ```

2. **Initialize EB application**:
   ```bash
   eb init -p python-3.8 dengue-prediction
   ```

3. **Create EB environment**:
   ```bash
   eb create dengue-prediction-env
   ```

4. **Deploy the application**:
   ```bash
   eb deploy
   ```

#### Using AWS Lambda for Serverless Deployment

1. **Create a Lambda deployment package**:
   ```bash
   pip install -t ./package -r requirements.txt
   cp -r models preprocessing api ./package/
   cd package
   zip -r ../lambda_deployment.zip .
   ```

2. **Deploy using AWS CLI**:
   ```bash
   aws lambda create-function --function-name dengue-prediction \
     --runtime python3.8 --handler api.lambda_handler.handler \
     --zip-file fileb://lambda_deployment.zip \
     --role arn:aws:iam::123456789012:role/lambda-execution-role \
     --timeout 30 --memory-size 1024
   ```

3. **Create API Gateway**:
   ```bash
   aws apigateway create-rest-api --name 'Dengue Prediction API'
   ```

### Azure Deployment

#### Using Azure App Service

1. **Create App Service**:
   ```bash
   az webapp up --sku F1 --name dengue-prediction-app --resource-group dengue-prediction
   ```

2. **Deploy with GitHub Actions**:
   Create a `.github/workflows/azure-deploy.yml` file:
   ```yaml
   name: Deploy to Azure

   on:
     push:
       branches: [ main ]

   jobs:
     build-and-deploy:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v2
       - name: Set up Python
         uses: actions/setup-python@v2
         with:
           python-version: '3.8'
       - name: Install dependencies
         run: pip install -r requirements.txt
       - name: Deploy to Azure Web App
         uses: azure/webapps-deploy@v2
         with:
           app-name: 'dengue-prediction-app'
           publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
   ```

#### Using Azure Machine Learning Service

1. **Create a deployment configuration file** (`deployment.yml`):
   ```yaml
   name: dengue-prediction
   version: 1
   services:
     dengue-api:
       type: mlflow_model
       model: azureml:models/dengue-hybrid:1
       environment: azureml:AzureML-tensorflow-2.4-ubuntu18.04-py37:1
       compute: dengue-aks
   ```

2. **Deploy the model**:
   ```bash
   az ml model deploy -f deployment.yml
   ```

### Google Cloud Deployment

#### Using Google App Engine

1. **Create an `app.yaml` file**:
   ```yaml
   runtime: python38
   entrypoint: gunicorn -b :$PORT api.app:app

   env_variables:
     MODEL_PATH: "models/saved_models/hybrid_model.h5"
   ```

2. **Deploy to App Engine**:
   ```bash
   gcloud app deploy
   ```

#### Using Google Cloud Run

1. **Build container with Cloud Build**:
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT_ID]/dengue-prediction
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy dengue-prediction \
     --image gcr.io/[PROJECT_ID]/dengue-prediction \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Model Serving API

### Flask API

Create a Flask API for serving predictions in `api/app.py`:

```python
import os
import tempfile
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from preprocessing.data_loader import load_and_prepare_data
from preprocessing.feature_engineering import preprocess_single_dataset

app = Flask(__name__)

# Load the model
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/saved_models/hybrid_model.h5')
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'smape_loss': lambda y_true, y_pred: tf.reduce_mean(
        2 * tf.abs(y_pred - y_true) / (tf.abs(y_true) + tf.abs(y_pred) + tf.keras.backend.epsilon())
    )}
)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        file.save(tmp.name)
        
        # Load and preprocess data
        data = load_and_prepare_data(tmp.name)
        
        # Preprocess for prediction
        feature_columns = ['dengue_incidence_per_lakh', 'EN.POP.DNST', 'SP.URB.TOTL.IN.ZS']
        vector_column = 'conditional_low_dim_padded'
        X, _ = preprocess_single_dataset(data, feature_columns, vector_column, 20)
        
        # Make predictions
        predictions = model.predict(X).flatten()
        
        # Create response
        response = {
            'predictions': predictions.tolist(),
            'countries': data['adm_0_name'].tolist()
        }
        
    # Remove temporary file
    os.unlink(tmp.name)
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

### FastAPI Alternative

For better performance and documentation, use FastAPI:

```python
import os
import tempfile
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from preprocessing.data_loader import load_and_prepare_data
from preprocessing.feature_engineering import preprocess_single_dataset

app = FastAPI(
    title="Dengue Prediction API",
    description="API for predicting dengue incidence using climate data"
)

# Load the model
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/saved_models/hybrid_model.h5')
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'smape_loss': lambda y_true, y_pred: tf.reduce_mean(
        2 * tf.abs(y_pred - y_true) / (tf.abs(y_true) + tf.abs(y_pred) + tf.keras.backend.epsilon())
    )}
)

class PredictionResponse(BaseModel):
    predictions: list
    countries: list

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict dengue incidence from climate data.
    
    The uploaded file should be a CSV containing the required features.
    """
    # Check file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)
    
    try:
        # Load and preprocess data
        data = load_and_prepare_data(tmp_path)
        
        # Preprocess for prediction
        feature_columns = ['dengue_incidence_per_lakh', 'EN.POP.DNST', 'SP.URB.TOTL.IN.ZS']
        vector_column = 'conditional_low_dim_padded'
        X, _ = preprocess_single_dataset(data, feature_columns, vector_column, 20)
        
        # Make predictions
        predictions = model.predict(X).flatten()
        
        # Create response
        response = {
            'predictions': predictions.tolist(),
            'countries': data['adm_0_name'].tolist()
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Remove temporary file
        os.unlink(tmp_path)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

## Streamlit Dashboard

Create an interactive dashboard using Streamlit in `dashboard/app.py`:

```python
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Define API URL
API_URL = "http://localhost:5000/predict"

st.set_page_config(
    page_title="Dengue Prediction Dashboard",
    page_icon="🦟",
    layout="wide"
)

st.title("Dengue Incidence Prediction Dashboard")
st.write("Upload climate data to predict dengue incidence across multiple countries.")

# Sidebar
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Make predictions
    st.subheader("Making Predictions...")
    
    # Prepare file for API call
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    files = {'file': (uploaded_file.name, csv_buffer.getvalue(), 'text/csv')}
    
    with st.spinner("Predicting..."):
        response = requests.post(API_URL, files=files)
    
    if response.status_code == 200:
        # Get prediction results
        result = response.json()
        predictions = result['predictions']
        countries = result['countries']
        
        # Create a results dataframe
        results_df = pd.DataFrame({
            'Country': countries,
            'Predicted Dengue Incidence': predictions
        })
        
        # Group by country
        country_results = results_df.groupby('Country')['Predicted Dengue Incidence'].mean().reset_index()
        
        # Visualization
        st.subheader("Prediction Results")
        
        # Bar chart of predictions by country
        fig1 = px.bar(
            country_results, 
            x='Country', 
            y='Predicted Dengue Incidence',
            title='Average Predicted Dengue Incidence by Country',
            color='Predicted Dengue Incidence',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Time series visualization (if time data is available)
        if 'Year' in df.columns and 'Month' in df.columns:
            df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
            results_df['Date'] = df['Date'].values
            
            # Group by country and date
            time_results = results_df.groupby(['Country', 'Date'])['Predicted Dengue Incidence'].mean().reset_index()
            
            # Time series plot
            fig2 = px.line(
                time_results, 
                x='Date', 
                y='Predicted Dengue Incidence', 
                color='Country',
                title='Predicted Dengue Incidence Over Time'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Display results table
        st.subheader("Detailed Results")
        st.dataframe(results_df)
        
        # Download link for results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="dengue_predictions.csv",
            mime="text/csv"
        )
    else:
        st.error(f"Error making predictions: {response.text}")
else:
    # Show placeholder content
    st.info("Please upload a CSV file to make predictions.")
    
    # Example visualization
    st.subheader("Example Visualization")
    example_countries = ['Brazil', 'India', 'Thailand', 'Vietnam', 'Malaysia']
    example_values = np.random.rand(5) * 100
    
    fig = px.bar(
        x=example_countries, 
        y=example_values,
        title='Example: Dengue Incidence by Country',
        labels={'x': 'Country', 'y': 'Dengue Incidence per 100,000'},
        color=example_values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
```

## CI/CD Pipeline

### GitHub Actions

Create a GitHub Actions workflow in `.github/workflows/ci-cd.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - name: Test with pytest
      run: |
        pytest

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Build Docker image
      run: |
        docker build -t dengue-prediction-api:latest .
    - name: Login to GitHub Container Registry
      uses: docker/login-action@v1
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    - name: Push Docker image
      run: |
        docker tag dengue-prediction-api:latest ghcr.io/${{ github.repository }}/dengue-prediction-api:latest
        docker push ghcr.io/${{ github.repository }}/dengue-prediction-api:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SSH_HOST }}
        username: ${{ secrets.SSH_USERNAME }}
        key: ${{ secrets.SSH_PRIVATE_KEY }}
        script: |
          cd /opt/dengue-prediction
          docker-compose pull
          docker-compose up -d
```

## Monitoring and Maintenance

### Model Monitoring with MLflow

1. **Set up MLflow server**:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0
   ```

2. **Track model performance**:
   ```python
   import mlflow
   import mlflow.tensorflow
   
   mlflow.set_tracking_uri("http://localhost:5000")
   mlflow.set_experiment("dengue_prediction")
   
   with mlflow.start_run():
       mlflow.log_param("model_type", "hybrid")
       mlflow.log_param("lstm_units", 64)
       
       # Train model
       # ...
       
       # Log metrics
       mlflow.log_metric("test_smape", test_smape)
       mlflow.log_metric("test_mse", test_mse)
       
       # Log model
       mlflow.tensorflow.log_model(model, "model")
   ```

### Automated Retraining

Create a script for automated retraining in `maintenance/retrain.py`:

```python
import os
import argparse
import pandas as pd
from datetime import datetime
import mlflow
import mlflow.tensorflow
from training.train import train_model
from evaluation.metrics import evaluate_model

def retrain_model(data_path, model_type):
    """Retrain model with new data and compare with previous version."""
    # Set up MLflow
    mlflow.set_experiment("dengue_prediction_retraining")
    
    # Load new data
    data = pd.read_csv(data_path)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Train model
        model, metrics = train_model(data, model_type)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("timestamp", datetime.now().isoformat())
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Save model
        mlflow.tensorflow.log_model(model, "model")
        
        # Compare with production model
        try:
            production_model = mlflow.tensorflow.load_model("models:/dengue_prediction/production")
            comparison_metrics = compare_models(production_model, model, data)
            
            for metric_name, metric_value in comparison_metrics.items():
                mlflow.log_metric(f"comparison_{metric_name}", metric_value)
            
            # If new model is better, promote to production
            if comparison_metrics["smape_improvement"] > 0.05:  # 5% improvement threshold
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name="dengue_prediction",
                    version=run.info.run_id,
                    description="New production model"
                )
                print("New model promoted to production!")
            else:
                print("New model did not meet improvement threshold.")
        
        except Exception as e:
            print(f"Error comparing models: {e}")
            # If no production model exists, promote this one
            client = mlflow.tracking.MlflowClient()
            client.create_registered_model("dengue_prediction")
            client.create_model_version(
                name="dengue_prediction",
                source=f"runs:/{run.info.run_id}/model",
                run_id=run.info.run_id
            )
            print("First model registered as production model!")

def compare_models(model1, model2, data):
    """Compare two models on the same dataset."""
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Evaluate both models
    metrics1 = evaluate_model(model1, X, y)
    metrics2 = evaluate_model(model2, X, y)
    
    # Calculate improvements
    improvements = {}
    for metric in metrics1:
        if metric in metrics2:
            # For metrics where lower is better (MSE, RMSE, SMAPE)
            if metric in ["mse", "rmse", "smape"]:
                improvements[f"{metric}_improvement"] = (metrics1[metric] - metrics2[metric]) / metrics1[metric]
            # For metrics where higher is better (R²)
            else:
                improvements[f"{metric}_improvement"] = (metrics2[metric] - metrics1[metric]) / abs(metrics1[metric])
    
    return improvements

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain dengue prediction model")
    parser.add_argument("--data", required=True, help="Path to new data CSV file")
    parser.add_argument("--model", choices=["simple_lstm", "hybrid", "gan"], default="hybrid", help="Model type")
    args = parser.parse_args()
    
    retrain_model(args.data, args.model)
```

### Automated Testing

Create automated tests in `tests/test_model.py`:

```python
import unittest
import numpy as np
import tensorflow as tf
from models.simple_lstm import build_simple_lstm_model
from models.hybrid_model import build_hybrid_model

class TestModels(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.input_shapes = [10, 4, 20]  # [features_dim, seq_length, vector_dim]
        self.batch_size = 16
        
        # Generate dummy input tensors
        self.features = np.random.rand(self.batch_size, self.input_shapes[0])
        self.vectors = np.random.rand(self.batch_size, self.input_shapes[1], self.input_shapes[2])
        
        # Generate dummy targets
        self.targets = np.random.rand(self.batch_size, 1)
    
    def test_simple_lstm_model(self):
        # Build model
        model = build_simple_lstm_model(self.input_shapes)
        
        # Check output shape
        output = model([self.features, self.vectors])
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check trainability
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(
            [self.features, self.vectors], 
            self.targets, 
            epochs=2, 
            batch_size=8,
            verbose=0
        )
        
        # Check loss decreases
        self.assertTrue(history.history['loss'][0] >= history.history['loss'][1])
    
    def test_hybrid_model(self):
        # Build model
        model = build_hybrid_model(self.input_shapes)
        
        # Check output shape
        output = model([self.features, self.vectors])
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check trainability
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(
            [self.features, self.vectors], 
            self.targets, 
            epochs=2, 
            batch_size=8,
            verbose=0
        )
        
        # Check loss decreases
        self.assertTrue(history.history['loss'][0] >= history.history['loss'][1])

if __name__ == '__main__':
    unittest.main()
```
