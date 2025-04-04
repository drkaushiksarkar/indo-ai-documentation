# Model API Documentation

This document provides detailed information about the API for implementing, training, and utilizing the dengue prediction models.

## Table of Contents
- [Module Structure](#module-structure)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Model Classes](#model-classes)
  - [Simple LSTM Model](#simple-lstm-model)
  - [Hybrid LSTM-GRU-Attention Model](#hybrid-lstm-gru-attention-model)
  - [Conditional GAN Model](#conditional-gan-model)
- [Training Functions](#training-functions)
- [Evaluation Functions](#evaluation-functions)
- [Visualization Tools](#visualization-tools)
- [MLflow Integration](#mlflow-integration)

## Module Structure

The codebase is organized into the following modules:

```
models/
├── simple_lstm.py                 # Simple LSTM implementation
├── hybrid_model.py                # Hybrid LSTM-GRU-Attention implementation
├── gan_model.py                   # Conditional GAN implementation
└── loss_functions.py              # Custom loss functions including SMAPE

preprocessing/
├── data_loader.py                 # Data loading and initial processing
├── feature_engineering.py         # Creation of lagged features and other engineering
└── sequence_processing.py         # Handling of sequence data and padding

training/
├── optimizer.py                   # Bayesian optimization implementation
├── train.py                       # Main training loop functions
└── callbacks.py                   # Custom callbacks for training

evaluation/
├── metrics.py                     # Custom evaluation metrics
├── residual_analysis.py           # Tools for residual analysis
└── country_evaluation.py          # Country-specific evaluation tools

visualization/
├── loss_plots.py                  # Training and validation loss plot functions
├── forecast_plots.py              # Forecast visualization tools
└── animated_training.py           # Animated visualization of training progress

utils/
├── mlflow_tracking.py             # MLflow integration utilities
├── config.py                      # Configuration management
└── file_handlers.py               # Utilities for file I/O operations
```

## Data Loading and Preprocessing

### `preprocessing.data_loader`

```python
def load_and_prepare_data(filepath):
    """
    Load, clean, and prepare the dengue and climate data.
    
    Args:
        filepath (str): Path to the CSV data file
    
    Returns:
        pandas.DataFrame: Cleaned and prepared dataframe
    """
```

```python
def estimate_missing_values(data):
    """
    Estimate missing values for 2022 based on 2020-2021 trend.
    
    Args:
        data (pandas.DataFrame): Input dataframe with missing values
    
    Returns:
        pandas.DataFrame: Dataframe with estimated values
    """
```

### `preprocessing.feature_engineering`

```python
def create_lagged_features(data, feature_columns, vector_column=None, lags=3):
    """
    Create lagged versions of features for time series modeling.
    
    Args:
        data (pandas.DataFrame): Input dataframe
        feature_columns (list): List of column names to create lags for
        vector_column (str, optional): Name of vector column to create lags for
        lags (int): Number of lagged versions to create
    
    Returns:
        pandas.DataFrame: Dataframe with additional lagged features
    """
```

```python
def split_data_country_wise(data, train_split_ratio, valid_split_ratio):
    """
    Split data into training, validation, and test sets by country.
    
    Args:
        data (pandas.DataFrame): Input dataframe
        train_split_ratio (float): Proportion of data for training
        valid_split_ratio (float): Proportion of data for validation
    
    Returns:
        tuple: (train_data, valid_data, test_data) dataframes
    """
```

### `preprocessing.sequence_processing`

```python
def pad_transformer_vectors(data, vector_column, lags=3, max_sequence_length=20):
    """
    Prepare padded sequences for the transformer vector and its lagged features.
    
    Args:
        data (pandas.DataFrame): Input dataframe
        vector_column (str): Name of the vector column
        lags (int): Number of lags to include
        max_sequence_length (int): Maximum length to pad sequences to
    
    Returns:
        numpy.ndarray: Padded vector sequences with shape (samples, lags+1, max_sequence_length)
    """
```

```python
def preprocess_after_split(train_data, valid_data, test_data, feature_columns, vector_column, max_sequence_length):
    """
    Preprocess train, validation, and test data independently after splitting.
    
    Args:
        train_data (pandas.DataFrame): Training dataframe
        valid_data (pandas.DataFrame): Validation dataframe
        test_data (pandas.DataFrame): Test dataframe
        feature_columns (list): List of feature column names
        vector_column (str): Name of the vector column
        max_sequence_length (int): Maximum length to pad sequences to
    
    Returns:
        tuple: Preprocessed X_train, X_valid, X_test, y_train, y_valid, y_test
    """
```

## Model Classes

### Simple LSTM Model

```python
class SimpleLSTM(tf.keras.Model):
    """
    Simple LSTM model for dengue prediction.
    
    Args:
        input_shapes (list): Shapes of input features and vectors
        lstm_units (int): Number of LSTM units
        dense_units (int): Number of output units
        dropout_rate (float): Dropout rate
        l2_reg (float): L2 regularization coefficient
    """
    
    def __init__(self, input_shapes, lstm_units=64, dense_units=1, dropout_rate=0.3, l2_reg=0.001):
        super(SimpleLSTM, self).__init__()
        # Initialize layers
        
    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        
        Args:
            inputs (list): List of [features, vectors]
            training (bool): Whether in training mode
            
        Returns:
            tf.Tensor: Model predictions
        """
```

### Hybrid LSTM-GRU-Attention Model

```python
class HybridModel(tf.keras.Model):
    """
    Hybrid LSTM-GRU-Attention model for dengue prediction.
    
    Args:
        input_shapes (list): Shapes of input features and vectors
        lstm_units (int): Number of LSTM units
        gru_units (int): Number of GRU units
        attention_heads (int): Number of attention heads
        dense_units (int): Number of output units
        dropout_rate (float): Dropout rate
        l2_reg (float): L2 regularization coefficient
    """
    
    def __init__(self, input_shapes, lstm_units=64, gru_units=64, attention_heads=4,
                 dense_units=1, dropout_rate=0.3, l2_reg=0.001):
        super(HybridModel, self).__init__()
        # Initialize layers
        
    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        
        Args:
            inputs (list): List of [features, vectors]
            training (bool): Whether in training mode
            
        Returns:
            tf.Tensor: Model predictions
        """
```

### Conditional GAN Model

```python
class Generator(tf.keras.Model):
    """
    Generator model for conditional GAN.
    
    Args:
        input_shapes (list): Shapes of input features and vectors
        lstm_units (int): Number of LSTM units
        gru_units (int): Number of GRU units
        attention_heads (int): Number of attention heads
        output_sequence_length (int): Length of output sequence to generate
        dropout_rate (float): Dropout rate
        l2_reg (float): L2 regularization coefficient
    """
    
    def __init__(self, input_shapes, lstm_units=64, gru_units=64, attention_heads=4,
                 output_sequence_length=2, dropout_rate=0.3, l2_reg=0.001):
        super(Generator, self).__init__()
        # Initialize layers
        
    def call(self, inputs, training=None):
        """
        Forward pass of the generator.
        
        Args:
            inputs (list): List of [features, vectors]
            training (bool): Whether in training mode
            
        Returns:
            tf.Tensor: Generated sequence predictions
        """
```

```python
class Discriminator(tf.keras.Model):
    """
    Discriminator model for conditional GAN.
    
    Args:
        input_shapes (list): Shapes of input features and vectors
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        l2_reg (float): L2 regularization coefficient
    """
    
    def __init__(self, input_shapes, lstm_units=64, dropout_rate=0.3, l2_reg=0.001):
        super(Discriminator, self).__init__()
        # Initialize layers
        
    def call(self, inputs, training=None):
        """
        Forward pass of the discriminator.
        
        Args:
            inputs (list): List of [sequence, features, vectors]
            training (bool): Whether in training mode
            
        Returns:
            tf.Tensor: Validity prediction (0-1)
        """
```

```python
class ConditionalGAN:
    """
    Conditional GAN for dengue prediction.
    
    Args:
        generator (Generator): Generator model
        discriminator (Discriminator): Discriminator model
        learning_rate (float): Learning rate for optimizers
    """
    
    def __init__(self, generator, discriminator, learning_rate=1e-3):
        # Initialize models and optimizers
        
    def train_step(self, features_batch, vectors_batch, real_sequences):
        """
        Single training step for the GAN.
        
        Args:
            features_batch (tf.Tensor): Batch of feature inputs
            vectors_batch (tf.Tensor): Batch of vector inputs
            real_sequences (tf.Tensor): Batch of real target sequences
            
        Returns:
            tuple: (generator_loss, discriminator_loss)
        """
        
    def train(self, X_train, y_train_sequences, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the GAN model.
        
        Args:
            X_train (list): Training inputs [features, vectors]
            y_train_sequences (tf.Tensor): Training target sequences
            X_val (list): Validation inputs [features, vectors]
            y_val (tf.Tensor): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            tuple: (trained_generator, validation_smape)
        """
```

## Training Functions

### Custom Loss Functions

```python
def smape_loss(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error loss function.
    
    Args:
        y_true (tf.Tensor): True values
        y_pred (tf.Tensor): Predicted values
        
    Returns:
        tf.Tensor: SMAPE loss value
    """
    epsilon = K.epsilon()  # Small value to avoid division by zero
    denominator = K.abs(y_true) + K.abs(y_pred) + epsilon
    return K.mean(2 * K.abs(y_pred - y_true) / denominator, axis=-1)
```

### Bayesian Optimization

```python
def bayesian_optimization(X_train, y_train, X_valid, y_valid, input_shapes):
    """
    Use Bayesian Optimization to find the best hyperparameters.
    
    Args:
        X_train (list): Training inputs [features, vectors]
        y_train (tf.Tensor): Training targets
        X_valid (list): Validation inputs [features, vectors]
        y_valid (tf.Tensor): Validation targets
        input_shapes (list): Shapes of input features and vectors
        
    Returns:
        dict: Optimized hyperparameters
    """
```

### Training Loop

```python
def train_with_cross_validation(model, X_train, y_train, X_valid, y_valid, 
                               epochs=100, batch_size=32, learning_rate=1e-3):
    """
    Train model with time series cross-validation.
    
    Args:
        model (tf.keras.Model): Model to train
        X_train (list): Training inputs [features, vectors]
        y_train (tf.Tensor): Training targets
        X_valid (list): Validation inputs [features, vectors]
        y_valid (tf.Tensor): Validation targets
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        
    Returns:
        tuple: (trained_model, average_validation_metric)
    """
```

## Evaluation Functions

### Metrics Calculation

```python
def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        
    Returns:
        float: SMAPE score (%)
    """
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
```

```python
def brier_score(y_true, y_pred):
    """
    Calculate Brier Score.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        
    Returns:
        float: Brier score
    """
    return np.mean((y_true - y_pred) ** 2)
```

```python
def mean_bias_error(y_true, y_pred):
    """
    Calculate Mean Bias Error.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        
    Returns:
        float: Mean bias error
    """
    return np.mean(y_pred - y_true)
```

### Model Evaluation

```python
def evaluate_model(model, X_test, y_test, countries):
    """
    Evaluate the model on test data and perform extensive validation.
    
    Args:
        model (tf.keras.Model): Model to evaluate
        X_test (list): Test inputs [features, vectors]
        y_test (numpy.ndarray): Test targets
        countries (numpy.ndarray): Country labels for test data
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
```

```python
def evaluate_model_with_smape_analysis(model, X_test, y_test, countries):
    """
    Evaluate the model with detailed SMAPE analysis.
    
    Args:
        model (tf.keras.Model): Model to evaluate
        X_test (list): Test inputs [features, vectors]
        y_test (numpy.ndarray): Test targets
        countries (numpy.ndarray): Country labels for test data
    """
```

### Residual Analysis

```python
def plot_residual_histogram(y_test, y_pred):
    """
    Plot histogram of residuals.
    
    Args:
        y_test (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
    """
```

```python
def plot_residual_vs_fitted(y_pred, residuals):
    """
    Plot residuals versus fitted values.
    
    Args:
        y_pred (numpy.ndarray): Predicted values
        residuals (numpy.ndarray): Residuals (y_test - y_pred)
    """
```

```python
def plot_qq_residuals(residuals):
    """
    Create Q-Q plot for residuals.
    
    Args:
        residuals (numpy.ndarray): Residuals (y_test - y_pred)
    """
```

```python
def calculate_durbin_watson(residuals):
    """
    Calculate Durbin-Watson statistic for residuals.
    
    Args:
        residuals (numpy.ndarray): Residuals (y_test - y_pred)
        
    Returns:
        float: Durbin-Watson statistic
    """
    return durbin_watson(residuals)
```

## Visualization Tools

### Training Visualization

```python
def animate_training(history, save_path=None):
    """
    Create an animated visualization of the training process.
    
    Args:
        history (tf.keras.callbacks.History): Training history
        save_path (str, optional): Path to save animation file
        
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
```

```python
def plot_loss_curves(history):
    """
    Plot training and validation loss curves.
    
    Args:
        history (tf.keras.callbacks.History): Training history
    """
```

### Forecast Visualization

```python
def plot_predictions_by_country(y_test, y_pred, countries):
    """
    Plot predictions versus actuals for each country.
    
    Args:
        y_test (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        countries (numpy.ndarray): Country labels
    """
```

```python
def plot_smape_distribution(y_test, y_pred):
    """
    Plot distribution of SMAPE scores.
    
    Args:
        y_test (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
    """
```

## MLflow Integration

```python
def start_mlflow_run(experiment_name="dengue_prediction"):
    """
    Start a new MLflow run.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        
    Returns:
        mlflow.ActiveRun: Active MLflow run
    """
```

```python
def log_model_parameters(params):
    """
    Log model parameters to MLflow.
    
    Args:
        params (dict): Dictionary of model parameters
    """
```

```python
def log_model_metrics(metrics):
    """
    Log model metrics to MLflow.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
    """
```

```python
def log_model(model, model_name):
    """
    Log model to MLflow.
    
    Args:
        model (tf.keras.Model): Model to log
        model_name (str): Name of the model
    """
```

```python
def end_mlflow_run():
    """
    End the current MLflow run.
    """
```

```python
def load_model_from_mlflow(run_id, model_name):
    """
    Load a model from MLflow.
    
    Args:
        run_id (str): MLflow run ID
        model_name (str): Name of the model to load
        
    Returns:
        tf.keras.Model: Loaded model
    """
```