

### SMAPE Analysis for Insights

```python
def evaluate_model_with_smape_analysis(model, X_test, y_test, countries):
    """Evaluate model with detailed SMAPE analysis."""
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Ensure predictions are flattened
    y_pred = y_pred.flatten()
    
    # Calculate SMAPE for each data point
    def calculate_smape_scores(y_true, y_pred):
        return 100 * (2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    
    smape_scores = calculate_smape_scores(y_test, y_pred)
    
    # Distribution of SMAPE scores
    plt.figure(figsize=(10, 6))
    sns.histplot(smape_scores, kde=True, bins=50)
    plt.title("Distribution of SMAPE Scores")
    plt.xlabel("SMAPE (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    
    # Identify high SMAPE predictions (outliers)
    high_smape_threshold = 50  # Consider SMAPE > 50% as high
    high_smape_indices = np.where(smape_scores > high_smape_threshold)[0]
    
    print(f"Number of high SMAPE predictions (>{high_smape_threshold}%): {len(high_smape_indices)}")
    
    # Print details for high SMAPE predictions
    if len(high_smape_indices) > 0:
        print("\nHigh SMAPE Predictions (Outliers):")
        print(f"{'Index':<10}{'Actual':<15}{'Predicted':<15}{'SMAPE (%)':<15}")
        print("-" * 55)
        
        for idx in high_smape_indices[:10]:  # Limit to first 10 for brevity
            print(f"{idx:<10}{y_test[idx]:<15.4f}{y_pred[idx]:<15.4f}{smape_scores[idx]:<15.2f}")
        
        if len(high_smape_indices) > 10:
            print(f"... and {len(high_smape_indices) - 10} more.")
    
    # Country-wise SMAPE analysis
    countries_test = countries[-len(y_test):]
    unique_countries = np.unique(countries_test)
    
    print("\nCountry-wise SMAPE Analysis:")
    print(f"{'Country':<15}{'Avg SMAPE (%)':<15}{'Min SMAPE (%)':<15}{'Max SMAPE (%)':<15}{'Count':<10}")
    print("-" * 70)
    
    for country in unique_countries:
        country_indices = (countries_test == country)
        
        if np.any(country_indices):
            country_smape = smape_scores[country_indices]
            avg_smape = np.mean(country_smape)
            min_smape = np.min(country_smape)
            max_smape = np.max(country_smape)
            count = np.sum(country_indices)
            
            print(f"{country:<15}{avg_smape:<15.2f}{min_smape:<15.2f}{max_smape:<15.2f}{count:<10}")
    
    # Relationship between actual values and SMAPE
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, smape_scores, alpha=0.5)
    plt.title("SMAPE vs Actual Values")
    plt.xlabel("Actual Dengue Incidence")
    plt.ylabel("SMAPE (%)")
    plt.grid(True)
    plt.show()
    
    return smape_scores

# Evaluate the model with SMAPE analysis
smape_scores = evaluate_model_with_smape_analysis(final_model, X_test, y_test, test_data['adm_0_name'])
```

### Animated Training Visualization

```python
from matplotlib.animation import FuncAnimation

def animate_training(history, save_path=None):
    """Create an animated visualization of the training process."""
    
    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, len(history.history['loss']))
    ax.set_ylim(0, max(max(history.history['loss']), max(history.history['val_loss'])) * 1.1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Over Epochs")
    ax.grid(True)
    
    # Plot lines for training and validation loss
    line_train, = ax.plot([], [], label='Training Loss', color='blue')
    line_val, = ax.plot([], [], label='Validation Loss', color='orange')
    epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    ax.legend()
    
    # Animation update function
    def update(frame):
        ax.set_ylim(0, max(history.history['loss'][:frame + 1] + history.history['val_loss'][:frame + 1]) * 1.1)
        line_train.set_data(range(frame + 1), history.history['loss'][:frame + 1])
        line_val.set_data(range(frame + 1), history.history['val_loss'][:frame + 1])
        epoch_text.set_text(f'Epoch: {frame + 1}')
        return line_train, line_val, epoch_text
    
    # Add interactivity for play/pause
    is_paused = [False]
    
    def on_click(event):
        is_paused[0] = not is_paused[0]
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Custom update function to handle play/pause
    def update_wrapper(frame):
        if not is_paused[0]:
            return update(frame)
        return line_train, line_val, epoch_text
    
    # Create the animation
    ani = FuncAnimation(fig, update_wrapper, frames=len(history.history['loss']), blit=True, repeat=False)
    
    # Save the animation if a path is provided
    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=2)  # Requires ffmpeg to be installed
    
    # Show the animation
    plt.show()
    
    return ani

# Create animated visualization
animation = animate_training(history, save_path="training_animation.mp4")
```

## Complete Pipeline Example

Here is a complete example putting everything together from data loading to model evaluation:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.stattools import durbin_watson
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, MultiHeadAttention, Concatenate
from tensorflow.keras.layers import Embedding, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from bayes_opt import BayesianOptimization
import mlflow

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define file paths and constants
DATA_PATH = "preprocessing/data/merged_monthly_dengue_climate_padded_data.csv"
MAX_SEQUENCE_LENGTH = 20
TRAIN_SPLIT_RATIO = 0.6
VALID_SPLIT_RATIO = 0.2

# Define SMAPE loss function
def smape_loss(y_true, y_pred):
    epsilon = K.epsilon()
    denominator = K.abs(y_true) + K.abs(y_pred) + epsilon
    return K.mean(2 * K.abs(y_pred - y_true) / denominator, axis=-1)

# Calculate SMAPE for numpy arrays
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# 1. Load and prepare data
print("1. Loading and preparing data...")
data = pd.read_csv(DATA_PATH)
data = data[data['adm_0_name'] != 'TAIWAN']

# Define feature columns
feature_columns = ['dengue_incidence_per_lakh', 'EN.POP.DNST', 'SP.URB.TOTL.IN.ZS']
vector_column = 'conditional_low_dim_padded'

# 2. Create lagged features
print("2. Creating lagged features...")
def create_lagged_features(data, feature_columns, vector_column, lags=3):
    for col in feature_columns:
        for lag in range(1, lags + 1):
            data[f"{col}_lag{lag}"] = data[col].shift(lag)
    
    for lag in range(1, lags + 1):
        data[f"{vector_column}_lag{lag}"] = data[vector_column].shift(lag)
    
    data = data.dropna().reset_index(drop=True)
    return data

data = create_lagged_features(data, feature_columns, vector_column, lags=3)

# 3. Split data by country
print("3. Splitting data by country...")
def split_data_country_wise(data, train_ratio, valid_ratio):
    country_groups = data.groupby('adm_0_name')
    train_data, valid_data, test_data = [], [], []
    
    for country, group in country_groups:
        group = group.sort_values(by=['Year', 'Month']).reset_index(drop=True)
        train_size = int(len(group) * train_ratio)
        valid_size = int(len(group) * valid_ratio)
        
        train_data.append(group.iloc[:train_size])
        valid_data.append(group.iloc[train_size:train_size + valid_size])
        test_data.append(group.iloc[train_size + valid_size:])
    
    train_data = pd.concat(train_data, ignore_index=True)
    valid_data = pd.concat(valid_data, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)
    
    return train_data, valid_data, test_data

train_data, valid_data, test_data = split_data_country_wise(data, TRAIN_SPLIT_RATIO, VALID_SPLIT_RATIO)

# 4. Preprocess data
print("4. Preprocessing data...")
def preprocess_after_split(train_data, valid_data, test_data, feature_columns, vector_column, max_sequence_length):
    # Initialize scalers and encoders
    scaler = MinMaxScaler()
    country_encoder = LabelEncoder()
    
    # Process training data
    train_data['country_encoded'] = country_encoder.fit_transform(train_data['adm_0_name'])
    scaled_train_features = scaler.fit_transform(train_data[feature_columns])
    
    # Process validation and test data
    valid_data['country_encoded'] = country_encoder.transform(valid_data['adm_0_name'])
    test_data['country_encoded'] = country_encoder.transform(test_data['adm_0_name'])
    
    scaled_valid_features = scaler.transform(valid_data[feature_columns])
    scaled_test_features = scaler.transform(test_data[feature_columns])
    
    # Combine features with country encoding
    def combine_features(scaled_features, country_encoding):
        country_encoding = country_encoding.reshape(-1, 1)
        return np.hstack([scaled_features, country_encoding])
    
    X_train_combined = combine_features(scaled_train_features, train_data['country_encoded'].values)
    X_valid_combined = combine_features(scaled_valid_features, valid_data['country_encoded'].values)
    X_test_combined = combine_features(scaled_test_features, test_data['country_encoded'].values)
    
    # Process vector data
    def pad_transformer_vectors(data, vector_column, lags=3, max_length=20):
        all_vectors = []
        for lag in range(lags + 1):
            col_name = f"{vector_column}_lag{lag}" if lag > 0 else vector_column
            vectors = data[col_name].apply(lambda x: np.fromstring(x, sep=';')).tolist()
            
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded_vectors = pad_sequences(vectors, maxlen=max_length, padding='post', dtype='float32')
            all_vectors.append(padded_vectors)
        
        return np.stack(all_vectors, axis=1)
    
    X_train_vectors = pad_transformer_vectors(train_data, vector_column)
    X_valid_vectors = pad_transformer_vectors(valid_data, vector_column)
    X_test_vectors = pad_transformer_vectors(test_data, vector_column)
    
    # Create final feature sets
    X_train = [X_train_combined, X_train_vectors]
    X_valid = [X_valid_combined, X_valid_vectors]
    X_test = [X_test_combined, X_test_vectors]
    
    # Target variables
    y_train = scaled_train_features[:, 0]
    y_valid = scaled_valid_features[:, 0]
    y_test = scaled_test_features[:, 0]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, country_encoder

X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, country_encoder = preprocess_after_split(
    train_data, valid_data, test_data, feature_columns, vector_column, MAX_SEQUENCE_LENGTH
)

# Define input shapes
input_shapes = [X_train[0].shape[1], X_train[1].shape[1], X_train[1].shape[2]]

# 5. Define model
print("5. Defining hybrid model...")
def build_hybrid_model(input_shapes, lstm_units=64, gru_units=64, attention_heads=4,
                      dropout_rate=0.3, l2_reg=0.001):
    """Build hybrid LSTM-GRU-Attention model."""
    
    # Input layers
    input_features = Input(shape=(input_shapes[0],))
    input_vectors = Input(shape=(input_shapes[1], input_shapes[2]))
    
    # LSTM layer
    lstm_out = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg))(input_vectors)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    # GRU layer
    gru_out = GRU(gru_units, return_sequences=True, kernel_regularizer=l2(l2_reg))(lstm_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out = Dropout(dropout_rate)(gru_out)
    
    # Multi-head attention
    attention_out = MultiHeadAttention(
        num_heads=attention_heads, 
        key_dim=gru_units
    )(gru_out, gru_out)
    attention_out = BatchNormalization()(attention_out)
    attention_out = Dropout(dropout_rate)(attention_out)
    
    # Flatten attention output
    attention_flat = Flatten()(attention_out)
    
    # Concatenate with features
    combined = Concatenate()([input_features, attention_flat])
    combined = BatchNormalization()(combined)
    
    # Dense layers
    fc_1 = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(combined)
    fc_1 = Dropout(dropout_rate)(fc_1)
    fc_1 = BatchNormalization()(fc_1)
    
    fc_2 = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(fc_1)
    fc_2 = Dropout(dropout_rate)(fc_2)
    fc_2 = BatchNormalization()(fc_2)
    
    # Output layer
    output = Dense(1, activation='linear')(fc_2)
    
    # Create model
    model = Model(inputs=[input_features, input_vectors], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=smape_loss,
        metrics=[smape_loss]
    )
    
    return model

# 6. Start MLflow tracking
print("6. Starting MLflow tracking...")
mlflow.start_run(run_name="hybrid_lstm_gru_attention")

# 7. Train model with early stopping
print("7. Training model...")
model = build_hybrid_model(input_shapes)

# Log model architecture
model.summary()
mlflow.log_param("model_type", "hybrid_lstm_gru_attention")
mlflow.log_param("lstm_units", 64)
mlflow.log_param("gru_units", 64)
mlflow.log_param("attention_heads", 4)
mlflow.log_param("dropout_rate", 0.3)
mlflow.log_param("max_sequence_length", MAX_SEQUENCE_LENGTH)

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,  # We'll use early stopping
    batch_size=32,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

# 8. Evaluate model
print("8. Evaluating model...")
# Basic evaluation
loss, smape_val = model.evaluate(X_test, y_test)
print(f"Test SMAPE Loss: {smape_val:.4f}")

# Make predictions
y_pred = model.predict(X_test).flatten()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
smape_val = smape(y_test, y_pred)

# Log metrics to MLflow
mlflow.log_metric("test_smape", smape_val)
mlflow.log_metric("test_mse", mse)
mlflow.log_metric("test_rmse", rmse)
mlflow.log_metric("test_r2", r2)
mlflow.log_metric("epochs_trained", len(history.history['loss']))

# 9. Visualize results
print("9. Visualizing results...")
# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (SMAPE)')
plt.legend()
plt.grid(True)
plt.savefig("loss_curves.png")
plt.show()

# Save loss figure for MLflow
mlflow.log_artifact("loss_curves.png")

# Generate residual plots
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residual Histogram')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig("residual_histogram.png")
plt.show()

# Save residual histogram for MLflow
mlflow.log_artifact("residual_histogram.png")

# 10. End MLflow run
print("10. Ending MLflow run...")
mlflow.end_run()

print("Process completed successfully!")
```
# Model Implementation Code Examples

This document provides practical code examples for implementing, training, and evaluating the dengue prediction models.

## Table of Contents
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Model Implementation](#model-implementation)
- [Training Process](#training-process)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Complete Pipeline Example](#complete-pipeline-example)

## Data Loading and Preprocessing

### Loading and Preparing Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the data
data_path = "preprocessing/data/merged_monthly_dengue_climate_padded_data.csv"
data = pd.read_csv(data_path)

# Filter out Taiwan (if needed)
data = data[data['adm_0_name'] != 'TAIWAN']

# Define feature columns
feature_columns = ['dengue_incidence_per_lakh', 'EN.POP.DNST', 'SP.URB.TOTL.IN.ZS']
vector_column = 'conditional_low_dim_padded'

# Check for missing values
print("Missing values in data:")
print(data[feature_columns].isna().sum())

# Create lagged features
def create_lagged_features(data, feature_columns, vector_column, lags=3):
    # Create lagged features for scalar columns
    for col in feature_columns:
        for lag in range(1, lags + 1):
            data[f"{col}_lag{lag}"] = data[col].shift(lag)
    
    # Create lagged features for transformer vector column
    for lag in range(1, lags + 1):
        data[f"{vector_column}_lag{lag}"] = data[vector_column].shift(lag)

    # Drop rows with NaN values due to lagging
    data = data.dropna().reset_index(drop=True)
    return data

data = create_lagged_features(data, feature_columns, vector_column, lags=3)

# Split data by country
def split_data_country_wise(data, train_ratio=0.6, valid_ratio=0.2):
    country_groups = data.groupby('adm_0_name')
    train_data, valid_data, test_data = [], [], []

    for country, group in country_groups:
        # Sort by time
        group = group.sort_values(by=['Year', 'Month']).reset_index(drop=True)
        
        # Calculate split sizes
        train_size = int(len(group) * train_ratio)
        valid_size = int(len(group) * valid_ratio)
        
        # Split data
        train_data.append(group.iloc[:train_size])
        valid_data.append(group.iloc[train_size:train_size + valid_size])
        test_data.append(group.iloc[train_size + valid_size:])
    
    # Concatenate data
    train_data = pd.concat(train_data, ignore_index=True)
    valid_data = pd.concat(valid_data, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)
    
    return train_data, valid_data, test_data

train_data, valid_data, test_data = split_data_country_wise(data)

# Preprocess data
def preprocess_split_data(train_data, valid_data, test_data, feature_columns, vector_column, max_sequence_length=20):
    # Initialize scalers and encoders
    scaler = MinMaxScaler()
    country_encoder = LabelEncoder()
    
    # Fit on training data
    train_data['country_encoded'] = country_encoder.fit_transform(train_data['adm_0_name'])
    scaled_train_features = scaler.fit_transform(train_data[feature_columns])
    
    # Transform validation and test data
    valid_data['country_encoded'] = country_encoder.transform(valid_data['adm_0_name'])
    test_data['country_encoded'] = country_encoder.transform(test_data['adm_0_name'])
    
    scaled_valid_features = scaler.transform(valid_data[feature_columns])
    scaled_test_features = scaler.transform(test_data[feature_columns])
    
    # Combine features and country encoding
    def combine_features(scaled_features, country_encoding):
        country_encoding = country_encoding.reshape(-1, 1)
        return np.hstack([scaled_features, country_encoding])
    
    X_train_combined = combine_features(scaled_train_features, train_data['country_encoded'].values)
    X_valid_combined = combine_features(scaled_valid_features, valid_data['country_encoded'].values)
    X_test_combined = combine_features(scaled_test_features, test_data['country_encoded'].values)
    
    # Process vector data
    def pad_transformer_vectors(data, vector_column, lags=3, max_length=20):
        all_vectors = []
        for lag in range(lags + 1):
            col_name = f"{vector_column}_lag{lag}" if lag > 0 else vector_column
            vectors = data[col_name].apply(lambda x: np.fromstring(x, sep=';')).tolist()
            
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded_vectors = pad_sequences(vectors, maxlen=max_length, padding='post', dtype='float32')
            all_vectors.append(padded_vectors)
        
        return np.stack(all_vectors, axis=1)
    
    X_train_vectors = pad_transformer_vectors(train_data, vector_column)
    X_valid_vectors = pad_transformer_vectors(valid_data, vector_column)
    X_test_vectors = pad_transformer_vectors(test_data, vector_column)
    
    # Create final feature sets
    X_train = [X_train_combined, X_train_vectors]
    X_valid = [X_valid_combined, X_valid_vectors]
    X_test = [X_test_combined, X_test_vectors]
    
    # Target variables
    y_train = scaled_train_features[:, 0]  # dengue_incidence_per_lakh
    y_valid = scaled_valid_features[:, 0]
    y_test = scaled_test_features[:, 0]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, country_encoder

X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, country_encoder = preprocess_split_data(
    train_data, valid_data, test_data, feature_columns, vector_column
)

print(f"X_train shapes: {X_train[0].shape}, {X_train[1].shape}")
print(f"y_train shape: {y_train.shape}")
```

## Model Implementation

### Simple LSTM Model

```python
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Custom SMAPE loss function
def smape_loss(y_true, y_pred):
    epsilon = K.epsilon()
    denominator = K.abs(y_true) + K.abs(y_pred) + epsilon
    return K.mean(2 * K.abs(y_pred - y_true) / denominator, axis=-1)

def build_simple_lstm_model(input_shapes, lstm_units=64, dense_units=1, dropout_rate=0.3, l2_reg=0.001):
    """Build a simple LSTM model for sequence prediction."""
    
    # Input layers
    input_features = Input(shape=(input_shapes[0],))
    input_vectors = Input(shape=(input_shapes[1], input_shapes[2]))
    
    # LSTM layer
    lstm_out = LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(l2_reg))(input_vectors)
    lstm_out = BatchNormalization()(lstm_out)
    
    # Concatenate with non-sequential features
    combined = Concatenate()([input_features, lstm_out])
    combined = BatchNormalization()(combined)
    
    # Dense layers with dropout
    fc_1 = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(combined)
    fc_1 = Dropout(dropout_rate)(fc_1)
    fc_1 = BatchNormalization()(fc_1)
    
    # Output layer
    output = Dense(dense_units, activation='linear')(fc_1)
    
    # Create model
    model = Model(inputs=[input_features, input_vectors], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=smape_loss,
        metrics=[smape_loss]
    )
    
    return model

# Define input shapes based on our preprocessed data
input_shapes = [X_train[0].shape[1], X_train[1].shape[1], X_train[1].shape[2]]

# Build the model
lstm_model = build_simple_lstm_model(input_shapes)

# Print model summary
lstm_model.summary()
```

### Hybrid LSTM-GRU-Attention Model

```python
from tensorflow.keras.layers import GRU, MultiHeadAttention, Flatten, TimeDistributed

def build_hybrid_model(input_shapes, lstm_units=64, gru_units=64, attention_heads=4,
                      embedding_dim=32, dense_units=1, dropout_rate=0.3, l2_reg=0.001):
    """Build a hybrid LSTM-GRU-Attention model."""
    
    # Input layers
    input_features = Input(shape=(input_shapes[0],))
    input_vectors = Input(shape=(input_shapes[1], input_shapes[2]))
    
    # Embedding
    embedding = TimeDistributed(Dense(embedding_dim))(input_vectors)
    
    # LSTM layer
    lstm_out = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg))(embedding)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    # GRU layer
    gru_out = GRU(gru_units, return_sequences=True, kernel_regularizer=l2(l2_reg))(lstm_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out = Dropout(dropout_rate)(gru_out)
    
    # Multi-head attention
    attention_out = MultiHeadAttention(
        num_heads=attention_heads, 
        key_dim=embedding_dim
    )(gru_out, gru_out)
    attention_out = BatchNormalization()(attention_out)
    attention_out = Dropout(dropout_rate)(attention_out)
    
    # Flatten and concatenate
    attention_flat = Flatten()(attention_out)
    combined = Concatenate()([input_features, attention_flat])
    combined = BatchNormalization()(combined)
    
    # Dense layers
    fc_1 = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(combined)
    fc_1 = Dropout(dropout_rate)(fc_1)
    fc_1 = BatchNormalization()(fc_1)
    
    fc_2 = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(fc_1)
    fc_2 = Dropout(dropout_rate)(fc_2)
    fc_2 = BatchNormalization()(fc_2)
    
    # Output layer
    output = Dense(dense_units, activation='linear')(fc_2)
    
    # Create model
    model = Model(inputs=[input_features, input_vectors], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=smape_loss,
        metrics=[smape_loss]
    )
    
    return model

# Build hybrid model
hybrid_model = build_hybrid_model(input_shapes)

# Print model summary
hybrid_model.summary()
```

### Conditional GAN Model

```python
from tensorflow.keras.layers import Reshape, RepeatVector, Permute, LeakyReLU

def build_generator(input_shapes, lstm_units=64, gru_units=64, attention_heads=4,
                  output_sequence_length=2, dropout_rate=0.3, l2_reg=0.001):
    """Build generator for conditional GAN."""
    
    # Input layers
    input_features = Input(shape=(input_shapes[0],))
    input_vectors = Input(shape=(input_shapes[1], input_shapes[2]))
    
    # LSTM layer
    lstm_out = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg))(input_vectors)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    # GRU layer
    gru_out = GRU(gru_units, return_sequences=True, kernel_regularizer=l2(l2_reg))(lstm_out)
    gru_out = BatchNormalization()(gru_out)
    gru_out = Dropout(dropout_rate)(gru_out)
    
    # Multi-head attention
    attention_out = MultiHeadAttention(
        num_heads=attention_heads, 
        key_dim=input_vectors.shape[-1]
    )(gru_out, gru_out)
    attention_out = BatchNormalization()(attention_out)
    attention_out = Dropout(dropout_rate)(attention_out)
    
    # Flatten attention output
    attention_flat = Flatten()(attention_out)
    
    # Concatenate with features
    combined = Concatenate()([input_features, attention_flat])
    combined = BatchNormalization()(combined)
    
    # Dense layers
    fc_1 = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(combined)
    fc_1 = Dropout(dropout_rate)(fc_1)
    fc_1 = BatchNormalization()(fc_1)
    
    fc_2 = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(fc_1)
    fc_2 = Dropout(dropout_rate)(fc_2)
    fc_2 = BatchNormalization()(fc_2)
    
    # Reshape for sequence output
    reshaped = Reshape((1, -1))(fc_2)
    repeated = TimeDistributed(RepeatVector(output_sequence_length))(reshaped)
    repeated = Flatten()(repeated)
    
    # Output layer for sequence
    sequence_output = Dense(output_sequence_length, activation='linear')(repeated)
    
    # Create model
    generator = Model(inputs=[input_features, input_vectors], outputs=sequence_output)
    
    return generator

def build_discriminator(input_shapes, lstm_units=64, dropout_rate=0.3, l2_reg=0.001):
    """Build discriminator for conditional GAN."""
    
    # Input layers
    incidence_input = Input(shape=(2,))  # Sequence of length 2
    input_features = Input(shape=(input_shapes[0],))
    input_vectors = Input(shape=(input_shapes[1], input_shapes[2]))
    
    # Reshape incidence input for LSTM
    reshaped_incidence = Reshape((2, 1))(incidence_input)
    
    # LSTM for incidence sequence
    lstm_out = LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(l2_reg))(reshaped_incidence)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    # Flatten vectors
    flattened_vectors = Flatten()(input_vectors)
    
    # Concatenate all inputs
    combined = Concatenate()([lstm_out, input_features, flattened_vectors])
    combined = BatchNormalization()(combined)
    
    # Dense layers
    fc_1 = Dense(128, kernel_regularizer=l2(l2_reg))(combined)
    fc_1 = LeakyReLU(alpha=0.2)(fc_1)
    fc_1 = Dropout(dropout_rate)(fc_1)
    fc_1 = BatchNormalization()(fc_1)
    
    fc_2 = Dense(64, kernel_regularizer=l2(l2_reg))(fc_1)
    fc_2 = LeakyReLU(alpha=0.2)(fc_2)
    fc_2 = Dropout(dropout_rate)(fc_2)
    fc_2 = BatchNormalization()(fc_2)
    
    # Output layer (validity)
    validity = Dense(1, activation='sigmoid')(fc_2)
    
    # Create model
    discriminator = Model(inputs=[incidence_input, input_features, input_vectors], outputs=validity)
    
    # Compile discriminator
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy'
    )
    
    return discriminator

# Create multi-step targets for GAN training
def create_multi_step_targets(df, target_col='dengue_incidence_per_lakh', steps=2):
    """Create multi-step targets for sequence prediction."""
    for i in range(1, steps + 1):
        df[f"{target_col}_t{i}"] = df[target_col].shift(-i)
    
    # Get target columns
    future_cols = [f"{target_col}_t{i}" for i in range(1, steps + 1)]
    
    # Drop rows without future data
    df.dropna(subset=future_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

# Build generator and discriminator for GAN
generator = build_generator(input_shapes, output_sequence_length=2)
discriminator = build_discriminator(input_shapes)

print("Generator Summary:")
generator.summary()
print("\nDiscriminator Summary:")
discriminator.summary()
```

## Training Process

### Training Simple LSTM Model with Bayesian Optimization

```python
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping

def bayesian_optimization(X_train, y_train, X_valid, y_valid, input_shapes):
    """Find optimal hyperparameters using Bayesian optimization."""
    
    def train_model(lstm_units, batch_size, epochs, learning_rate, dropout_rate):
        # Convert parameters to appropriate types
        lstm_units = int(lstm_units)
        batch_size = int(batch_size)
        epochs = int(epochs)
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        val_smapes = []
        
        for train_index, val_index in tscv.split(X_train[0]):
            # Split data for this fold
            X_fold_train = [X_train[0][train_index], X_train[1][train_index]]
            y_fold_train = y_train[train_index]
            X_fold_val = [X_train[0][val_index], X_train[1][val_index]]
            y_fold_val = y_train[val_index]
            
            # Build model
            model = build_simple_lstm_model(
                input_shapes=input_shapes, 
                lstm_units=lstm_units, 
                dropout_rate=dropout_rate
            )
            
            # Compile with specified learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=smape_loss,
                metrics=[smape_loss]
            )
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_fold_train, y_fold_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_fold_val, y_fold_val),
                verbose=0,
                callbacks=[early_stopping]
            )
            
            # Record validation SMAPE
            val_smapes.append(history.history['val_loss'][-1])
        
        # Return negative mean SMAPE (for maximization)
        return -np.mean(val_smapes)
    
    # Define parameter bounds
    pbounds = {
        'lstm_units': (32, 256),
        'batch_size': (16, 64),
        'epochs': (30, 100),
        'learning_rate': (1e-5, 5e-4),
        'dropout_rate': (0.05, 0.2)
    }
    
    # Create optimizer
    optimizer = BayesianOptimization(
        f=train_model,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    # Run optimization
    optimizer.maximize(init_points=5, n_iter=10)
    
    # Return best parameters
    if 'params' in optimizer.max:
        return optimizer.max['params']
    else:
        return {
            'lstm_units': 64,
            'batch_size': 32,
            'epochs': 20,
            'learning_rate': 1e-3,
            'dropout_rate': 0.3
        }

# Find best hyperparameters
best_params = bayesian_optimization(X_train, y_train, X_valid, y_valid, input_shapes)
print("Best Parameters:", best_params)

# Extract parameters
lstm_units = int(best_params.get('lstm_units', 64))
batch_size = int(best_params.get('batch_size', 32))
epochs = int(best_params.get('epochs', 20))
learning_rate = float(best_params.get('learning_rate', 1e-3))
dropout_rate = float(best_params.get('dropout_rate', 0.3))

# Train final model with best parameters
final_model = build_simple_lstm_model(
    input_shapes=input_shapes,
    lstm_units=lstm_units,
    dropout_rate=dropout_rate
)

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=smape_loss,
    metrics=[smape_loss]
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train model
history = final_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)
```

### Training Conditional GAN Model

```python
def train_gan(generator, discriminator, X_train, y_train_sequences, X_val, y_val,
              epochs=100, batch_size=32, learning_rate=1e-3):
    """Train conditional GAN model."""
    
    # Optimizers
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Discriminator model
    discriminator.trainable = True
    discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')
    
    # Combined model (for training generator)
    input_features = Input(shape=(X_train[0].shape[1],))
    input_vectors = Input(shape=(X_train[1].shape[1], X_train[1].shape[2]))
    
    # Generate sequences
    generated_incidence = generator([input_features, input_vectors])
    
    # For the combined model, don't train discriminator
    discriminator.trainable = False
    
    # Determine validity of generated sequences
    validity = discriminator([generated_incidence, input_features, input_vectors])
    
    # Combined model
    combined = Model([input_features, input_vectors], validity)
    combined.compile(optimizer=g_optimizer, loss='binary_crossentropy')
    
    # Arrays to store losses
    g_losses = []
    d_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Sample a batch of inputs
        idx = np.random.randint(0, X_train[0].shape[0], batch_size)
        features_batch = X_train[0][idx]
        vectors_batch = X_train[1][idx]
        
        # Sample real sequences
        idx_seq = np.random.randint(0, y_train_sequences.shape[0], batch_size)
        real_sequences = y_train_sequences[idx_seq]
        
        # Generate fake sequences
        gen_sequences = generator.predict([features_batch, vectors_batch])
        
        # Create labels
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(
            [real_sequences, features_batch, vectors_batch], valid
        )
        d_loss_fake = discriminator.train_on_batch(
            [gen_sequences, features_batch, vectors_batch], fake
        )
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # Train generator
        g_loss = combined.train_on_batch([features_batch, vectors_batch], valid)
        
        # Store losses
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
    
    # Evaluate generator on validation set
    y_val_pred = generator.predict(X_val)
    val_smape = 100 * np.mean(2 * np.abs(y_val_pred - y_val) / (np.abs(y_val_pred) + np.abs(y_val) + 1e-8))
    print(f"Validation SMAPE: {val_smape:.4f}%")
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return generator, val_smape

# Create sequence targets for GAN
# Note: This requires modifying original data
multi_step_data = create_multi_step_targets(data.copy(), steps=2)
train_data_ms, valid_data_ms, test_data_ms = split_data_country_wise(multi_step_data)

# Preprocess multi-step data
multi_step_cols = ['dengue_incidence_per_lakh_t1', 'dengue_incidence_per_lakh_t2']
X_train_ms, X_valid_ms, X_test_ms, _, _, _, scaler_ms, _ = preprocess_split_data(
    train_data_ms, valid_data_ms, test_data_ms, feature_columns, vector_column
)

# Get sequence targets
y_train_ms = scaler_ms.transform(train_data_ms[multi_step_cols])
y_valid_ms = scaler_ms.transform(valid_data_ms[multi_step_cols])
y_test_ms = scaler_ms.transform(test_data_ms[multi_step_cols])

# Train GAN
trained_generator, val_smape = train_gan(
    generator, discriminator, 
    X_train_ms, y_train_ms, 
    X_valid_ms, y_valid_ms,
    epochs=100, batch_size=32
)
```

## Evaluation and Visualization

### Model Evaluation with Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, countries):
    """Evaluate model with comprehensive metrics and visualizations."""
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Ensure predictions are flattened
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate SMAPE
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    
    smape_value = smape(y_test, y_pred)
    
    # Brier score
    brier_score = np.mean((y_test - y_pred) ** 2)
    
    # Mean bias error
    mbe = np.mean(y_pred - y_test)
    
    # Print metrics
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_value:.4f}%")
    print(f"Brier Score: {brier_score:.6f}")
    print(f"Mean Bias Error (MBE): {mbe:.4f}")
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Residual histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='blue')
    plt.title("Residual Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    
    # Residual vs Fitted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted Values (Predictions)")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.show()
    
    # Q-Q plot for residuals
    plt.figure(figsize=(10, 6))
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.grid(True)
    plt.show()
    
    # Durbin-Watson test
    dw_statistic = durbin_watson(residuals)
    print(f"Durbin-Watson Statistic: {dw_statistic:.4f}")
    
    # Predictions vs Actual by country
    countries_test = countries[-len(y_test):]
    unique_countries = np.unique(countries_test)
    
    plt.figure(figsize=(14, 8))
    for country in unique_countries:
        country_indices = (countries_test == country)
        if np.any(country_indices):
            y_test_country = y_test[country_indices]
            y_pred_country = y_pred[country_indices]
            
            plt.plot(y_test_country, label=f'Actual - {country}')
            plt.plot(y_pred_country, linestyle='--', label=f'Predicted - {country}')
    
    plt.xlabel('Time')
    plt.ylabel('Dengue Incidence per Lakh')
    plt.title('Predictions vs Actuals for Different Countries')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'smape': smape_value,
        'brier_score': brier_score,
        'mbe': mbe,
        'dw_statistic': dw_statistic
    }
