<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Climate Data Analysis with Variational Autoencoders</title>
  <!-- MathJax for rendering LaTeX formulas -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        displayMath: [['$$','$$'], ['\\[','\\]']],
        processEscapes: true
      },
      TeX: {
        equationNumbers: { autoNumber: "AMS" }
      }
    });
  </script>
  <style>
    body {
      font-family: 'Arial', sans-serif;
      line-height: 1.6;
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px;
      color: #333;
    }
    h1, h2, h3, h4 {
      color: #224870;
      margin-top: 25px;
    }
    h1 {
      text-align: center;
      border-bottom: 2px solid #224870;
      padding-bottom: 10px;
    }
    h2 {
      border-bottom: 1px solid #ddd;
      padding-bottom: 5px;
    }
    code {
      background-color: #f5f5f5;
      padding: 2px 4px;
      border-radius: 4px;
    }
    pre {
      background-color: #f5f5f5;
      padding: 15px;
      border-radius: 4px;
      overflow-x: auto;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 20px 0;
    }
    th, td {
      border: 1px solid #ddd;
      padding: 8px;
      text-align: left;
    }
    th {
      background-color: #f2f2f2;
    }
    tr:nth-child(even) {
      background-color: #f9f9f9;
    }
    .math-block {
      overflow-x: auto;
      margin: 20px 0;
    }
    .schematic-diagram {
      background-color: #f8f9fa;
      border: 1px solid #e9ecef;
      border-radius: 4px;
      padding: 15px;
      margin: 20px 0;
      overflow-x: auto;
    }
    .schematic-diagram pre {
      font-family: 'Courier New', monospace;
      font-size: 14px;
      line-height: 1.2;
      white-space: pre;
      margin: 0;
      padding: 0;
      background-color: transparent;
    }
    .architecture-description {
      background-color: #f0f4f8;
      border-left: 4px solid #4a7bab;
      padding: 10px 15px;
      margin: 15px 0;
    }
  </style>
</head>
<body>
  <h1>Climate Data Analysis with Variational Autoencoders</h1>

  <h2>Project Overview</h2>
  <p>This project implements and evaluates two types of Variational Autoencoders (VAEs) for meteorological data analysis in Indonesia:</p>
  <ol>
    <li>A standard VAE</li>
    <li>A Conditional Attention VAE with country and year embeddings</li>
  </ol>
  <p>The models encode high-dimensional meteorological features into a lower-dimensional latent space, enabling more efficient data representation and analysis of climate patterns across different regions and time periods in Indonesia.</p>

  <h2>Data Sources</h2>
  <p>The primary dataset is stored in <code>combined_data.csv</code>, which contains meteorological measurements with the following characteristics:</p>
  <ul>
    <li><strong>Temporal Coverage</strong>: Data spans multiple years, beginning from 1955</li>
    <li><strong>Spatial Coverage</strong>: Multiple regions across Indonesia</li>
    <li><strong>Features</strong>: 28 meteorological measurements including:
      <ul>
        <li>Temperature variables (2t, skt)</li>
        <li>Moisture variables (swvl1-4)</li>
        <li>Pressure variables (msl, sp)</li>
        <li>Precipitation (tp, cp)</li>
        <li>Wind (10si)</li>
        <li>Geographic coordinates (latitude, longitude)</li>
      </ul>
    </li>
  </ul>
  <p>Data is organized by date and region, with each row representing a specific location and time point.</p>

  <h2>Mathematical Model</h2>
  <h3>Mathematical Foundations for Dengue Prediction Models</h3>

  <h4>1. Introduction</h4>
  <p>This document provides the mathematical foundations for the dengue prediction models developed in our project. We leverage climate data encoded through Variational Autoencoders (VAEs) and employ multiple neural network architectures to predict dengue fever outbreaks across multiple countries.</p>

  <h4>2. Variational Autoencoders (VAEs)</h4>

  <h5>2.1 Standard VAE Formulation</h5>
  <p>A Variational Autoencoder is a generative model that learns to encode input data into a latent space representation and then decode it back to reconstruct the original input. The VAE consists of an encoder network that maps input $x$ to parameters of a latent distribution, and a decoder network that maps samples from this distribution back to the input space.</p>

  <p>The encoder produces parameters $\mu$ and $\sigma$ of a Gaussian distribution:</p>
  <div class="math-block">
    $$q_\phi(z|x) = \mathcal{N}(z|\mu_\phi(x), \sigma_\phi^2(x))$$
  </div>

  <p>Where:</p>
  <ul>
    <li>$\phi$ represents the encoder network parameters</li>
    <li>$\mu_\phi(x)$ is the mean vector of the latent distribution</li>
    <li>$\sigma_\phi^2(x)$ is the variance vector of the latent distribution</li>
  </ul>

  <p>For numerical stability, the encoder actually outputs $\log \sigma^2$ rather than $\sigma^2$:</p>
  <div class="math-block">
    $$\mu, \log \sigma^2 = \text{Encoder}_\phi(x)$$
  </div>

  <p>The reparameterization trick is used to enable backpropagation through the random sampling process:</p>
  <div class="math-block">
    $$z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$
  </div>

  <p>The decoder reconstructs the input from the latent representation:</p>
  <div class="math-block">
    $$p_\theta(x|z) = \text{Decoder}_\theta(z)$$
  </div>

  <h5>2.2 Conditional Attention VAE</h5>
  <p>Our Conditional Attention VAE extends the standard VAE with country and year embeddings, positional encoding, and attention mechanisms. The conditional encoder can be mathematically represented as:</p>
  <div class="math-block">
    $$\mu, \log \sigma^2 = \text{Encoder}_\phi(x, c, y)$$
  </div>

  <p>Where:</p>
  <ul>
    <li>$x$ is the input data</li>
    <li>$c$ is the country embedding</li>
    <li>$y$ is the year embedding</li>
  </ul>

  <p>The attention mechanism is defined as:</p>
  <div class="math-block">
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  </div>

  <p>Where:</p>
  <ul>
    <li>$Q$ is the query matrix</li>
    <li>$K$ is the key matrix</li>
    <li>$V$ is the value matrix</li>
    <li>$d_k$ is the dimensionality of the keys</li>
  </ul>

  <p>For spatial attention over meteorological features:</p>
  <div class="math-block">
    $$\text{SpatialAttention}(X) = \text{Attention}(W_Q X, W_K X, W_V X)$$
  </div>

  <p>For temporal attention over time periods:</p>
  <div class="math-block">
    $$\text{TemporalAttention}(X) = \text{Attention}(W_Q X, W_K X, W_V X)$$
  </div>

  <h5>2.3 VAE Loss Function</h5>
  <p>The VAE is trained to minimize a loss function that consists of two components:</p>
  <ol>
    <li>Reconstruction loss: measures how well the decoder can reconstruct the input from the latent representation.</li>
    <li>KL divergence: measures how close the latent distribution is to a standard normal distribution.</li>
  </ol>

  <p>The VAE loss function is:</p>
  <div class="math-block">
    $$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{KL}(q_\phi(z|x) \parallel p(z))$$
  </div>

  <p>Where:</p>
  <ul>
    <li>$p(z) = \mathcal{N}(0, I)$ is the prior distribution</li>
    <li>$D_{KL}$ is the Kullback-Leibler divergence</li>
  </ul>

  <p>For a Gaussian latent distribution, the KL divergence has a closed-form solution:</p>
  <div class="math-block">
    $$D_{KL}(q_\phi(z|x) \parallel p(z)) = -\frac{1}{2} \sum_{j=1}^J \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
  </div>

  <p>Where $J$ is the dimensionality of the latent space.</p>

  <h4>3. Time Series Models</h4>

  <h5>3.1 LSTM Model Formulation</h5>
  <p>Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies in sequential data. The LSTM unit contains a cell state and three gates: input, forget, and output gates.</p>

  <h5>3.2 Hybrid LSTM-GRU-Attention</h5>
  <p>Our hybrid model combines LSTM, GRU (Gated Recurrent Unit), and attention mechanisms. The GRU equations at time step $t$ are:</p>
  <div class="math-block">
    \begin{align}
    z_t &= \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \\
    r_t &= \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \\
    \tilde{h}_t &= \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h) \\
    h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
    \end{align}
  </div>

  <p>Where:</p>
  <ul>
    <li>$z_t$ is the update gate output</li>
    <li>$r_t$ is the reset gate output</li>
    <li>$\tilde{h}_t$ is the candidate hidden state</li>
    <li>$h_t$ is the hidden state</li>
  </ul>

  <p>For the hybrid model, the LSTM outputs are fed into the GRU:</p>
  <div class="math-block">
    \begin{align}
    h_t^{\text{LSTM}} &= \text{LSTM}(x_t, h_{t-1}^{\text{LSTM}}) \\
    h_t^{\text{GRU}} &= \text{GRU}(h_t^{\text{LSTM}}, h_{t-1}^{\text{GRU}})
    \end{align}
  </div>

  <p>Multi-head attention is then applied to the GRU outputs:</p>
  <div class="math-block">
    $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W^O$$
  </div>

  <p>Where:</p>
  <ul>
    <li>$W_i^Q, W_i^K, W_i^V, W^O$ are parameter matrices</li>
  </ul>

  <h5>3.3 Conditional GAN</h5>
  <p>Our Conditional Generative Adversarial Network (CGAN) approach involves a generator $G$ and a discriminator $D$. The generator produces dengue incidence predictions, while the discriminator distinguishes between real and generated sequences.</p>

  <p>The adversarial min-max game can be formulated as:</p>
  <div class="math-block">
    $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x|c)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|c)))]$$
  </div>

  <p>Where:</p>
  <ul>
    <li>$c$ is the condition (climate and other features)</li>
    <li>$G(z|c)$ is the generator output given latent noise $z$ and condition $c$</li>
    <li>$D(x|c)$ is the discriminator's probability that $x$ is real given condition $c$</li>
  </ul>

  <h4>4. Loss Functions and Optimization</h4>

  <h5>4.1 Symmetric Mean Absolute Percentage Error (SMAPE)</h5>
  <p>In TensorFlow, this is implemented as:</p>
  <pre><code>def smape_loss(y_true, y_pred):
    epsilon = K.epsilon()  # Small value to avoid division by zero
    denominator = K.abs(y_true) + K.abs(y_pred) + epsilon
    return K.mean(2 * K.abs(y_pred - y_true) / denominator, axis=-1)
</code></pre>

  <h5>4.2 Bayesian Optimization</h5>
  <p>Bayesian optimization is used for hyperparameter tuning. It models the objective function $f(\theta)$ with a Gaussian Process (GP):</p>
  <div class="math-block">
    $$p(f|\mathcal{D}) = \mathcal{GP}(\mu(\theta), k(\theta, \theta'))$$
  </div>

  <p>Where:</p>
  <ul>
    <li>$\mathcal{D} = \{(\theta_i, f(\theta_i))\}_{i=1}^N$ is the observed data</li>
    <li>$\mu(\theta)$ is the mean function</li>
    <li>$k(\theta, \theta')$ is the kernel function</li>
  </ul>

  <p>The acquisition function guides the search for the next point to evaluate. We use the Expected Improvement (EI) acquisition function:</p>
  <div class="math-block">
    $$\text{EI}(\theta) = \mathbb{E}\max(0, f(\theta) - f(\theta^+))$$
  </div>

  <p>Where $f(\theta^+)$ is the best observed value so far.</p>

  <p>Our search space includes:</p>
  <ul>
    <li>LSTM units: [32, 256]</li>
    <li>Batch size: [16, 64]</li>
    <li>Learning rate: [1e-5, 5e-4]</li>
    <li>Dropout rate: [0.05, 0.2]</li>
  </ul>

  <h4>5. Evaluation Metrics</h4>

  <h5>5.1 Performance Metrics</h5>
  <p>We use several metrics to evaluate model performance:</p>

  <p><strong>Mean Squared Error (MSE):</strong></p>
  <div class="math-block">
    $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
  </div>

  <p><strong>Root Mean Squared Error (RMSE):</strong></p>
  <div class="math-block">
    $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
  </div>

  <p><strong>R-squared (R²):</strong></p>
  <div class="math-block">
    $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
  </div>

  <p><strong>Symmetric Mean Absolute Percentage Error (SMAPE):</strong></p>
  <div class="math-block">
    $$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{2 |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$
  </div>

  <p><strong>Brier Score:</strong></p>
  <div class="math-block">
    $$\text{Brier} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
  </div>

  <p><strong>Mean Bias Error (MBE):</strong></p>
  <div class="math-block">
    $$\text{MBE} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$$
  </div>

  <h5>5.2 Residual Analysis</h5>
  <p>Residual analysis involves examining the differences between observed and predicted values:</p>
  <div class="math-block">
    $$\text{residuals} = y - \hat{y}$$
  </div>

  <p>We assess residuals through:</p>
  <ol>
    <li>Histogram of residuals (should approximate a normal distribution)</li>
    <li>Residuals vs. fitted values plot (should show no pattern)</li>
    <li>Q-Q plot (should follow a straight line if residuals are normally distributed)</li>
  </ol>

  <h4>6. Statistical Tests and Validation</h4>

  <h5>6.1 Durbin-Watson Test</h5>
  <p>The Durbin-Watson test checks for autocorrelation in the residuals:</p>
  <div class="math-block">
    $$\text{DW} = \frac{\sum_{i=2}^n (e_i - e_{i-1})^2}{\sum_{i=1}^n e_i^2}$$
  </div>

  <p>Where $e_i$ are the residuals.</p>

  <p>Interpretation:</p>
  <ul>
    <li>DW ≈ 2: No autocorrelation</li>
    <li>DW < 2: Positive autocorrelation</li>
    <li>DW > 2: Negative autocorrelation</li>
  </ul>

  <h5>6.2 Time Series Cross-Validation</h5>
  <p>For time series data, we use TimeSeriesSplit for cross-validation, which respects the temporal ordering of observations:</p>

  <p>For $k$ splits, the data is divided into $k$ folds:</p>
  <ol>
    <li>Training set: observations [0, n₁)<br>
       Validation set: observations [n₁, n₂)</li>
    <li>Training set: observations [0, n₂)<br>
       Validation set: observations [n₂, n₃)</li>
  </ol>
  <p>...and so on, where n₁ < n₂ < n₃ < ... < n.</p>

  <p>This approach ensures that we never train on future data and validate on past data, which would lead to data leakage and overly optimistic performance estimates.</p>

  <h2>Architecture</h2>

  <h3>Basic VAE</h3>
  
  <div class="architecture-description">
    <p>The standard VAE uses a simple fully-connected architecture:</p>
    <pre>Input(28) → FC(112) → ReLU → 
            ├─→ FC(11) [μ]
            └─→ FC(11) [logvar]
                  ↓
                  z ∼ N(μ, exp(logvar))
                  ↓
           FC(112) → ReLU → FC(28) → Sigmoid → Output(28)</pre>
  </div>

  <h4>Schematic Diagram</h4>
  
  <div class="schematic-diagram">
    <pre>                            ┌───────────────┐
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
                       └───────────────────┘</pre>
  </div>

  <h3>Conditional Attention VAE</h3>
  
  <div class="architecture-description">
    <p>The Conditional Attention VAE extends the basic VAE with country and year embeddings, positional encoding, and attention mechanisms:</p>
    <pre>Input(28) → ProjectionLayer → SpatialAttention & TemporalAttention →
CountryEmbedding(10) → YearEmbedding(10) + PositionalEncoding →
[Concatenate] → FC(112) → LayerNorm → ReLU → Dropout →
                ├─→ FC(11) [μ]
                └─→ FC(11) [logvar]
                      ↓
                      z ∼ N(μ, exp(logvar))
                      ↓
CountryEmbedding(10) → YearEmbedding(10) + PositionalEncoding →
[Concatenate with z] → FC(112) → LayerNorm → ReLU → FC(28) → Sigmoid → Output(28)</pre>
  </div>

  <h4>Schematic Diagram</h4>
  
  <div class="schematic-diagram">
    <pre>            ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
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
│
  <p>This approach ensures that we never train on future data and validate on past data, which would lead to data leakage and overly optimistic performance estimates.</p>

  <h2>Model Parameters</h2>

  <h3>Basic VAE Parameters</h3>
  <table>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>input_dim</td>
        <td>28</td>
        <td>Number of input features</td>
      </tr>
      <tr>
        <td>hidden_dim</td>
        <td>112</td>
        <td>Size of hidden layers (4x input_dim)</td>
      </tr>
      <tr>
        <td>latent_dim</td>
        <td>11</td>
        <td>Dimension of latent space representation</td>
      </tr>
      <tr>
        <td>learning_rate</td>
        <td>1e-3</td>
        <td>Learning rate for Adam optimizer</td>
      </tr>
      <tr>
        <td>epochs</td>
        <td>5000</td>
        <td>Maximum number of training epochs</td>
      </tr>
      <tr>
        <td>patience</td>
        <td>20</td>
        <td>Early stopping patience (epochs)</td>
      </tr>
      <tr>
        <td>min_delta</td>
        <td>1e-4</td>
        <td>Minimum improvement for early stopping</td>
      </tr>
    </tbody>
  </table>

  <h3>Conditional Attention VAE Parameters</h3>
  <table>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>input_dim</td>
        <td>28</td>
        <td>Number of input features</td>
      </tr>
      <tr>
        <td>hidden_dim</td>
        <td>112</td>
        <td>Size of hidden layers (4x input_dim)</td>
      </tr>
      <tr>
        <td>latent_dim</td>
        <td>11</td>
        <td>Dimension of latent space representation</td>
      </tr>
      <tr>
        <td>country_embedding_dim</td>
        <td>10</td>
        <td>Size of country embedding vectors</td>
      </tr>
      <tr>
        <td>year_embedding_dim</td>
        <td>10</td>
        <td>Size of year embedding vectors</td>
      </tr>
      <tr>
        <td>attention_dim</td>
        <td>14</td>
        <td>Dimension for attention mechanism (input_dim/2)</td>
      </tr>
      <tr>
        <td>num_countries</td>
        <td>*</td>
        <td>Number of unique countries in dataset</td>
      </tr>
      <tr>
        <td>num_years</td>
        <td>*</td>
        <td>Number of unique years in dataset</td>
      </tr>
      <tr>
        <td>dropout_rate</td>
        <td>0.3</td>
        <td>Dropout rate for regularization</td>
      </tr>
      <tr>
        <td>learning_rate</td>
        <td>varied</td>
        <td>Different learning rates for different parameter groups:<br>- Embeddings: 1e-3<br>- Attention: 1e-3<br>- Encoder/Decoder: 1e-4</td>
      </tr>
      <tr>
        <td>epochs</td>
        <td>5000</td>
        <td>Maximum number of training epochs</td>
      </tr>
      <tr>
        <td>patience</td>
        <td>20</td>
        <td>Early stopping patience (epochs)</td>
      </tr>
      <tr>
        <td>min_delta</td>
        <td>1e-4</td>
        <td>Minimum improvement for early stopping</td>
      </tr>
    </tbody>
  </table>

  <h2>Evaluation Results</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Test RMSE</th>
        <th>Test R²</th>
        <th>Test SMAPE</th>
        <th>Durbin-Watson</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Simple LSTM</td>
        <td>0.9912</td>
        <td>0.42</td>
        <td>24.37%</td>
        <td>1.86</td>
      </tr>
      <tr>
        <td>Hybrid LSTM-GRU-Attention</td>
        <td>0.8733</td>
        <td>0.53</td>
        <td>19.45%</td>
        <td>1.94</td>
      </tr>
      <tr>
        <td>Conditional GAN</td>
        <td>0.8412</td>
        <td>0.56</td>
        <td>18.72%</td>
        <td>1.97</td>
      </tr>
    </tbody>
  </table>
</body>
</html>
