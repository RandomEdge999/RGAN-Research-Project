from typing import Optional

import tensorflow as tf

"""
RGAN (Recurrent Generative Adversarial Network) Architecture Documentation
=======================================================================

This module implements the RGAN architecture for time series forecasting, consisting of:

1. GENERATOR ARCHITECTURE:
   - Input: Sequential data of shape (L, n_in) where L is sequence length and n_in is number of features
   - LSTM Stack: Configurable number of LSTM layers with dropout regularization
   - Dense Layer: Projects LSTM output to forecast horizon H
   - Reshape: Outputs predictions of shape (H, 1) for H-step ahead forecasting
   
   Purpose: Generates realistic future time series values that fool the discriminator
   
2. DISCRIMINATOR ARCHITECTURE:
   - Input: Concatenated sequences of shape (L + H, 1) - input sequence + generated/predicted sequence
   - LSTM Stack: Configurable number of LSTM layers with dropout regularization  
   - Dense Layer: Single output with sigmoid activation for real/fake classification
   
   Purpose: Distinguishes between real historical patterns and generated predictions
   
3. TRAINING PROCESS:
   - Adversarial Training: Generator tries to fool discriminator, discriminator learns to detect fakes
   - Regularization Loss: MSE between generated and actual values ensures realistic predictions
   - Combined Loss: G_loss = adversarial_loss + λ * regularization_loss
   
4. KEY PARAMETERS:
   - L: Input sequence length (lookback window)
   - H: Forecast horizon (number of future steps to predict)
   - units: Number of LSTM units in each layer
   - num_layers: Depth of LSTM stack
   - dropout: Regularization rate
   - activation: LSTM activation function (default: tanh)
   - recurrent_activation: LSTM recurrent activation (default: sigmoid)
   - lambda_reg: Regularization weight in generator loss
"""

def _apply_lstm_stack(x, units, num_layers, dropout, prefix, activation, recurrent_activation):
    """Builds a stack of LSTM layers with optional dropout between layers."""
    for i in range(num_layers):
        return_sequences = i < (num_layers - 1)
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=return_sequences,
            activation=activation,
            recurrent_activation=recurrent_activation,
            name=f"{prefix}_lstm_{i+1}"
        )(x)
        if dropout > 0 and (return_sequences or i == num_layers - 1):
            x = tf.keras.layers.Dropout(dropout, name=f"{prefix}_dropout_{i+1}")(x)
    return x

def build_generator(
    L: int,
    H: int,
    n_in: int = 1,
    units: int = 64,
    dropout: float = 0.0,
    num_layers: int = 1,
    activation: str = "tanh",
    recurrent_activation: str = "sigmoid",
    dense_activation: Optional[str] = None,
) -> tf.keras.Model:
    """Construct an LSTM generator with configurable depth and activations."""
    x_in = tf.keras.Input(shape=(L, n_in), name="generator_input")
    x = _apply_lstm_stack(x_in, units, num_layers, dropout, "gen", activation, recurrent_activation)
    x = tf.keras.layers.Dense(H, activation=dense_activation, name="gen_dense_out")(x)
    y = tf.keras.layers.Reshape((H, 1), name="gen_reshape_out")(x)
    return tf.keras.Model(x_in, y, name="Generator")

def build_discriminator(
    L: int,
    H: int,
    units: int = 64,
    dropout: float = 0.0,
    num_layers: int = 1,
    activation: str = "tanh",
    recurrent_activation: str = "sigmoid",
) -> tf.keras.Model:
    """Construct an LSTM discriminator with configurable depth and activations."""
    x_in = tf.keras.Input(shape=(L + H, 1), name="discriminator_input")
    x = _apply_lstm_stack(x_in, units, num_layers, dropout, "disc", activation, recurrent_activation)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="disc_dense_out")(x)
    return tf.keras.Model(x_in, out, name="Discriminator")
