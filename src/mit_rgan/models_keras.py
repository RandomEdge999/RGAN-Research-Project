from typing import Optional

import tensorflow as tf

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
