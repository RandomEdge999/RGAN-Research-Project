import tensorflow as tf

def build_generator(L: int, H: int, n_in: int = 1, units: int = 64, dropout: float = 0.0) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=(L, n_in))
    x = tf.keras.layers.LSTM(units, return_sequences=False)(x_in)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(H)(x)
    y = tf.keras.layers.Reshape((H,1))(x)
    return tf.keras.Model(x_in, y, name="Generator")

def build_discriminator(L: int, H: int, units: int = 64, dropout: float = 0.0) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=(L + H, 1))
    x = tf.keras.layers.LSTM(units, return_sequences=False)(x_in)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(x_in, out, name="Discriminator")
