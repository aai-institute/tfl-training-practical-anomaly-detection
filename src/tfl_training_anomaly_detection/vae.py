#
# Example implementation of a variational autoencoder
# based on https://keras.io/examples/generative/vae/
#

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Sampling Layer
class Sampling(layers.Layer):
    """
    Sample from Z from N(z_mean, z_log_var)
    """

    def call(self, inputs):
        """
        Implementation of call function
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(.5 * z_log_var) * epsilon


# Encoder network
def build_encoder_minst(latent_dim=2):
    """Build convolutional encoder for the MNIST data set.

    :param letent_dim: output dimension of the encoder
    """
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    return encoder


def build_decoder_mnist(latent_dim=2):
    """
    Build decoder for the MNIST data set.

    :param latent_dim: input dimension of the decoder
    """
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

    return decoder


class VAE(keras.Model):
    """
    Implementation of a variational autoencoder
    """

    def __init__(self, encoder, decoder, **kwargs):
        """
        Initilization. Note: the output dimension of the encoder needs to match the input dimension of the decoder.

        :param encoder: the encoder network
        :param decoder: the decoder network
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        """
        Get metrics for the vae
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        """
        Implements a single training step for the vae.

        :param data: the training batch
        :return: training loss
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction, axis=(1, 2))
                )
            )

            kl_loss = -.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }


def build_contaminated_minst(data, contamination=.03, p_noise=.1):
    """
    Build contaminated MNIST datset. We use 3 different contamination types: noise, occlusion, and transposing.

    :param data: MNIST data set with shape (None, 28, 28, 1) whith values between 0 and 1
    :param contamination: proportion of contamination.
    :return: MNIST data set with random corruptions
    """
    data = data.copy()
    n = len(data)
    n_cont = int(contamination * n)
    selection = np.random.choice(n, n_cont)

    # Transposed images
    transpose = int(n_cont / 3)
    selection_trans = selection[:transpose]
    data[selection_trans] = np.transpose(data[selection_trans], axes=[0, 2, 1, 3])

    # Blacked out image parts
    blackout = int(n_cont * (2 / 3))
    selection_blackout = selection[transpose: blackout]
    n_blackout = len(selection_blackout)
    x = np.random.binomial(n=1, size=n_blackout, p=.5)
    # y = np.random.binomial(n=1, size=n_blackout, p=.5)
    for i, sel in enumerate(selection_blackout):
        xi = x[i]
        data[sel][xi * 14:14 + xi * 14, :] = 0

    # Noisy images
    selection_noisy = selection[blackout:]
    n_noisy = len(selection_noisy)
    noise = np.random.binomial(n=1, p=p_noise, size=(n_noisy, 28, 28, 1))
    data[selection_noisy] = (1 - noise) * data[selection_noisy] + noise * (1 - data[selection_noisy])

    y = np.zeros(data.shape[0])
    for i, sel in enumerate([selection_trans, selection_noisy, selection_blackout]):
        y[sel] = i

    return data, y
