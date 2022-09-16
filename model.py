import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, BatchNormalization, InputLayer, 
    ReLU, LeakyReLU, Activation, Conv3DTranspose, 
    Conv3D, Reshape, 
)

import config



def make_generator_model():
    '''
    Create the generator model with structure:
    (100,) -> 512x4x4x4 -> 256x8x8x8 -> 128x16x16x16 -> 64x32x32x32 -> 64x64x64
    '''
    model = Sequential()

    model.add(InputLayer(input_shape=(config.Z_DIM,)))
    model.add(Reshape((1, 1, 1, config.Z_DIM))) 
    assert model.output_shape == (None, 1, 1, 1, config.Z_DIM) # None is the batch size

    model.add(Conv3DTranspose(512, (4, 4, 4), strides=4, padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 4, 512)
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv3DTranspose(256, (8, 8, 8), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 8, 256)
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv3DTranspose(128, (16, 16, 16), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 16, 128)
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv3DTranspose(64, (32, 32, 32), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 32, 64)
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv3DTranspose(1, (64, 64, 64), strides=2, padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 64, 64, 64, 1)

    return model


def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def make_discriminator_model():
    '''
    Create the discriminator model with structure:
    64x64x64 -> 64x32x32x32 -> 128x16x16x16 -> 256x8x8x8 -> 512x4x4x4 -> 1
    '''
    model = Sequential()

    model.add(InputLayer(input_shape=(64, 64, 64)))
    model.add(Reshape((64, 64, 64, 1)))

    model.add(Conv3D(64, (32, 32, 32), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(128, (16, 16, 16), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(256, (8, 8, 8), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(512, (4, 4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.add(Reshape((32768,)))
    model.add(Dense(1))

    return model


def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss