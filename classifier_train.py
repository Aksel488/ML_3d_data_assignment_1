from json import load
import os
from random import seed
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 

from keras.layers import (
    InputLayer, Reshape, Conv3D, 
    LeakyReLU, Dense, MaxPooling3D, 
    Concatenate, Flatten, Activation, 
    BatchNormalization, 
)

from keras.models import Sequential
import numpy as np
from train import make_discriminator_model

MODEL = 'models'
MODEL_NAME = 'test_1'
MODEL_PATH = f'./{MODEL}/{MODEL_NAME}'

DATASET = 'modelnet10.npz'
CHECKPOINT_DIS = f'{MODEL_PATH}/discriminator.pth'
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BUFFER_SIZE = 5000
BATCH_SIZE = 32
EPOCHS = 4
N_CLASSES = 10

def make_classifier():
    '''
    Create the generator model with structure:
    64x64x64 -> 64x32x32x32 -> 128x16x16x16 -> 256x8x8x8 -> 512x4x4x4 -> 1
    '''
    model = Sequential()

    model.add(InputLayer(input_shape=(64, 64, 64)))
    model.add(Reshape((64, 64, 64, 1)))

    model.add(Conv3D(64, (32, 32, 32), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    c2 = model.add(Conv3D(128, (16, 16, 16), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling3D(pool_size=(8, 8, 8), padding='same'))

    c3 = model.add(Conv3D(256, (8, 8, 8), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling3D(pool_size=(4, 4, 4), padding='same'))

    c4 = model.add(Conv3D(512, (4, 4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Concatenate([c2, c3, c4], axis=-1))
    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))

    return model

def load_data():
    '''
    Function for loading training data and training labels
    from the dataset and returning them as tf.Dataset objects.
    '''
    data = np.load(DATASET, allow_pickle=True)
    train_voxel = data["train_voxel"] # Training 3D voxel samples
    test_voxel = data["test_voxel"]
    train_labels = data["train_labels"] # Training labels (integers from 0 to 9)
    test_labels = data["test_labels"]
    class_map = data['class_map']

    train_voxel = tf.data.Dataset.from_tensor_slices(train_voxel).shuffle(BUFFER_SIZE, seed=10).batch(BATCH_SIZE)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels).shuffle(BUFFER_SIZE, seed=10).batch(BATCH_SIZE)

    return train_voxel, test_voxel, train_labels, test_labels, class_map


def save_plots(history):
    """
    Function for plotting and saving accuracy and loss of a model.
    """
    img_save_path = os.path.join(MODEL_PATH, 'plots')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(img_save_path + '/accuracy.png')
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(img_save_path + '/loss.png')


def main():
    s = int(time.time())
    file = open('run.txt','w')
    file.write(f'started training at {s}' + "\n")
    file.close()

    train_voxel, test_voxel, train_labels, test_labels, class_map = load_data()

    model = make_discriminator_model() # Create discriminator from Task A
    model.load_weights('./training_checkpoints/ckpt-10')
    model.summary() # Show model architecture

    model.compile(
        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = tf.metrics.SparseCategoricalAccuracy(name='accuracy'),
    )

    # Train the model on the training data and training labels 
    history = model.fit(
        training_data = (train_voxel, train_labels),
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        shuffle = True,
        verbose = 1,
    )

    save_plots(history) # Save loss and accuracy plots 
    

if __name__ == '__main__':
    main()


