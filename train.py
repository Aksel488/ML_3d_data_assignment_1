from gc import callbacks
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, MaxPooling3D, Concatenate, Flatten, BatchNormalization,
    InputLayer, ReLU, LeakyReLU, Activation, Conv3DTranspose, Conv3D,
    Reshape, 
)



MODEL = 'models'
MODEL_NAME = 'test_2'
MODEL_PATH = f'./{MODEL}/{MODEL_NAME}'

DATASET = 'modelnet10.npz'
LEARNING_RATE = 1e-4
LEARNING_RATE_CLASSIFIER = 1e-3
BUFFER_SIZE = 5000
BATCH_SIZE = 32
Z_DIM = 100
EPOCHS = 3
NUM_IMAGES_TO_GENERATE = 4
N_CLASSES = 10

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


"""# Load data and create models"""

''' load data from file '''
# data = np.load('modelnet10.npz', allow_pickle=True)
# train = data['train_voxel']
# test = data['test_voxel']
# train_y = data['train_labels']
# test_y = data['test_labels']
# classes = data['class_map']
classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

def make_generator_model():
    '''
    Create the generator model with structure:
    (100,) -> 512x4x4x4 -> 256x8x8x8 -> 128x16x16x16 -> 64x32x32x32 -> 64x64x64
    '''
    model = Sequential()

    model.add(InputLayer(input_shape=(Z_DIM,)))
    model.add(Reshape((1,1,1,Z_DIM))) 
    assert model.output_shape == (None, 1, 1, 1, Z_DIM) # None is the batch size

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

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def make_discriminator_model():
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

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

"""# Save samples from training data"""

def render_training_data():
    '''
    Take first voxel_array of each class and store inn shapes.
    Plot the 3d shape and saves a render to images
    '''

    data = np.load('modelnet10.npz', allow_pickle=True)
    train = data['train_voxel']
    train_y = data['train_labels']
    
    shapes = []
    for i in range(10):
        for j in range(len(train_y)):
            if i == train_y[j]:
                shapes.append([classes[i], train[j]])
                break

    for shape in shapes:
        ax = plt.axes(projection='3d')
        ax.voxels(shape[1], facecolors='red')
        ax.view_init(azim=-60, elev=30)
        plt.title(shape[0])
        plt.axis('off')
        plt.savefig(f'images/{shape[0]}.png')
        # plt.show()
        plt.clf()

"""# Load data function"""

def load_data(type):
    '''
    function for loading data from the training data and returning it as a Dataset
    '''

    data = np.load(DATASET, allow_pickle=True)
    train_voxel = data["train_voxel"] # Training 3D voxel samples
    
    if type == 'classifier':
        train_labels = data["train_labels"] # Training labels (integers from 0 to 9)
        #train_voxel = tf.data.Dataset.from_tensor_slices(train_voxel).shuffle(BUFFER_SIZE, seed=10).batch(BATCH_SIZE)
        #train_labels = tf.data.Dataset.from_tensor_slices(train_labels).shuffle(BUFFER_SIZE, seed=10).batch(BATCH_SIZE)
        return train_voxel, train_labels
        
    else:
        #train_labels = data["train_labels"] # Training labels (integers from 0 to 9)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_voxel).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return train_dataset

def load_model():
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')


"""Save images"""

def save_images(model, epoch, test_input):
    img_save_path = os.path.join(MODEL_PATH, 'epoch_images')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    predictions = model(test_input, training=False)
    
    i = 1
    for shape in predictions:
        shape = np.reshape(shape, (64, 64, 64))
        ax = plt.axes(projection='3d')
        ax.voxels(shape, facecolors='red')
        ax.view_init(azim=-60, elev=30)
        plt.axis('off')
        plt.savefig(img_save_path + '/image_{}_at_epoch_{:03d}.png'.format(i, epoch))
        plt.clf()

        i += 1

"""# Train function"""

@tf.function
def train_fn(voxel_objects):
    noise = tf.random.normal([BATCH_SIZE, Z_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_voxel_objects = generator(noise, training=True)

        real_output = discriminator(voxel_objects, training=True)
        fake_output = discriminator(generated_voxel_objects, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    # averaging the loss between generated and real and only updates if more than 20% wrong
    # to avoid the descriminator outlearning the generator
    if (disc_loss > 2):
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Create folder for saving and loading model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# initiate models
generator = make_generator_model()
discriminator = make_discriminator_model()
print(generator.summary())
print(discriminator.summary())

# optimizers
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# checkpoint for saving during traing
checkpoint_dir = f'{MODEL_PATH}/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)

"""# Train loop"""

def train():
    s = int(time.time())
    file = open('run.txt','w')
    file.write(f'started trainign at {s}' + "\n")
    file.close()

    # fixed seed for generating images after each epoch
    seed = tf.random.normal([NUM_IMAGES_TO_GENERATE, Z_DIM]) 

    g_loss = []
    d_loss = []
    iteration = 1

    file = open('run.txt','a')
    file.write(f'Loading data' + "\n")
    train_dataset = load_data()
    file.write(f'Done loading data' + "\n")
    file.close()

    for epoch in range(EPOCHS):
        file = open('run.txt','a')
        file.write(f'Started on epoch {epoch + 1}' + "\n")
        file.close() 
        start = time.time()

        for voxel_batch in train_dataset:
            gen_loss, disc_loss = train_fn(voxel_batch)

            g_loss.append([iteration, gen_loss])
            d_loss.append([iteration, disc_loss])
            iteration += 1

            file = open('run.txt','a')
            file.write(f'Done with iteration {iteration}' + "\n")
            file.close()

        # save
        ckpt_manager.save()
        save_images(generator, epoch + 1, seed) 

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        file = open('run.txt','a')
        file.write(f'epoch {epoch + 1} took {time.time()-start} seconds' + "\n")
        file.close()

    save_images(generator, EPOCHS, seed)
    np.savez_compressed(f'loss_plot_{MODEL_NAME}', generator = g_loss, discriminator = d_loss)

    s = int(time.time())

    file = open('run.txt','a')
    file.write(f'finished trainign at {s}' + "\n")
    file.close()
    
    
def myprint(s):
    with open('classifier_summary.txt','a') as f:
        print(s, file=f)

    
def save_plots(history):
    """
    Function for plotting and saving accuracy and loss of a model.
    """
    img_save_path = os.path.join('./models', 'plots')
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
    
    
def convert_sequential_model():
    input_layer = Input(batch_shape=discriminator.layers[0].input_shape)
    prev_layer = input_layer
    for layer in discriminator.layers:
        layer._inbound_nodes = []
        prev_layer = layer(prev_layer)
        
    return Model([input_layer], [prev_layer]), input_layer


def descriminator_to_classifier():
    load_model()
    discriminator.build()
    discriminator.pop() # Drop last Dense layer
    discriminator.pop() # Drop last reshape layer
    
    func_model, input_layer = convert_sequential_model()
    
    c2 = MaxPooling3D(pool_size=(8, 8, 8), padding='same')(func_model.layers[7].output)
    c3 = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(func_model.layers[10].output)
    c4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(func_model.layers[13].output)
    concat = Concatenate(axis=-1)([c2, c3, c4])
    out = Flatten()(concat)
    out = Dense(N_CLASSES, activation='softmax')(out)
    
    return Model(input_layer, out)

def train_classifier():
    s = int(time.time())
    file = open('run.txt','w')
    file.write(f'started trainign at {s}' + "\n")
    file.close()
    
    classifier = descriminator_to_classifier()
    classifier.summary(print_fn=myprint)
    
    train_voxel, train_labels = load_data(type='classifier')
    
    classifier.compile(
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_CLASSIFIER),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(name='loss'), 
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    )
    
    checkpoint_path = './models/classifier/cp.ckpt'
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    
        # Train the model on the training data and training labels 
    history = classifier.fit(
        train_voxel, 
        train_labels,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        shuffle = True,
        verbose = 1,
        callbacks=[cp_callback]
    )
    
    save_plots(history) # Save loss and accuracy plots 
    
    s = int(time.time())
    file = open('run.txt','a')
    file.write(f'finished trainign at {s}' + "\n")
    file.close()
        


def test():
    load_model()

    inputs = tf.random.normal([4, Z_DIM])
    predictions = generator(inputs, training=False)
    np.savez_compressed('predictions', predictions = predictions)

    # save_images(generator, 1, inputs)

def main():
    # render_training_data()
    #train()
    # test()
    train_classifier()


if __name__ == "__main__":
    main()