import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Sequential


MODEL = 'models'
MODEL_NAME = 'test_1'
MODEL_PATH = f'./{MODEL}/{MODEL_NAME}'

DATASET = 'modelnet10.npz'
LEARNING_RATE = 1e-4
BUFFER_SIZE = 500
BATCH_SIZE = 32
Z_DIM = 100
EPOCHS = 10
NUM_IMAGES_TO_GENERATE = 4
NUM_WORKERS = 1

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

    model.add(layers.InputLayer(input_shape=(Z_DIM,)))
    model.add(layers.Reshape((1,1,1,Z_DIM)))
    assert model.output_shape == (None, 1, 1, 1, Z_DIM) # None is the batch size

    model.add(layers.Conv3DTranspose(512, (4, 4, 4), strides=4, padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 4, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(256, (8, 8, 8), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(128, (16, 16, 16), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(64, (32, 32, 32), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(1, (64, 64, 64), strides=2, padding='same', use_bias=False, activation='sigmoid'))
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

    model.add(layers.InputLayer(input_shape=(64, 64, 64)))
    model.add(layers.Reshape((64, 64, 64, 1)))

    model.add(layers.Conv3D(64, (32, 32, 32), strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv3D(128, (16, 16, 16), strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv3D(256, (8, 8, 8), strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv3D(512, (4, 4, 4), strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Reshape((32768,)))
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

"""# Save samples from training data"""

'''
Take first voxel_array of each class and store inn shapes.
Plot the 3d shape and saves an image to images
'''
# shapes = []
# for i in range(10):
#     for j in range(len(train_y)):
#         if i == train_y[j]:
#             shapes.append([classes[i], train[j]])
#             break

# for shape in shapes:
#     ax = plt.axes(projection='3d')
#     ax.voxels(shape[1], facecolors='red')
#     ax.view_init(azim=-60, elev=30)
#     plt.title(shape[0])
#     plt.axis('off')
#     plt.savefig(f'images/{shape[0]}.png')
#     # plt.show()
#     plt.clf()

"""# Load data function"""

def load_data():
    '''
    function for loading data from the training data and returning it as a Dataset
    '''

    data = np.load(DATASET, allow_pickle=True)
    train_voxel = data["train_voxel"] # Training 3D voxel samples
    #train_labels = data["train_labels"] # Training labels (integers from 0 to 9)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_voxel).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset

def load_model():
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')


"""Save images"""

def save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    img_save_path = os.path.join(MODEL_PATH, 'epoch_images')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    
    i = 1
    for shape in predictions:
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
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

"""# Train loop"""

# Create folder for saving and loading model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# initiate models
generator = make_generator_model()
discriminator = make_discriminator_model()
# generator.summary()
# discriminator.summary()

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
        # save_images(generator, epoch + 1, seed) 

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        file = open('run.txt','a')
        file.write(f'epoch {epoch + 1} took {time.time()-start} seconds' + "\n")
        file.close()

    # save_images(generator, EPOCHS, seed)
    np.savez_compressed('loss_plot', generator = g_loss, discriminator = d_loss)

    s = int(time.time())

    file = open('run.txt','a')
    file.write(f'finished trainign at {s}' + "\n")
    file.close()

def descriminator_to_classifier():
    load_model()
    

def test():
    load_model()

    ''' 
    @TODO 
    code for generating images
    '''

def main():
    # train()
    test()


if __name__ == "__main__":
    main()