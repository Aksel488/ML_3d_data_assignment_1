from gc import callbacks
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, MaxPooling3D, Concatenate, Flatten,
)

from model import (
    make_generator_model, generator_loss,
    make_discriminator_model, discriminator_loss
)
import config


def render_training_data():
    '''
    Take first voxel_array of each class and store inn shapes.
    Plot the 3d shape and save them as images
    '''

    data = np.load('modelnet10.npz', allow_pickle=True)
    train = data['train_voxel']
    train_y = data['train_labels']

    classes = [
        'bathtub', 'bed', 'chair', 
        'desk', 'dresser', 'monitor', 
        'night_stand', 'sofa', 'table', 'toilet'
    ]
    
    shapes = []
    for i in range(len(classes)):
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


def load_data(type):
    '''
    function for loading data from the training data and returning it as a Dataset
    '''

    data = np.load(config.DATASET, allow_pickle=True)
    train_voxel = data["train_voxel"] # Training 3D voxel samples
    
    if type == 'classifier':
        train_labels = data["train_labels"] # Training labels (integers from 0 to 9)
        #train_voxel = tf.data.Dataset.from_tensor_slices(train_voxel).shuffle(BUFFER_SIZE, seed=10).batch(BATCH_SIZE)
        #train_labels = tf.data.Dataset.from_tensor_slices(train_labels).shuffle(BUFFER_SIZE, seed=10).batch(BATCH_SIZE)
        return train_voxel, train_labels
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_voxel).shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
        return train_dataset


def load_model():
    '''
    Load the trained models from the checkpoint manager
    '''
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    else:
        print('No checkpoints to restore from')


def save_images(model, epoch, test_input):
    '''
    uses the test_input to generate 3d shapes,
    renders and saves them
    '''

    img_save_path = os.path.join(config.MODEL_PATH, 'epoch_images')
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


'''
Code down until line 237
based on: https://www.tensorflow.org/tutorials/generative/dcgan
'''

@tf.function
def train_gan_fn(voxel_objects):
    '''
    The training function for the GAN on one iteration / batch.
    Generates BATCH_SIZE number of fake and real outputs.
    Calculates the loss and gradients for the generator and the discriminator, and applies it on them 
    '''
    noise = tf.random.normal([config.BATCH_SIZE, config.Z_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_voxel_objects = generator(noise, training=True)

        real_output = discriminator(voxel_objects, training=True)
        fake_output = discriminator(generated_voxel_objects, training=True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    # summing the loss between generated and real, only updates if more than 20% wrong
    # to avoid the descriminator outlearning the generator
    # 
    # Not sure if implemented correctly, might be: 
    # if (disc_loss / (config.BATCH_SIZE * 2) > 0.2):
    if (disc_loss > ((config.BATCH_SIZE * 2) * 0.2)):
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train_GAN():
    '''
    Training loop for the GAN.
    Calls the train_gan_fn() function for every batch in every epoch.
    Will save the loss for the generator and the discriminator,
    and render images after each epoch.
    '''
    
    s = int(time.time())
    file = open('run.txt','w')
    file.write(f'started trainign at {s}' + "\n")
    file.close()

    # fixed seed for generating images from same vectors after each epoch
    seed = tf.random.normal([config.NUM_IMAGES_TO_GENERATE, config.Z_DIM]) 

    g_loss = []
    d_loss = []
    iteration = 1

    file = open('run.txt','a')
    file.write(f'Loading data' + "\n")
    train_dataset = load_data()
    file.write(f'Done loading data' + "\n")
    file.close()

    for epoch in range(config.EPOCHS):
        file = open('run.txt','a')
        file.write(f'Started on epoch {epoch + 1}' + "\n")
        file.close() 
        start = time.time()

        for voxel_batch in train_dataset:
            gen_loss, disc_loss = train_gan_fn(voxel_batch)

            g_loss.append([iteration, gen_loss])
            d_loss.append([iteration, disc_loss])
            iteration += 1

            file = open('run.txt','a')
            file.write(f'Done with iteration {iteration}' + "\n")
            file.close()

        # save checkpoint and render images
        ckpt_manager.save()
        save_images(generator, epoch + 1, seed) 

        epoch_summary = f'epoch {epoch + 1} took {time.time()-start} seconds' + "\n"
        print (epoch_summary)

        file = open('run.txt','a')
        file.write(epoch_summary)
        file.close()

    save_images(generator, config.EPOCHS, seed)
    np.savez_compressed(f'loss_plot_{config.MODEL_NAME}', generator = g_loss, discriminator = d_loss)

    s = int(time.time())
    file = open('run.txt','a')
    file.write(f'finished trainign at {s}' + "\n")
    file.close()


# Create folder for saving and loading model if not exists
if not os.path.exists(config.MODEL_PATH):
    os.makedirs(config.MODEL_PATH)

# initiate loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# models
generator = make_generator_model()
discriminator = make_discriminator_model()
print(generator.summary())
print(discriminator.summary())

# optimizers
generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)

# checkpoint for saving during traing
checkpoint_dir = f'{config.MODEL_PATH}/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)


def test_GAN():
    '''
    Test GAN by generating shapes and save them as images
    '''
    load_model()

    inputs = tf.random.normal([config.NUM_IMAGES_TO_GENERATE, config.Z_DIM])
    save_images(generator, 1, inputs)

    
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
    '''
    Function code from: [https://stackoverflow.com/questions/61130836/convert-functional-model-to-sequential-keras]
    Converts Sequential model to Functional model.
    '''
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
    out = Dense(config.N_CLASSES, activation='softmax')(out)
    
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
        optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_CLASSIFIER),
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
        batch_size = config.BATCH_SIZE,
        epochs = config.EPOCHS,
        shuffle = True,
        verbose = 1,
        callbacks=[cp_callback]
    )
    
    save_plots(history) # Save loss and accuracy plots 
    
    s = int(time.time())
    file = open('run.txt','a')
    file.write(f'finished trainign at {s}' + "\n")
    file.close()


def main():
    '''
    Uncomment the function you want to run, 
    remember to change parameters in the config file
    '''
    
    # render_training_data()
    # train_GAN()
    # test_GAN()
    train_classifier()


if __name__ == "__main__":
    main()
