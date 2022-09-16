'''
If you have predictions saved in the predictions.npz file
you can run this to render images of them 
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import config

data = np.load('predictions.npz', allow_pickle=True)
predictions = data['predictions']

img_save_path = os.path.join(config.MODEL_PATH, 'epoch_images')
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

i = 1
for shape in predictions:
    shape = np.reshape(shape, (64, 64, 64))
    print(shape.shape)
    ax = plt.axes(projection='3d')
    ax.voxels(shape, facecolors='red')
    ax.view_init(azim=-60, elev=30)
    plt.axis('off')
    plt.savefig(img_save_path + '/image_{}_at_epoch_{:03d}.png'.format(i, 10))
    plt.clf()

    i += 1