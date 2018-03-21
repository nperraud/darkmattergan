import sys
sys.path.insert(0, '../')

import utils,  sys
from JR_Scripts import dict_reader, time_toy_generator
from model import WGanModel, WNGanModel, TemporalGanModelv3
from gan import GAN
import tensorflow as tf
import numpy as np
from PIL import Image
import imageio


def main():
    # Load parameters
    param_paths = sys.argv[1]
    params = dict_reader.read_gan_dict(param_paths)

    print("All params")
    print(params)
    print("\nDiscriminator Params")
    print(params['discriminator'])
    print("\nGenerator Params")
    print(params['generator'])
    print("\nOptimization Params")
    print(params['optimization'])
    print("\nCosmo Params")
    print(params['cosmology'])
    print()

    # Initialize model
    if params['model_idx'] == 0:
        model = WGanModel
    if params['model_idx'] == 1:
        model = WNGanModel
    if params['model_idx'] == 2:
        model = TemporalGanModelv3
    gan = GAN(params, model)
    gan._sess = tf.Session()
    leo_results = "/home/jonathan/Documents/Master_Thesis/LeoResults/"
    folder = "models_M23_moving/TWGAN_TS8s_gp50_2018_3_21_19_15_checkpoints/"
    specific_model = "TWGAN_TS8s_gp50-176000"
    gan._load(leo_results + folder + specific_model)
    z = gan._sample_latent(1)
    first = 1.0 / params['num_classes']
    frames = 8
    images = []
    for i in range(frames):
        y = np.asarray([[first + i * (1 - first) / frames]])
        g_fake = gan._sess.run([gan._G_fake], feed_dict={gan._z:z, gan._model.y:y})[0]
        g_fake = np.reshape(g_fake, (128,128))
        img = (g_fake + 1) * 128
        images.append(img)
        img = Image.fromarray(img)
        img.convert('RGB').save("images_smooth/img_{}.png".format(str(i+100)))
    images = np.asarray(images)
    imageio.mimsave('gifs/' + specific_model + 't.gif', images, duration=0.5)
    return 0


main()
