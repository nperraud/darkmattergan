# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import os
from model import TemporalGanModelv3, TemporalGanModelv4
from gan import TimeCosmoGAN
import utils
import pickle
from data import fmap, path, Dataset
import tensorflow as tf
import numpy as np
import itertools


def load_gan(pathgan, GANtype=TimeCosmoGAN):
    """Load GAN object from path."""
    with open(os.path.join(pathgan, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    params['save_dir'] = pathgan
    obj = GANtype(params)

    return obj

Mpc = 500

gan = load_gan(sys.argv[1], GANtype=TimeCosmoGAN)
params = gan.params

s_dir = params['summary_dir']
q = ""
for i in range(5):
    q = q + s_dir.split('/')[i] + '/'
q = q + 'summaries_32_C2_v2/'
params['summary_dir'] = q
gan.params = params


img_list = []

filename = '/scratch/snx3000/rosenthj/data/nbody_{}Mpc_All.h5'.format(Mpc)
for box_idx in params['time']['classes']:
    images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')
    images = params['cosmology']['forward_map'](images)
    img_list.append(images)

images = np.array(img_list)
print ("Images shape: {}".format(images.shape))
dataset = Dataset.Dataset_time(images, spix=params['image_size'][0], shuffle=True)

chkp_lst = []
for f in os.listdir(params['save_dir']):
    if f.endswith("meta"):
        chkp_lst.append(int((f.split(".")[0]).split("-")[-1]))

chkp_lst.sort()
print("Checkpoints: {}".format(chkp_lst))

sess = tf.Session()
sample_z = None
X_real = None

print("Preparations Complete Commencing Stats Evaluation")

for i in range(len(chkp_lst)):
    chkp = chkp_lst[i]
    print("Loading checkpoint {}".format(chkp))
    gan.load(sess=sess, checkpoint=chkp)
    print("Finished loading checkpoint")
    gan._counter = chkp
    if i == 0:
        print("Initializing constant stats")
        gan._stats = gan.params['cosmology']['stats']
        gan._stats['N'] = gan.params['cosmology']['Nstats']
        gan._sum_data_iterator = itertools.cycle(dataset.iter(gan._stats['N']))

        #gan._var.eval()
        #gan._mean.eval()
        gan._summary_writer = tf.summary.FileWriter(
            gan.params['summary_dir'], gan._sess.graph)

        sample_z = gan._sample_latent(gan.batch_size)
        X_real = dataset.get_samples(gan.batch_size)
    gan._train_log(
        gan._get_dict(sample_z, X_real))
