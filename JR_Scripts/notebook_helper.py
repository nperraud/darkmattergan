import sys
sys.path.insert(0, '../')

import utils, evaluation
from data import fmap, Dataset
import metrics
import tensorflow as tf
import numpy as np
from PIL import Image
import imageio
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.cm as cm

root_folder = "/home/jonathan/Documents/Master_Thesis/"
cscs_root_folder = "/scratch/snx3000/rosenthj/"


def load_params(model_folder, cscs=False):
    cscs_results = ""
    if cscs:
        cscs_results = cscs_root_folder + "results/"
    else:
        cscs_results = root_folder + "CSCSResults/"
    model_folder = cscs_results + model_folder
    params = utils.load_dict_pickle(model_folder + "params.pkl")
    return params, model_folder


def get_dataset(mpc, num_images, params, cscs=False):
    filename = ""
    if cscs:
        filename = cscs_root_folder + 'data/nbody_{}Mpc_All.h5'.format(mpc)
    else:
        filename = root_folder + 'Data/nbody_{}Mpc_All.h5'.format(mpc)
    img_list = []
    for box_idx in np.arange(10):
        images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')[:num_images]
        images = params['cosmology']['forward_map'](images)
        img_list.append(images)
    images = np.array(img_list)
    return Dataset.Dataset_time(images, spix=params['image_size'][0])


def get_data(mpc, num_images, params, cscs=False):
    filename = ""
    if cscs:
        filename = cscs_root_folder + 'data/nbody_{}Mpc_All.h5'.format(mpc)
    else:
        filename = root_folder + 'Data/nbody_{}Mpc_All.h5'.format(mpc)
    img_list = []
    for box_idx in np.arange(10):
        images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')[:num_images]
        images = params['cosmology']['forward_map'](images)
        img_list.append(images)
    images = np.array(img_list)
    return images


def get_scaling(classes, weights):
    s = (weights[1] - weights[0]) / (classes[1] - classes[0])
    s = np.arange(10) * s
    return s + (weights[0] - s[classes[0]])


def get_cmap(name, n_colors):
    cmap = cm.get_cmap(name, n_colors)
    return cmap(np.arange(n_colors))


def get_viridis(n_colors):
    return get_cmap('viridis', n_colors)


def gen_images_and_add_to_list_old(lst, gan):
    width = gan.params['image_size'][0]
    z = gan._sample_latent(1)[:1]
    frames = get_scaling(gan.params['time']['classes'], gan.params['time']['class_weights'])
    for idx in frames:
        y = np.asarray([[idx]])
        g_fake = gan._sess.run([gan._model.G_c_fake], feed_dict={gan._z: z, gan._model.y: y})
        img = np.reshape(g_fake, (width, width))
        lst.append(img)


def gen_images_10_time_steps(gan, chpt):
    z = np.repeat(gan._sample_latent(1)[:1], 10, axis=0)
    print(z.shape)
    frames = get_scaling(gan.params['time']['classes'], gan.params['time']['class_weights'])
    imgs = gan.generate(z=z, checkpoint=chpt, y=np.reshape(frames, (10, 1)), single_channel=None)[0]
    imgs = np.array(imgs)
    print(imgs.shape)
    return np.reshape(imgs, imgs.shape[0:3])


def gen_images_and_add_to_list(lst, gan, chpt):
    lst.extend(gen_images_10_time_steps(gan, chpt))


def wasserstein_distance(x_og, y, w):
    assert (x_og.shape == y.shape == w.shape)
    x = np.copy(x_og)
    loss = 0
    for idx in range(x.shape[0] - 1):
        d = y[idx] - x[idx]
        x[idx] = x[idx] + d
        x[idx + 1] = x[idx + 1] - d
        loss = loss + np.abs(d * (w[idx + 1] - w[idx]))
    return loss / (w[-1] - w[0])


def get_main_and_contained_classes(params):
    contained_classes = np.array(params['time']['classes'])
    mi = np.min(contained_classes)
    ma = np.max(contained_classes)
    main_classes = np.arange(mi, ma + 1)
    return main_classes, contained_classes


def gen_contained_marker(d1, d2, classes, h=1):
    mat = np.zeros((d1.shape[0],h, d1.shape[1]))
    for c in classes:
        mat[c,:,:] = np.ones((h,d1.shape[1]))
    ma = np.max([np.max(d1), np.max(d2)])
    return mat * ma * 0.7


def visual_comparison_fake_real(dset, gan, chpt):
    series = dset.get_samples(1)[0]
    series = np.transpose(series, [2, 0, 1])
    img_series = np.array(gen_images_10_time_steps(gan, chpt))
    _, contained_classes = get_main_and_contained_classes(gan.params)
    marker = gen_contained_marker(img_series, series, contained_classes)
    fig, ax = plt.subplots(figsize=(128, 16))
    ax.imshow(np.vstack([np.hstack(img_series), np.hstack(marker), np.hstack(series)]), interpolation=None)
    # ax.imshow(np.hstack(series), interpolation=None)
    plt.tight_layout()
    return img_series, series


def zoom_comparison(fake_img, real_img, params):
    main_classes, contained_classes = get_main_and_contained_classes(params)
    real_img = real_img[main_classes]
    fake_img = fake_img[main_classes]
    marker = gen_contained_marker(fake_img, real_img, contained_classes)
    fig, ax = plt.subplots(figsize=(128, 16))
    ax.imshow(np.vstack([np.hstack(fake_img), np.hstack(marker), np.hstack(real_img)]), interpolation=None)
    plt.tight_layout()


def gen_latent(gan, n):
    prior = gan.params['prior_distribution']
    return utils.sample_latent(n, gan.params['generator']['latent_dim'], prior)


def gen_fake_images(gan, n, t, checkpoint=None):
    t = np.array(t)
    z = gen_latent(gan, n)
    z = np.repeat(z, t.shape[0], axis=0)
    t = np.tile(t, n)
    frames = get_scaling(gan.params['time']['classes'], gan.params['time']['class_weights'])
    imgs = gan.generate(z=z, y=np.reshape(t, (z.shape[0], 1)), single_channel=None,
                        checkpoint=checkpoint)[0]
    imgs = np.array(imgs)
    print(imgs.shape)
    return np.reshape(imgs, imgs.shape[0:3])


def print_sub_dict_params(d_name, params):
    print("\n{} params".format(d_name).title())
    for key, value in params.items():
        if isinstance(value, dict):
            print(" {}.{}: dict".format(d_name, key))
        else:
            print(" {}.{}: {}".format(d_name, key, value))


def print_param_dict(params):
    print("General Params")
    for key, value in params.items():
        if not isinstance(value, dict):
            print(" {}: {}".format(key, value))
    for key, value in params.items():
        if isinstance(value, dict):
            print_sub_dict_params(key, value)


def reshape_data_to_old_format(data, params):
    data = np.transpose(data, [3, 0, 1, 2])[params['time']['classes']]
    data = np.transpose(data, [1, 0, 2, 3])
    s = data.shape
    return np.reshape(data, (s[0] * s[1], s[2], s[3]))


def plot_mass_hist(data, data_name, lim, params, cmap):
    plt.figure()
    plt.title("Mass Histogram of {} Data".format(data_name))
    nc = params['time']['classes']
    for i in range(nc):
        hist_f, bins, _ = metrics.mass_hist(dat=data[i::nc], lim=lim)
        plt.plot(bins, hist_f, '-', label='t{}'.format(params['time']['classes'][i]), c=cmap[i])
        plt.legend()
    plt.yscale("log")
    plt.xscale("log")


def plot_real_vs_fake_mass_hists(real, fake, lim, params):
    nc = params['time']['num_classes']
    for i in range(nc):
        plt.figure()
        plt.title("Mass Histograms for t{}".format(params['time']['classes'][i]))
        plt.yscale("log")
        plt.xscale("log")
        hist_r, bins, _ = metrics.mass_hist(dat=real[i::nc], lim=lim)
        plt.plot(bins, hist_r, '-', label='Real', c='r')
        hist_f, bins, _ = metrics.mass_hist(dat=fake[i::nc], lim=lim)
        plt.plot(bins, hist_f, '-', label='Fake', c='b')
        plt.legend()


def get_lim_mass(data):
    _, _, lim_mass = metrics.mass_hist(data)
    lim_mass = list(lim_mass)
    lim_mass[1] = lim_mass[1] + 1
    return lim_mass