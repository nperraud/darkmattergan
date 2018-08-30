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
red = ['z = 0.000', 'z = 0.111', 'z = 0.250', 'z = 0.428', 'z = 0.666', 'z = 1.000', 'z = 1.500', 'z = 2.333', 'z = 4.000', 'z = 9.000']

def load_params(model_folder, cscs=False):
    cscs_results = ""
    if cscs:
        cscs_results = cscs_root_folder + "results/"
    else:
        cscs_results = root_folder + "CSCSResults/"
    model_folder = cscs_results + model_folder
    params = utils.load_dict_pickle(model_folder + "params.pkl")
    return params, model_folder


def get_dataset(mpc, num_images, params, shuffle=False, cscs=False):
    filename = ""
    if cscs:
        filename = cscs_root_folder + 'data/nbody_{}Mpc_All.h5'.format(mpc)
    else:
        filename = root_folder + 'Data/nbody_{}Mpc_All.h5'.format(mpc)
    img_list = []
    for box_idx in np.arange(10):
        images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')[:num_images]
        images = params['cosmology']['forward_map'](images / 3)
        img_list.append(images)
    images = np.array(img_list)
    return Dataset.Dataset_time(images, shuffle=shuffle, spix=params['image_size'][0])


def get_data(mpc, num_images, params, cscs=False):
    filename = ""
    if cscs:
        filename = cscs_root_folder + 'data/nbody_{}Mpc_All.h5'.format(mpc)
    else:
        filename = root_folder + 'Data/nbody_{}Mpc_All.h5'.format(mpc)
    img_list = []
    for box_idx in np.arange(10):
        images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')[:num_images]
        images = params['cosmology']['forward_map'](images / 3)
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


def visual_comparison_fake_real_compact(dset, gan, chpt):
    series = dset.get_samples(1)[0]
    series = np.transpose(series, [2, 0, 1])
    img_series = np.array(gen_images_10_time_steps(gan, chpt))
    main_classes, contained_classes = get_main_and_contained_classes(gan.params)
    marker = gen_contained_marker(img_series, series, contained_classes)
    img_series = np.flip(img_series[main_classes], 0)
    series = np.flip(series[main_classes], 0)
    marker = np.flip(marker[main_classes], 0)
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
    t = np.array(gan.params['time']['class_weights'])
    z = gen_latent(gan, n)
    z = np.repeat(z, t.shape[0], axis=0)
    t = np.tile(t, n)
    # frames = get_scaling(gan.params['time']['classes'], gan.params['time']['class_weights'])
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
    nc = params['time']['num_classes']
    for i in range(nc):
        hist_f, bins, _ = metrics.mass_hist(dat=data[i::nc], lim=lim)
        plt.plot(bins, hist_f, '-', label='${}$'.format(red[params['time']['classes'][i]]), c=cmap[i])
        plt.legend()
    plt.ylabel("Frequency", labelpad=22, rotation=0)
    plt.xlabel("Pixel Intensity")
    plt.yscale("log")
    plt.xscale("log")


def plot_real_vs_fake_mass_hists(real, fake, lim, params):
    nc = params['time']['num_classes']
    for i in range(nc):
        plt.figure()
        plt.title("Mass Histograms for {}".format(red[params['time']['classes'][i]]))
        plt.ylabel("Frequency", labelpad=22, rotation=0)
        plt.xlabel("Pixel Intensity")
        plt.yscale("log")
        plt.xscale("log")
        hist_r, bins, _ = metrics.mass_hist(dat=real[i::nc], lim=lim)
        plt.plot(bins, hist_r, '-', label='Real', c='r')
        hist_f, bins, _ = metrics.mass_hist(dat=fake[i::nc], lim=lim)
        plt.plot(bins, hist_f, '-', label='Fake', c='b')
        plt.legend()


def plot_mass_hists(data_list, labels, colors, lim, params):
    nc = params['time']['num_classes']
    for i in range(nc):
        plt.figure()
        plt.title("Mass Histograms for {}".format(red[params['time']['classes'][i]]))
        plt.ylabel("Frequency", labelpad=22, rotation=0)
        plt.xlabel("Pixel Intensity")
        plt.yscale("log")
        plt.xscale("log")
        for j in range(len(data_list)):
            hist, bins, _ = metrics.mass_hist(dat=data_list[j][i::nc], lim=lim)
            plt.plot(bins, hist, '-', label=labels[j], c=colors[j])
        plt.legend()


def get_lim_mass(data):
    _, _, lim_mass = metrics.mass_hist(data)
    lim_mass = list(lim_mass)
    lim_mass[1] = lim_mass[1] + 1
    return lim_mass


def peak_hist_over_time(a, b, params, lim=None, title_a="Real Data", title_b="Fake Data"):
    nc = params['time']['num_classes']
    cmap = get_cmap('viridis', nc + 2)
    cmap = cmap[1:-1]
    plt.figure()
    plt.title("Peak Histograms of {}".format(title_a))
    plt.ylabel("Frequency", labelpad=26, rotation=0)
    plt.xlabel("Peak Intensity")
    plt.yscale("log")
    plt.xscale("log")
    for i in range(nc):
        peak_hist, x, _ = metrics.peak_count_hist(a[i::nc], lim=lim)
        plt.plot(x, peak_hist, '-', label='${}$'.format(red[params['time']['classes'][i]]), c=cmap[i])
        plt.legend()
    plt.figure()
    plt.title("Peak Histograms of {}".format(title_b))
    for i in range(nc):
        peak_hist, x, _ = metrics.peak_count_hist(b[i::nc], lim=lim)
        plt.plot(x, peak_hist, '-', label='${}$'.format(red[params['time']['classes'][i]]), c=cmap[i])
        plt.legend()
    plt.ylabel("Frequency", labelpad=26, rotation=0)
    plt.xlabel("Peak Intensity")
    plt.yscale("log")
    plt.xscale("log")


def peak_hist_a_vs_b(a, b, params, lim, label_a="Real", label_b="Fake"):
    nc = params['time']['num_classes']
    # cmap = get_cmap('viridis', nc)
    for i in range(params['time']['num_classes']):
        plt.figure()
        plt.title("Peak Histograms for ${}$".format(red[params['time']['classes'][i]]))
        plt.ylabel("Frequency", labelpad=26, rotation=0)
        plt.xlabel("Peak Intensity")
        plt.yscale("log")
        plt.xscale("log")
        peak_hist, x, _ = metrics.peak_count_hist(a[i::nc], lim=lim)
        plt.plot(x, peak_hist, '-', label='{}'.format(label_a), c='r')
        peak_hist, x, _ = metrics.peak_count_hist(b[i::nc], lim=lim)
        plt.plot(x, peak_hist, '-', label='{}'.format(label_b), c='b')
        plt.legend()


def plot_peak_hists(data_list, labels, colors, lim, params):
    nc = params['time']['num_classes']
    for i in range(params['time']['num_classes']):
        plt.figure()
        plt.title("Peak Histograms for ${}$".format(red[params['time']['classes'][i]]))
        plt.ylabel("Frequency", labelpad=26, rotation=0)
        plt.xlabel("Peak Intensity")
        plt.yscale("log")
        plt.xscale("log")
        for j in range(len(data_list)):
            peak_hist, x, _ = metrics.peak_count_hist(data_list[j][i::nc], lim=lim)
            plt.plot(x, peak_hist, '-', label=labels[j], c=colors[j])
        plt.legend()


def power_spectral_densities(data, params, data_name):
    nc = params['time']['num_classes']
    cmap = get_cmap('viridis', nc + 2)
    cmap = cmap[1:-1]
    plt.figure()
    plt.title("Power Spectrum Densities of {}".format(data_name))
    # plt.ylabel("Frequency", labelpad=26, rotation=0)
    plt.xlabel("Wavelength")
    plt.yscale("log")
    plt.xscale("log")
    for i in range(params['time']['num_classes']):
        psd, bins = metrics.power_spectrum_batch_phys(X1=data[i::nc])
        psd = np.mean(psd, axis=0)
        plt.plot(bins, psd, '-', label='${}$'.format(red[params['time']['classes'][i]]), c=cmap[i])
        plt.legend()


def power_spectral_density_a_vs_b(a, b, params, label_a="Real", label_b="Fake"):
    nc = params['time']['num_classes']
    for i in range(nc):
        plt.figure()
        plt.title("Power Spectral Densities for Redshift ${}$".format(red[params['time']['classes'][i]]))
        plt.ylabel("Energy", labelpad=22, rotation=0)
        plt.xlabel("Frequency")
        plt.yscale("log")
        plt.xscale("log")
        psd_r, x = metrics.power_spectrum_batch_phys(a[i::nc])
        plt.plot(x, np.mean(psd_r, axis=0), '-', label=label_a, c='r')
        psd_f, x = metrics.power_spectrum_batch_phys(b[i::nc])
        plt.plot(x, np.mean(psd_f, axis=0), '-', label=label_b, c='b')
        plt.legend()


def plot_power_spectral_densities(data_list, labels, colors, params):
    nc = params['time']['num_classes']
    for i in range(nc):
        plt.figure()
        plt.title("Power Spectral Densities for Redshift ${}$".format(red[params['time']['classes'][i]]))
        plt.ylabel("Energy", labelpad=22, rotation=0)
        plt.xlabel("Frequency")
        plt.yscale("log")
        plt.xscale("log")
        for j in range(len(data_list)):
            psd_curve, x = metrics.power_spectrum_batch_phys(data_list[j][i::nc])
            plt.plot(x, np.mean(psd_curve, axis=0), '-', label=labels[j], c=colors[j])
        plt.legend()
