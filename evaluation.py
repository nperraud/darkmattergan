"""Evaluation module.

This module contains helping functions for the evaluation of the models.
"""

import pickle
import numpy as np
import metrics
import plot
import matplotlib.pyplot as plt
from model import *
from gan import *


def load_gan(pathgan, GANtype=CosmoGAN):
    """Load GAN object from path."""
    with open(pathgan + 'params.pkl', 'rb') as f:
        params = pickle.load(f)
    params['save_dir'] = pathgan
    obj = GANtype(params)

    return obj


def generate_samples(obj, N=None, checkpoint=None, **kwards):
    """Generate sample from gan object."""
    gen_sample, gen_sample_raw = obj.generate(
        N=N, checkpoint=checkpoint, **kwards)
    gen_sample = np.squeeze(gen_sample)
    gen_sample_raw = np.squeeze(gen_sample_raw)
    return gen_sample, gen_sample_raw


def compute_and_plot_psd(raw_images, gen_sample_raw, display=True, is_3d=False):
    """Compute and plot PSD from raw images."""
    psd_real, x = metrics.power_spectrum_batch_phys(X1=raw_images, is_3d=is_3d)
    psd_real_mean = np.mean(psd_real, axis=0)

    psd_gen, x = metrics.power_spectrum_batch_phys(X1=gen_sample_raw, is_3d=is_3d)
    psd_gen_mean = np.mean(psd_gen, axis=0)
    l2, logel2, l1, logel1 = metrics.diff_vec(psd_real_mean, psd_gen_mean)

    if display:
        print('Log l2 PSD loss: {}\n'
              'L2 PSD loss: {}\n'
              'Log l1 PSD loss: {}\n'
              'L1 PSD loss: {}'.format(logel2, l2, logel1, l1))

        plt.Figure()
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        linestyle = {
            "linewidth": 1,
            "markeredgewidth": 0,
            "markersize": 3,
            "marker": "o",
            "linestyle": "-"
        }

        plot.plot_with_shade(
            ax,
            x,
            psd_gen,
            color='r',
            label="Fake $\mathcal{F}(X))^2$",
            **linestyle)
        plot.plot_with_shade(
            ax,
            x,
            psd_real,
            color='b',
            label="Real $\mathcal{F}(X))^2$",
            **linestyle)
        # ax.set_ylim(bottom=0.1)
        ax.title.set_text("2D Power Spectrum\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()

    return logel2, l2, logel1, l1


def compute_and_plot_peak_cout(raw_images, gen_sample_raw, display=True):
    """Compute and plot peak count histogram from raw images."""
    y_real, y_fake, x = metrics.peak_count_hist_real_fake(raw_images, gen_sample_raw)
    l2, logel2, l1, logel1 = metrics.diff_vec(y_real, y_fake)
    if display:
        print('Log l2 Peak Count loss: {}\n'
              'L2 Peak Count loss: {}\n'
              'Log l1 Peak Count loss: {}\n'
              'L1 Peak Count loss: {}'.format(logel2, l2, logel1, l1))
        plt.Figure()
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        linestyle = {
            "linewidth": 1,
            "markeredgewidth": 0,
            "markersize": 3,
            "marker": "o",
            "linestyle": "-"
        }
        ax.plot(x, y_fake, label="Fake", color='r', **linestyle)
        ax.plot(x, y_real, label="Real", color='b', **linestyle)

        # ax.set_ylim(bottom=0.1)
        ax.title.set_text("Peak count\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()
    return l2, logel2, l1, logel1


def compute_and_plot_mass_hist(raw_images, gen_sample_raw, display=True):
    """Compute and plot mass histogram from raw images."""
    y_real, y_fake, x = metrics.mass_hist_real_fake(raw_images, gen_sample_raw)
    l2, logel2, l1, logel1 = metrics.diff_vec(y_real, y_fake)
    if display:
        print('Log l2 Mass histogram loss: {}\n'
              'L2 Peak Mass histogram: {}\n'
              'Log l1 Mass histogram loss: {}\n'
              'L1 Mass histogram loss: {}'.format(logel2, l2, logel1, l1))
        plt.Figure()
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        linestyle = {
            "linewidth": 1,
            "markeredgewidth": 0,
            "markersize": 3,
            "marker": "o",
            "linestyle": "-"
        }
        ax.plot(x, y_fake, label="Fake", color='r', **linestyle)
        ax.plot(x, y_real, label="Real", color='b', **linestyle)

        # ax.set_ylim(bottom=0.1)
        ax.title.set_text("Mass histogram\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()
    return l2, logel2, l1, logel1


def upscale_image(obj, small, checkpoint=None):
    """Upscale image using the lappachsimple model.

    This function can be accelerated if the model is created only once.
    """
    # Number of sample to produce
    N = small.shape[0]

    # Dimension of the low res image
    lx, ly = small.shape[1:3]

    # Output dimension of the generator
    soutx, souty = obj.params['image_size'][:2]

    # Input dimension of the generator
    sinx = soutx // obj.params['generator']['upsampling']
    siny = souty // obj.params['generator']['upsampling']

    # Number of part to be generated
    nx = lx // sinx
    ny = ly // siny

    # Final output image
    output_image = np.zeros(
        shape=[N, soutx * nx, souty * ny, 1], dtype=np.float32)
    output_image[:] = np.nan

    # C) Generate the rest of the lines
    for j in range(ny):
        for i in range(nx):
            # 1) Generate the border
            border = np.zeros([N, soutx, souty, 3])
            if i:
                border[:, :, :, :1] = output_image[:, (
                    i - 1) * soutx:i * soutx, j * souty:(j + 1) * souty, :]
            if j:
                border[:, :, :, 1:2] = output_image[:, i * soutx:(
                    i + 1) * soutx, (j - 1) * souty:j * souty, :]
            if i and j:
                border[:, :, :, 2:3] = output_image[:, (
                    i - 1) * soutx:i * soutx, (j - 1) * souty:j * souty, :]

            # 2) Prepare low resolution
            y = np.expand_dims(small[:N][:, i * sinx:(
                i + 1) * sinx, j * siny:(j + 1) * siny], 3)

            # 3) Generate the image
            gen_sample, _ = obj.generate(
                N=N, border=border, y=y, checkpoint=checkpoint)
            output_image[:, i * soutx:(i + 1) * soutx, j * souty:(
                j + 1) * souty, :] = gen_sample

    return np.squeeze(output_image)


def equalizing_histogram(real, fake, nbins=1000, bs=2000):
    sh = fake.shape
    real = real.flatten()
    fake = fake.flatten()
    v, x = np.histogram(real, bins=nbins)
    v2, x2 = np.histogram(fake, bins=nbins)
    c = np.cumsum(v) / np.sum(v)
    c2 = np.cumsum(v2) / np.sum(v2)
    ax = np.cumsum(np.diff(x)) + np.min(x)
    ax2 = np.cumsum(np.diff(x2)) + np.min(x2)
    N = len(fake)
    res = []
    print('This implementation is slow...')
    for index, batch in enumerate(np.split(fake, N // bs)):
        if np.mod(index, N // bs // 100) == 0:
            print('{}% done'.format(bs * index * 100 // N))
        ind1 = np.argmin(np.abs(np.expand_dims(ax2, axis=0) - np.expand_dims(batch, axis=1)), axis=1)
        ind2 = np.argmin(np.abs(np.expand_dims(c, axis=0) - np.expand_dims(c2[ind1], axis=1)), axis=1)
        res.append(ax[ind2])
    return np.reshape(np.concatenate(res), sh)