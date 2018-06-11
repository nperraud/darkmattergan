"""Evaluation module.

This module contains helping functions for the evaluation of the models.
"""

import tensorflow as tf
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


def upscale_image(obj, small=None, num_samples=None, resolution=None, checkpoint=None, sess=None, is_3d=False):
    """Upscale image using the lappachsimple model, or upscale_WGAN_pixel_CNN model.

    For model upscale_WGAN_pixel_CNN, pass num_samples to generate and resolution of the final bigger histogram.
    for model lappachsimple         , pass small.

    3D only works for upscale_WGAN_pixel_CNN.

    This function can be accelerated if the model is created only once.
    """
    # Number of sample to produce
    if small is None:
        if num_samples is None:
            raise ValueError("Both small and num_samples cannot be None")
        else:
            N = num_samples
    else:
        N = small.shape[0]

    # Output dimension of the generator
    soutx, souty = obj.params['image_size'][:2]
    if is_3d:
        soutz = obj.params['image_size'][2]

    if small is not None:
        # Dimension of the low res image
        lx, ly = small.shape[1:3]
        if is_3d:
            lz = small.shape[3]

        # Input dimension of the generator
        sinx = soutx // obj.params['generator']['downsampling'] #32
        siny = souty // obj.params['generator']['downsampling'] #32
        if is_3d:
            sinz = soutz // obj.params['generator']['downsampling'] #32

        # Number of part to be generated
        nx = lx // sinx
        ny = ly // siny
        if is_3d:
            nz = lz // sinz

    else:
        sinx = siny = sinz = None
        if resolution is None:
            raise ValueError("Both small and resolution cannot be None")
        else:
            nx = resolution // soutx
            ny = resolution // souty
            nz = resolution // soutz


    # Final output image
    if sess is None:
        sess = tf.Session()

    obj.load(sess=sess, checkpoint=checkpoint)

    if is_3d:
        output_image = generate_3d_output(sess, obj, N, nx, ny, nz, soutx, souty, soutz, small, sinx, siny, sinz)
    else:
        output_image = generate_2d_output(sess, obj, N, nx, ny, soutx, souty, small, sinx, siny)
        
    return np.squeeze(output_image)


def generate_3d_output(sess, obj, N, nx, ny, nz, soutx, souty, soutz, small, sinx, siny, sinz):
    output_image = np.zeros(
            shape=[N, soutz * nz, souty * ny, soutx * nx, 1], dtype=np.float32)
    output_image[:] = np.nan

    print('Total number of patches = {}*{}*{} = {}'.format(nx, ny, nz, nx*ny*nz) )

    for k in range(nz): # height
        for j in range(ny): # rows
            for i in range(nx): # columns

                # 1) Generate the border
                border = np.zeros([N, soutz, souty, soutx, 7])

                if j: # one row above, same height
                    border[:, :, :, :, 0:1] = output_image[:, 
                                                            k * soutz:(k + 1) * soutz,
                                                            (j - 1) * souty:j * souty, 
                                                            i * soutx:(i + 1) * soutx, 
                                                        :]
                if i: # one column left, same height
                    border[:, :, :, :, 1:2] = output_image[:,
                                                            k * soutz:(k + 1) * soutz,
                                                            j * souty:(j + 1) * souty, 
                                                            (i - 1) * soutx:i * soutx, 
                                                        :]
                if i and j: # one row above, one column left, same height
                    border[:, :, :, :, 2:3] = output_image[:,
                                                            k * soutz:(k + 1) * soutz,
                                                            (j - 1) * souty:j * souty, 
                                                            (i - 1) * soutx:i * soutx, 
                                                        :]
                if k: # same row, same column, one height above
                    border[:, :, :, :, 3:4] = output_image[:,
                                                            (k - 1) * soutz:k * soutz,
                                                            j * souty:(j + 1) * souty, 
                                                            i * soutx:(i + 1) * soutx, 
                                                        :]
                if k and j: # one row above, same column, one height above
                    border[:, :, :, :, 4:5] = output_image[:,
                                                            (k - 1) * soutz:k * soutz,
                                                            (j - 1) * souty:j * souty, 
                                                            i * soutx:(i + 1) * soutx, 
                                                        :]
                if k and i: # same row, one column left, one height above
                    border[:, :, :, :, 5:6] = output_image[:,
                                                            (k - 1) * soutz:k * soutz,
                                                            j * souty:(j + 1) * souty, 
                                                            (i - 1) * soutx:i * soutx, 
                                                        :]
                if k and i and j: # one row above, one column left, one height above
                    border[:, :, :, :, 6:7] = output_image[:,
                                                            (k - 1) * soutz:k * soutz,
                                                            (j - 1) * souty:j * souty, 
                                                            (i - 1) * soutx:i * soutx, 
                                                        :]


                # 2) Prepare low resolution
                if small is not None:
                    downsampled = small[:, k * sinz:(k + 1) * sinz,
                                           j * siny:(j + 1) * siny,
                                           i * sinx:(i + 1) * sinx,
                                           :]
                    print('downsampled: min={} max={}'.format( np.min(downsampled), np.max(downsampled) ))
                else:
                    downsampled = None

                # 3) Generate the image
                print('Current patch: column={}, row={}, height={}'.format(i+1, j+1, k+1))
                gen_sample, _ = obj.generate(
                    N=N, border=border, downsampled=downsampled, sess=sess)
                print('gen_sample: min={} max={}\n\n'.format( np.min(gen_sample), np.max(gen_sample) ))

                output_image[:, 
                                k * soutz:(k + 1) * soutz,
                                j * souty:(j + 1) * souty, 
                                i * soutx:(i + 1) * soutx, 
                            :] = gen_sample

    return output_image


def generate_2d_output(sess, obj, N, nx, ny, soutx, souty, small, sinx, siny):
    output_image = np.zeros(
            shape=[N, soutx * nx, souty * ny, 1], dtype=np.float32)
    output_image[:] = np.nan

    for j in range(ny):
        for i in range(nx):
            # 1) Generate the border
            border = np.zeros([N, soutx, souty, 3])
            if i:
                border[:, :, :, :1] = output_image[:, 
                                                    (i - 1) * soutx:i * soutx, 
                                                    j * souty:(j + 1) * souty, 
                                                :]
            if j:
                border[:, :, :, 1:2] = output_image[:, 
                                                    i * soutx:(i + 1) * soutx, 
                                                    (j - 1) * souty:j * souty, 
                                                :]
            if i and j:
                border[:, :, :, 2:3] = output_image[:, 
                                                    (i - 1) * soutx:i * soutx, 
                                                    (j - 1) * souty:j * souty, 
                                                :]


            if small is not None:
                # 2) Prepare low resolution
                downsampled = np.expand_dims(small[:N][:, i * sinx:(i + 1) * sinx, j * siny:(j + 1) * siny], 3)
            else:
                downsampled = None

            # 3) Generate the image
            print('Current patch: column={}, row={}'.format(j+1, i+1))
            gen_sample, _ = obj.generate(
                N=N, border=border, downsampled=downsampled, sess=sess)

            output_image[:, 
                            i * soutx:(i + 1) * soutx, 
                            j * souty:(j + 1) * souty, 
                        :] = gen_sample

    return output_image


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