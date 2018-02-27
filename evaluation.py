import pickle
import numpy as np
import metrics
import plot
import matplotlib.pyplot as plt
from model import *
from gan import *


def load_gan(pathgan, GANtype=CosmoGAN):
    # try:
    #     del(params)
    #     del(obj)
    # except:
    #     pass

    with open(pathgan + 'params.pkl', 'rb') as f:
        params = pickle.load(f)
    obj = GANtype(params)

    return obj


def generate_samples(N, obj, pathgan, checkpoint=None, y=None):
    if checkpoint is None:
        gen_sample, gen_sample_raw = obj.generate(N=N, y=y)
    else:
        file_name = pathgan + obj.model_name + '-' + checkpoint

        gen_sample, gen_sample_raw = obj.generate(
            N=N, y=y, file_name=file_name)

    gen_sample = np.squeeze(gen_sample)
    gen_sample_raw = np.squeeze(gen_sample_raw)
    return gen_sample, gen_sample_raw


def compute_and_plot_psd(raw_images, gen_sample_raw, display=True):
    psd_real, x = metrics.power_spectrum_batch_phys(X1=raw_images)
    psd_real_mean = np.mean(psd_real, axis=0)

    psd_gen, x = metrics.power_spectrum_batch_phys(X1=gen_sample_raw)
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

    y_real, y_fake, x = metrics.peak_count_hist(raw_images, gen_sample_raw)
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
        ax.plot(np.exp(x), y_fake, label="Fake", color='r', **linestyle)
        ax.plot(np.exp(x), y_real, label="Real", color='b', **linestyle)

        # ax.set_ylim(bottom=0.1)
        ax.title.set_text("Peak count\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()
    return l2, logel2, l1, logel1


def compute_and_plot_mass_hist(raw_images, gen_sample_raw, display=True):

    y_real, y_fake, x = metrics.mass_hist(raw_images, gen_sample_raw)
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
        ax.plot(np.exp(x), y_fake, label="Fake", color='r', **linestyle)
        ax.plot(np.exp(x), y_real, label="Real", color='b', **linestyle)

        # ax.set_ylim(bottom=0.1)
        ax.title.set_text("Mass histogram\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()
    return l2, logel2, l1, logel1
