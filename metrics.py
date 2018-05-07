"""This module contains the different metric functions."""

import numpy as np
from scipy import stats
import power_spectrum_phys as ps
import scipy.ndimage.filters as filters
import itertools
import utils
import functools
import multiprocessing as mp


def wrapper_func(x, bin_k=50, box_l=100 / 0.7):
    tmp = ps.dens2overdens(np.squeeze(x), np.mean(x))
    return ps.power_spectrum(field_x=tmp, box_l=box_l, bin_k=bin_k)[0]

def wrapper_func_cross(a,
                       X2,
                       self_comp,
                       sx,
                       sy,
                       sz=None,
                       bin_k=50,
                       box_l=100 / 0.7,
                       is_3d=False):
    inx, x = a
    _result = []
    for iny, y in enumerate(X2):
        # if it is a comparison with it self only do the low triangular matrix
        if (self_comp and (inx < iny)) or not self_comp:
            if is_3d:
                over_dens_x = ps.dens2overdens(x.reshape(sx, sy, sz))
                over_dens_y = ps.dens2overdens(y.reshape(sx, sy, sz))
            else:
                over_dens_x = ps.dens2overdens(x.reshape(sx, sy))
                over_dens_y = ps.dens2overdens(y.reshape(sx, sy))
            tmp = ps.power_spectrum(
                field_x=over_dens_x,
                box_l=box_l,
                bin_k=bin_k,
                field_y=over_dens_y)[0]
            # Nati: Why is there a [0] here. There is probably a good reason...
            _result.append(tmp)
    return _result

def power_spectrum_batch_phys(X1,
                              X2=None,
                              bin_k=50,
                              box_l=100 / 0.7,
                              is_3d=False,
                              remove_nan=True):
    '''
    Calculates the 1-D PSD of a batch of variable size
    :param batch:
    :param size_image:
    :return:
    '''
    sx, sy = X1[0].shape[0], X1[0].shape[1]
    sz = None
    if is_3d:
        sz = X1[0].shape[2]

    if not (sx == sy):
        X1 = utils.makeit_square(X1)
        s = X1[0].shape[0]
    else:
        s = sx
        # ValueError('The image need to be squared')

    if is_3d:
        _, k = ps.power_spectrum(
            field_x=X1[0].reshape(s, s, s), box_l=box_l, bin_k=bin_k)
    else:
        _, k = ps.power_spectrum(
            field_x=X1[0].reshape(s, s), box_l=box_l, bin_k=bin_k)

    num_workers = mp.cpu_count() - 1
    # if num_workers == 23:
    #     # Small hack for CSCS
    #     num_workers = 2
    #     print('CSCS: Pool reduced!')
    # print('Pool with {} workers'.format(num_workers))
    with mp.Pool(processes=num_workers) as pool:
        if X2 is None:
            # # Pythonic version
            # over_dens = [ps.dens2overdens(x.reshape(s, s), np.mean(x)) for x in X1]
            # result = np.array([
            #     ps.power_spectrum(field_x=x, box_l=box_l, bin_k=bin_k)[0]
            #     for x in over_dens
            # ])
            # del over_dens

            # Make it multicore...
            func = functools.partial(wrapper_func, box_l=box_l, bin_k=bin_k)
            result = np.array(pool.map(func, X1))

        else:
            if not (sx == sy):
                X2 = utils.makeit_square(X2)
            self_comp = np.all(X2 == X1)
            _result = []
            # for inx, x in enumerate(X1):
            #     # for iny, y in enumerate(X2):
            #     #     # if it is a comparison with it self only do the low
            #     #     # triangular matrix
            #     #     if (self_comp and (inx < iny)) or not self_comp:
            #     #         over_dens_x = ps.dens2overdens(x.reshape(sx, sy))
            #     #         over_dens_y = ps.dens2overdens(y.reshape(sx, sy))
            #     _result += wrapper_func_cross(
            #         (inx, x), X2, self_comp, sx, sy, bin_k=50, box_l=100/0.7)
            func = functools.partial(
                wrapper_func_cross,
                X2=X2,
                self_comp=self_comp,
                sx=sx,
                sy=sy,
                sz=sz,
                bin_k=50,
                box_l=100 / 0.7,
                is_3d=is_3d)
            _result = pool.map(func, enumerate(X1))
            _result = list(itertools.chain.from_iterable(_result))
            result = np.array(_result)

    if remove_nan:
        # Some frequencies are not defined, remove them
        freq_index = ~np.isnan(result).any(axis=0)
        result = result[:, freq_index]
        k = k[freq_index]

    return result, k


def histogram(x, bins, probability=True):
    if x.ndim > 2:
        x = np.reshape(x, [int(x.shape[0]), -1])

    edges = np.histogram(x[0].ravel(), bins=bins)[1][:-1]

    counts = np.array([np.histogram(y, bins=bins)[0] for y in x])

    if probability:
        density = counts * 1.0 / np.sum(counts, axis=1, keepdims=True)
    else:
        density = counts

    return edges, density


def peak_count(X, neighborhood_size=5, threshold=0):
    '''
    :param X: numpy array shape [size_image,size_image] or as a vector
    :param neighborhood_size: size of the local neighborhood that should be filtered
    :param threshold: minimum distance betweent the minimum and the maximum to be considered a local maximum
                      Helps remove noise peaks
    :return: number of peaks found in the array (int)
    '''
    if len(X.shape) == 1:
        n = int(X.shape[0]**0.5)
    else:
        n = X.shape[0]
    try:
        X = X.reshape(n, n)
    except:
        try:
            X = X.reshape(n, n, n)
        except:
            raise Exception(" [!] Image not squared ")

    # PEAK COUNTS
    data_max = filters.maximum_filter(X, neighborhood_size)
    maxima = (X == data_max)
    data_min = filters.minimum_filter(X, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    return np.extract(maxima, X)


def describe(X):
    # DESCRIPTIVE STATS

    if len(X.shape) > 1:
        X = X.reshape(-1)

    _, range, mean, var, skew, kurt = stats.describe(X)

    return mean, var, range[0], range[1], np.median(X)


def chi2_distance(peaksA, peaksB, eps=1e-10, **kwargs):
    histA, _ = np.histogram(peaksA, **kwargs)
    histB, _ = np.histogram(peaksB, **kwargs)

    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b)**2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d


def distance_chi2_peaks(im1, im2, bins=100, range=[0, 2e5], **kwargs):
    if len(im1.shape) > 2:
        X = im1.reshape(-1)
    distance = []

    num_workers = mp.cpu_count() - 1
    # if num_workers == 23:
    #     # Small hack for CSCS
    #     num_workers = 2
    #     print('CSCS: Pool reduced!')
    # print('Pool with {} workers'.format(num_workers))
    with mp.Pool(processes=num_workers) as pool:
        for x in im1:
            # for y in im2:
            #     distance.append(chi2_distance(x, y, bins=bins, range=range, **kwargs))
            distance.append(
                np.array(
                    pool.map(
                        functools.partial(
                            chi2_distance,
                            peaksB=x,
                            bins=bins,
                            range=range,
                            **kwargs), im2)))
    return np.mean(np.array(distance))


def distance_rms(im1, im2):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    distance = []
    for x in im1:
        for y in im2:
            distance.append(np.linalg.norm(x - y, 2))

    return np.mean(np.array(distance))


def psd_metric(gen_sample_raw, real_sample_raw):
    psd_gen, _ = power_spectrum_batch_phys(X1=gen_sample_raw)
    psd_real, _ = power_spectrum_batch_phys(X1=real_sample_raw)
    psd_gen = np.mean(psd_gen, axis=0)
    psd_real = np.mean(psd_real, axis=0)
    return diff_vec(psd_real, psd_gen)


def diff_vec(y_real, y_fake):
    e = y_real - y_fake
    l2 = np.mean(e * e)
    l1 = np.mean(np.abs(e))
    loge = 10 * (np.log10(y_real + 1e-2) - np.log10(y_fake + 1e-2))
    logel2 = np.mean(loge * loge)
    logel1 = np.mean(np.abs(loge))
    return l2, logel2, l1, logel1


def peak_count_hist(dat, bins=20, lim=None):
    """Make the histogram of the peak count of data.

    Arguments
    ---------
    dat  : input data (numpy array, first dimension for the sample)
    bins : number of bins for the histogram (default 20)
    lim  : limit for the histogram, if None, then min(peak), max(peak)
    """
    peak = np.array(
        [peak_count(x, neighborhood_size=5, threshold=0) for x in dat])
    peak = np.log(np.hstack(peak)+np.e)
    if lim is None:
        lim = (np.min(peak), np.max(peak))
    y, x = np.histogram(peak, bins=bins, range=lim)
    x = np.exp((x[1:] + x[:-1]) / 2)-np.e
    # Normalization
    y = y / dat.shape[0]
    return y, x, lim


def peak_count_hist_real_fake(real, fake, bins=20, lim=None):
    y_real, x, lim = peak_count_hist(real, bins=bins, lim=lim)
    y_fake, _, _ = peak_count_hist(fake, bins=bins, lim=lim)
    return y_real, y_fake, x


def mass_hist(dat, bins=20, lim=None):
    """Make the histogram of log10(data) data.

    Arguments
    ---------
    dat  : input data
    bins : number of bins for the histogram (default 20)
    lim  : limit for the histogram, if None then min(log10(dat)), max(dat)
    """
    log_data = np.log10(dat.flatten() + 1)
    if lim is None:
        lim = (np.min(log_data), np.max(log_data))
    y, x = np.histogram(log_data, bins=bins, range=lim)
    x = 10**((x[1:] + x[:-1]) / 2) - 1
    y = y / dat.shape[0]
    return y, x, lim


def mass_hist_real_fake(real, fake, bins=20, lim=None):
    if lim is None:
        new_lim = True
    else:
        new_lim = False
    y_real, x, lim = mass_hist(real, bins=bins, lim=lim)
    if new_lim:
        print('Using new limits')
        lim = list(lim)
        lim[1] = lim[1]+1
        y_real, x, lim = mass_hist(real, bins=bins, lim=lim)

    y_fake, _, _ = mass_hist(fake, bins=bins, lim=lim)
    return y_real, y_fake, x


def wasserstein_distance(x, y, w=None, safe=True):
    """Wasserstein distance for 1D vectors."""
    if w is None:
        w = np.arange(x.shape[0])
    weights = np.diff(w)
    if safe:
        assert (x.shape == y.shape == w.shape)
        np.testing.assert_almost_equal(np.sum(x), np.sum(y))
        assert ((x >= 0).all())
        assert ((y >= 0).all())
        assert ((weights >= 0).all())
    cx = np.cumsum(x)[:-1]
    cy = np.cumsum(y)[:-1]
    return np.sum(weights * np.abs(cx - cy)) / (w[-1] - w[0])


def total_stats_error(feed_dict, params=None):

    if params is None:
        w_l2_logpsd = 1
        w_l1_logpsd = 1
        w_l2_logmass = 1
        w_l1_logmass = 1
        w_l2_logpeak = 0.1
        w_l1_logpeak = 0.1
        w_wasserstein_mass = 10
    else:
        raise NotImplementedError('TODO')

    v = 0
    v += w_l2_logpsd * feed_dict['log_l2_psd']
    v += w_l1_logpsd * feed_dict['log_l1_psd']
    v += w_l2_logmass * feed_dict['log_l2_mass_hist']
    v += w_l1_logmass * feed_dict['log_l1_mass_hist']
    v += w_l2_logpeak * feed_dict['log_l2_peak_hist']
    v += w_l1_logpeak * feed_dict['log_l1_peak_hist']
    v += np.log10(w_wasserstein_mass+np.e) * feed_dict['wasserstein_mass_hist']

    return v
