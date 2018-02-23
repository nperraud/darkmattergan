import numpy as np
from scipy import stats
import power_spectrum_phys as ps
import scipy.ndimage.filters as filters
import itertools
import utils
import functools
import multiprocessing as mp
# import pathos



## Functions with Power Spectrum

# def power_spectrum_old(X, Y=None):
#     # FFT on 2D and shift so that low spatial frequencies are in the center.
#     Fx = fftpack.fftshift(fftpack.fft2(X))
#     if Y is None:
#         psd2D = np.abs(Fx **2)
#     else:
#         Fy = fftpack.fftshift(fftpack.fft2(Y))
#         psd2D = np.real(Fx * np.conj(Fy))

#     # Calculate the azimuthally averaged 1D power spectrum
#     psd1D = radialprofile.azimuthalAverage(psd2D)
#     return psd2D, psd1D


# def myParallelProcess(ahugearray):
#     from multiprocessing import Pool, cpu_count
#     from contextlib import closing
#     with closing(Pool(cpu_count()-1)) as p:
#         res = p.imap_unordered(functools.partial(wrapper_func, box_l=box_l, bin_k=bin_k), X1, 100)
#     return res


def wrapper_func(x, bin_k = 50, box_l = 100/0.7):
    return ps.power_spectrum(field_x=ps.dens2overdens(np.squeeze(x), np.mean(x)), box_l=box_l, bin_k=bin_k)[0]

def wrapper_func_cross(a, X2, self_comp, sx, sy, bin_k = 50, box_l = 100/0.7):
    inx, x = a
    _result = []
    for iny, y in enumerate(X2):
        if (self_comp and ( inx < iny)) or not self_comp:  # if it is a comparison with it self only do the low triangular matrix
            over_dens_x = ps.dens2overdens(x.reshape(sx,sy))
            over_dens_y = ps.dens2overdens(y.reshape(sx,sy))

            _result.append(ps.power_spectrum(field_x=over_dens_x, box_l=box_l,bin_k=bin_k, field_y=over_dens_y)[0])
    return _result

def power_spectrum_batch_phys(X1, X2=None, bin_k = 50, box_l = 100/0.7):
    '''
    Calculates the 1-D PSD of a batch of variable size
    :param batch:
    :param size_image:
    :return:
    '''
    sx, sy = X1[0].shape[0], X1[0].shape[1]
    if not(sx == sy):
        X1 = utils.makeit_square(X1)
        s = X1[0].shape[0]
    else:
        s = sx
        # ValueError('The image need to be squared')
    _, k =  ps.power_spectrum(field_x=X1[0].reshape(s,s), box_l=box_l, bin_k=bin_k)

    num_workers = mp.cpu_count()-1
    # if num_workers == 23:
    #     # Small hack for CSCS
    #     num_workers = 2
    #     print('CSCS: Pool reduced!')
    # print('Pool with {} workers'.format(num_workers))
    with mp.Pool(processes=num_workers) as pool:
        if X2 is None:
            # # Pythonic version
            # over_dens = [ps.dens2overdens(x.reshape(s,s), np.mean(x)) for x in X1]
            # result = np.array([ps.power_spectrum(field_x= x, box_l=box_l, bin_k = bin_k )[0] for x in over_dens])
            # del over_dens

            # Make it multicore...       
            result = np.array(pool.map(functools.partial(wrapper_func, box_l=box_l, bin_k=bin_k), X1))

        else:
            if not(sx == sy):
                X2 = utils.makeit_square(X2)
            self_comp = np.all(X2 == X1)
            _result = []
            # for inx, x in enumerate(X1):
            #     # for iny, y in enumerate(X2):
            #     #     if (self_comp and ( inx < iny)) or not self_comp:  # if it is a comparison with it self only do the low triangular matrix
            #     #         over_dens_x = ps.dens2overdens(x.reshape(sx,sy))
            #     #         over_dens_y = ps.dens2overdens(y.reshape(sx,sy))
            #     _result += wrapper_func_cross((inx, x), X2, self_comp, sx, sy, bin_k = 50, box_l = 100/0.7)
            _result = pool.map(functools.partial(wrapper_func_cross, X2=X2, self_comp=self_comp, sx=sx, sy=sy, bin_k = 50, box_l = 100/0.7), enumerate(X1))
            _result = list(itertools.chain.from_iterable(_result))
            #  _result.append(ps.power_spectrum(field_x=over_dens_x, box_l=box_l,bin_k=bin_k, field_y=over_dens_y)[0])
            result = np.array(_result)

    freq_index = ~np.isnan(result).any(axis=0) # Some frequencies are not defined, remove them

    return result[:, freq_index], k[freq_index]

# def power_spectrum_batch(X1, X2=None, mean=True, log=False):
#     '''
#     Calculates the 1-D PSD of a batch of variable size
#     :param batch:
#     :param size_image:
#     :return:
#     '''
#     s = X1[0].shape[0]

#     if X2 is None:
#         ps_ = np.array([power_spectrum_old(x.reshape(s, s))[1] for x in X1])
#     else:
#         self_comp = np.all(X2 == X1)
#         result = []
#         for inx, x in enumerate(X1):
#             for iny, y in enumerate(X2):
#                 if (self_comp and (
#                     inx < iny)) or not self_comp:  # if it is a comparison with it self only do the low triangular matrix
#                     result.append(power_spectrum_old(x.reshape(s, s), y.reshape(s, s))[1])
#         ps_ = np.array(result)

#     ps = ps_[:, ~np.isnan(ps_).any(axis=0)]  # Some frequencies are not defined, remove them

#     if log:
#         ps = np.log(ps)

#     if mean:
#         return np.mean(ps, 0)  # Average per frequency not per sample
#     else:
#         return ps  # check


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
        n = int(X.shape[0] ** 0.5)
    else:
        n = X.shape[0]
    try:
        X = X.reshape(n, n)
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
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d


def distance_chi2_peaks(im1, im2, bins=100, range=[0, 2e5], **kwargs):
    if len(im1.shape) > 2:
        X = im1.reshape(-1)
    distance = []

    num_workers = mp.cpu_count()-1
    # if num_workers == 23:
    #     # Small hack for CSCS
    #     num_workers = 2
    #     print('CSCS: Pool reduced!') 
    # print('Pool with {} workers'.format(num_workers))
    with mp.Pool(processes=num_workers) as pool:
        for x in im1:
            # for y in im2:
            #     distance.append(chi2_distance(x, y, bins=bins, range=range, **kwargs))
            distance.append(np.array(pool.map(functools.partial(chi2_distance, peaksB=x, bins=bins, range=range, **kwargs), im2)))
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
    l2 = np.mean(e*e)
    l1 = np.mean(np.abs(e))
    loge = 10*(np.log10(y_real+1e-2) - np.log10(y_fake+1e-2))
    logel2 = np.mean(loge*loge)
    logel1 = np.mean(np.abs(loge))
    return l2, logel2, l1, logel1


def peak_count_hist(real, fake, bins=20):
    peak_real = np.array([peak_count(x, neighborhood_size=5, threshold=0) for x in real])
    peak_fake = np.array([peak_count(x, neighborhood_size=5, threshold=0) for x in fake])
    peak_real = np.log(np.hstack(peak_real))
    peak_fake = np.log(np.hstack(peak_fake))
    lim = (np.min(peak_real), np.max(peak_real))
    y_real, x = np.histogram(peak_real,bins=bins, range=lim)
    y_fake, _ = np.histogram(peak_fake,bins=bins, range=lim)
    x = (x[1:]+x[:-1])/2
    # Normalization
    y_real = y_real / real.shape[0]
    y_fake = y_fake / fake.shape[0]
    return y_real, y_fake, x



def mass_hist(real, fake, bins=20):
    log_real = np.log(real.flatten()+1)
    log_fake = np.log(fake.flatten()+1)
    lim = (np.min(log_real), np.max(log_real))
    y_real, x = np.histogram(log_real, bins=20, range=lim)
    y_fake, _ = np.histogram(log_fake, bins=20, range=lim)
    x = (x[1:]+x[:-1])/2
    y_real = y_real/real.shape[0]
    y_fake = y_fake/fake.shape[0]
    return y_real, y_fake, x
