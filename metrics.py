import numpy as np
from scipy import stats
import power_spectrum_phys as ps
import scipy.ndimage.filters as filters
from scipy import ndimage
import tensorflow as tf
import utils
import functools
import multiprocessing as mp



def build_metrics_summaries(fake, fake_raw, real, real_raw, batch_size):
    m = dict()

    m['descriptives'] = tf.placeholder(tf.float64, shape=[2, 5], name="DescriptiveStatistics")

    tf.summary.scalar("descriptives/mean_Fake", m['descriptives'][0, 0], collections=['metrics'])
    tf.summary.scalar("descriptives/var_Fake", m['descriptives'][0, 1], collections=['metrics'])
    tf.summary.scalar("descriptives/min_Fake", m['descriptives'][0, 2], collections=['metrics'])
    tf.summary.scalar("descriptives/max_Fake", m['descriptives'][0, 3], collections=['metrics'])
    tf.summary.scalar("descriptives/median_Fake", m['descriptives'][0, 4], collections=['metrics'])

    tf.summary.scalar("descriptives/mean_Real", m['descriptives'][1, 0], collections=['metrics'])
    tf.summary.scalar("descriptives/var_Real", m['descriptives'][1, 1], collections=['metrics'])
    tf.summary.scalar("descriptives/min_Real", m['descriptives'][1, 2], collections=['metrics'])
    tf.summary.scalar("descriptives/max_Real", m['descriptives'][1, 3], collections=['metrics'])
    tf.summary.scalar("descriptives/median_Real", m['descriptives'][1, 4], collections=['metrics'])

    m['ps_comp'] = tf.placeholder(tf.float64, name="ps_comp")
    #m['ps_array'] = tf.placeholder(tf.float64, shape=[2, None], name="ps_array")

    tf.summary.scalar("PSD/L2_Fake-Real", m['ps_comp'], collections=['metrics'])
    #tf.summary.histogram("metrics/Mean of log PSD Fake", m['ps_array'][0, :], collections=['metrics'])
    #tf.summary.histogram("metrics/Mean of log PSD Real", m['ps_array'][1, :], collections=['metrics'])

    m['peak_fake'] = tf.placeholder(tf.float64, shape=[None], name="peak_fake")
    m['peak_real'] = tf.placeholder(tf.float64, shape=[None], name="peak_real")
    tf.summary.histogram("Peaks/Fake_log", m['peak_fake'], collections=['metrics'])
    tf.summary.histogram("Peaks/Real_log", m['peak_real'], collections=['metrics'])

    m['distance_peak_comp'] = tf.placeholder(tf.float64, name='distance_peak_comp')
    m['distance_peak_fake'] = tf.placeholder(tf.float64, name='distance_peak_fake')
    m['distance_peak_real'] = tf.placeholder(tf.float64, name='distance_peak_real')

    tf.summary.scalar("Peaks/Ch2_Fake-Real", m['distance_peak_comp'], collections=['metrics'])
    tf.summary.scalar("Peaks/Ch2_Fake-Fake", m['distance_peak_fake'], collections=['metrics'])
    tf.summary.scalar("Peaks/Ch2_Real-Real", m['distance_peak_real'], collections=['metrics'])

    m['cross_ps'] = tf.placeholder(tf.float64, shape = [3],  name='distance_rms_comp')

    tf.summary.scalar("PSD/Cross_Fake-Real", m['cross_ps'][0], collections=['metrics'])
    tf.summary.scalar("PSD/Cross_Fake-Fake", m['cross_ps'][1], collections=['metrics'])
    tf.summary.scalar("PSD/Cross_Real-Real", m['cross_ps'][2], collections=['metrics'])

    if False:



        m['distance_rms_comp'] = tf.placeholder(tf.float64, name='distance_rms_comp')
        m['distance_rms_fake'] = tf.placeholder(tf.float64, name='distance_rms_fake')
        m['distance_rms_real'] = tf.placeholder(tf.float64, name='distance_rms_real')

        tf.summary.scalar("metrics/RMS Distance Fake-Real", m['distance_rms_comp'], collections=['metrics'])
        tf.summary.scalar("metrics/RMS Distance Fake-Fake", m['distance_rms_fake'], collections=['metrics'])
        tf.summary.scalar("metrics/RMS Distance Real-Real", m['distance_rms_real'], collections=['metrics'])

        m['ps_indep_comp'] = tf.placeholder(tf.float64, name='ps_indep_comp')
        m['ps_indep_fake'] = tf.placeholder(tf.float64, name='ps_indep_fake')
        m['ps_indep_real'] = tf.placeholder(tf.float64, name='ps_indep_real')

        tf.summary.scalar("metrics/Independence PSD Fake-Real", m['ps_indep_comp'], collections=['metrics'])
        tf.summary.scalar("metrics/Independence PSD Fake-Fake", m['ps_indep_fake'], collections=['metrics'])
        tf.summary.scalar("metrics/Independence PSD Real-Real", m['ps_indep_real'], collections=['metrics'])

    tf.summary.histogram("Pixel/Fake", fake, collections=['metrics'])
    tf.summary.histogram("Pixel/Real", real, collections=['metrics'])

    # To clip before using this summaries
    clip_max = 1e10
    tf.summary.histogram("Pixel/Fake_Raw", tf.clip_by_value(fake_raw, 0, clip_max), collections=['metrics'])
    tf.summary.histogram("Pixel/Real_Raw", tf.clip_by_value(real_raw, 0, clip_max), collections=['metrics'])

    return m


def calculate_metrics(fake, real,params, tensorboard=True, box_l=100/0.7):
    real = utils.makeit_square(real)
    fake = utils.makeit_square(fake)

    m = {}
    if params['clip_max_real']:
        clip_max = real.ravel().max()
    else:
        clip_max = 1e10

    fake = np.clip(np.nan_to_num(fake), params['log_clip'], clip_max)
    real = np.clip(np.nan_to_num(real), params['log_clip'], clip_max)

    if params['sigma_smooth'] is not None:
        fake = ndimage.gaussian_filter(fake, sigma=params['sigma_smooth'])
        real = ndimage.gaussian_filter(real, sigma=params['sigma_smooth'])

    # Descriptive Stats

    if tensorboard:
        descr_fake = np.array([describe(x) for x in fake])
        descr_real = np.array([describe(x) for x in real])

        m['descriptives'] = np.stack((np.mean(descr_fake, axis=0), np.mean(descr_real, axis=0)))

        del descr_fake, descr_real
    else:
        bins_o = np.linspace(0, real.max(), real.max() + 2, dtype=np.float64)
        m['pix_rxo'], m['pix_ryo'] = histogram(x=real, bins=bins_o, probability=False)
        m['pix_fxo'], m['pix_fyo'] = histogram(x=fake, bins=bins_o, probability=False)


        real_t = inputs.simple_numpy(real, k=params['k'])
        fake_t = inputs.simple_numpy(fake, k=params['k'])
        bins_t = inputs.simple_numpy(bins_o, k=params['k'])
        m['pix_rxt'], m['pix_ryt'] = histogram(x=real_t, bins=bins_t, probability=False)
        m['pix_fxt'], m['pix_fyt'] = histogram(x=fake_t, bins=bins_t, probability=False)

        print(' PIXELS chi2dist real-fake {:.3f}'.format(chi2_distance(real, fake)))

        del real_t, fake_t

    ps_fake, k = power_spectrum_batch_phys(X1=fake, box_l=box_l)
    ps_real, _ = power_spectrum_batch_phys(X1=real, box_l=box_l)

    if tensorboard:

        m['ps_comp'] = (np.mean(ps_fake) - np.mean(ps_real)) ** 2
    else:
        m['ps_fake'] = ps_fake
        m['ps_real'] = ps_real
        m['k'] = k
        print(' PSD L2 real-fake {:.3f}'.format((np.mean(ps_fake) - np.mean(ps_real)) ** 2))
    del ps_fake, ps_real

    # Distance of Peak Histogram

    peak_fake = np.array([peak_count(x, neighborhood_size=5, threshold=0) for x in fake])
    peak_real = np.array([peak_count(x, neighborhood_size=5, threshold=0) for x in real])

    if tensorboard:
        m['peak_fake'] = np.log(np.hstack(peak_fake))
        m['peak_real'] = np.log(np.hstack(peak_real))

        m['distance_peak_comp'] = distance_chi2_peaks(peak_fake, peak_real)
        m['distance_peak_fake'] = distance_chi2_peaks(peak_fake, peak_fake)
        m['distance_peak_real'] = distance_chi2_peaks(peak_real, peak_real)
    else:

        peak_all_ = np.hstack((np.hstack(peak_fake), np.hstack(peak_real)))
        n_bins = 20
        bins = np.logspace(np.log(peak_all_.min()), np.log(peak_all_.max()), n_bins, dtype=np.float64, base=np.e)
        del peak_all_

        m['peak_rx'], m['peak_ry'] = histogram(x = np.array(peak_real), bins=bins, probability=False)
        m['peak_fx'], m['peak_fy'] = histogram(x = np.array(peak_fake), bins=bins, probability=False)
        peak_real_ = np.array([item for sublist in peak_real for item in sublist])
        peak_fake_ = np.array([item for sublist in peak_fake for item in sublist])
        print(' PEAK chi2 dist real-fake {:.3f}'.format(chi2_distance(peak_real_, peak_fake_)))

        del peak_fake_, peak_real_

    del peak_real, peak_fake

    # Measure Cross PS

    index = np.random.choice(real.shape[0], min(50, real.shape[0]),
                             replace=False)  # computing all pairwise comparisons is expensive
    cross_rf, _ = power_spectrum_batch_phys(X1=real[index], X2=fake[index], box_l=box_l)
    cross_ff, _ = power_spectrum_batch_phys(X1= fake[index], X2 = fake[index], box_l=box_l)
    cross_rr, _ = power_spectrum_batch_phys(X1= real[index], X2 = real[index], box_l=box_l)

    if tensorboard:
        m['cross_ps'] = [np.mean(cross_rf),np.mean(cross_ff), np.mean(cross_rr)]
    else:
        m['cross_rf'] = cross_rf
        m['cross_ff'] = cross_ff
        m['cross_rr'] = cross_rr

    del cross_ff, cross_rf, cross_rr

    # Print the results
    if tensorboard:
        print(
        " [*] [Fake, Real] Min [{:.3f}, {:.3f}],\tMedian [{:.3f},{:.3f}],\tMean [{:.3E},{:.3E}],\t Max [{:.3E},{:.3E}],\t Var [{:.3E},{:.3E}]".format(
            m['descriptives'][0, 2], m['descriptives'][1, 2], m['descriptives'][0, 4], m['descriptives'][1, 4],
            m['descriptives'][0, 0], m['descriptives'][1, 0], m['descriptives'][0, 3], m['descriptives'][1, 3],
            m['descriptives'][0, 1], m['descriptives'][1, 1]))

        print(
            " [*] [Comp, Fake, Real] PeakDistance:[{:.3f}, {:.3f}, {:.3f}]\tCrossPS:[{:.3f}, {:.3f}, {:.3f}]\tPSD_Diff:{:.3f}".format(
                m['distance_peak_comp'],m['distance_peak_fake'], m['distance_peak_real'],
                m['cross_ps'][0], m['cross_ps'][1], m['cross_ps'][2], m['ps_comp']))

    return m


# ## Functions with Power Spectrum
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



def wrapper_func(x, bin_k = 50, box_l = 100/0.7):
    return ps.power_spectrum(field_x=ps.dens2overdens(np.squeeze(x), np.mean(x)), box_l=box_l, bin_k=bin_k)[0]

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

    if X2 is None:

        # # Pythonic version
        # over_dens = [ps.dens2overdens(x.reshape(s,s), np.mean(x)) for x in X1]
        # result = np.array([ps.power_spectrum(field_x= x, box_l=box_l, bin_k = bin_k )[0] for x in over_dens])
        # del over_dens
        # Make it multicore...
        # num_cores = pathos.pp.cpu_count()
        # with pathos.pools.ProcessPool(processes=num_cores-1) as pool:
        #     result = np.array(pool.map(functools.partial(wrapper_func, box_l=box_l, bin_k=bin_k), X1))
        # num_workers = mp.cpu_count()-1
        num_workers = 4
        print('Pool with {} workers'.format(num_workers))
        with mp.Pool(processes=num_workers) as pool:
            # over_dens = pool.map(funcA, X1)
            result = np.array(pool.map(functools.partial(wrapper_func, box_l=box_l, bin_k=bin_k), X1))
    else:
        if not(sx == sy):
            X2 = utils.makeit_square(X2)
        self_comp = np.all(X2 == X1)
        _result = []
        for inx, x in enumerate(X1):
            for iny, y in enumerate(X2):
                if (self_comp and (
                    inx < iny)) or not self_comp:  # if it is a comparison with it self only do the low triangular matrix
                    over_dens_x = ps.dens2overdens(x.reshape(sx,sy))
                    over_dens_y = ps.dens2overdens(y.reshape(sx,sy))

                    _result.append(ps.power_spectrum(field_x=over_dens_x, box_l=box_l,bin_k=bin_k, field_y=over_dens_y)[0])
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

    if len(X.shape) > 2:
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
    for x in im1:
        for y in im2:
            distance.append(chi2_distance(x, y, bins=bins, range=range, **kwargs))

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
    # e = psd_real - psd_gen
    # l2 = np.mean(e*e)
    # loge = 10*(np.log10(psd_real) - np.log10(psd_gen))
    # logel2 = np.mean(loge*loge)
    return diff_psd(psd_real, psd_gen)

def diff_psd(psd_real_mean, psd_gen_mean):
    e = psd_real_mean - psd_gen_mean
    l2 = np.mean(e*e)
    l1 = np.mean(np.abs(e))
    loge = 10*(np.log10(psd_real_mean+1e-5) - np.log10(psd_gen_mean+1e-5))
    logel2 = np.mean(loge*loge)
    logel1 = np.mean(np.abs(loge))
    return l2, logel2, l1, logel1
