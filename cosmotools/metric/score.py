"""Compute scores for the Nbody 3D histogram dataset. This score can be used for 2D or 3D images.

    Warnings:
    1)  The parameters used in this script are fitted to the data present at:
        https://zenodo.org/record/1464832
        It is likely to give strange results for other datasets.
    2)  The number obtained for different sizes are not comparables, i.e, the score is highly dependant on the size of the image.
    
"""


from .stats import mass_hist, peak_count_hist
from .stats import power_spectrum_batch_phys as psd
import numpy as np
from gantools.metric.fd import compute_fd


def score_histogram(real, fake):
    """Compute the histogram score from real and fake image."""
    
    assert(np.squeeze(real).shape==np.squeeze(fake).shape)
    # A) Define the limit for the histogram. We avoid some corner cases
    # raw_max = np.max(backward(dataset.get_all_data().flatten()))
    raw_max = 250884    
    lim = [np.log10(2), np.log10(raw_max/3)]
    
    # B) Compute the histograms
    y_real, x_real, lim_real = mass_hist(real, log=True, mean=False, lim=lim)
    y_fake, x_fake, lim_fake = mass_hist(fake, log=True, mean=False, lim=lim)
    
    # C) Do some testing
    np.testing.assert_allclose(x_real,x_fake)
    np.testing.assert_allclose(lim_real,lim_fake)
    assert(np.sum(y_real)==np.sum(y_fake))
    
    # D) Normalize to get value between 0 an 1...
    npix = np.prod(real.shape[1:])
    y_real = y_real/npix
    y_fake = y_fake/npix

    # E) Define a lower limit and compute the log
    # nsamples = min(real.shape[0], fake.shape[0])
    low_lim = 1/npix
    y_real_log = np.log10(y_real+low_lim)
    y_fake_log = np.log10(y_fake+low_lim)
    
    # F) Compute the fd
    d = compute_fd(y_real_log, y_fake_log)
    
    return 1/d


def score_peak_histogram(real, fake):
    """Compute the peak histogram score from real and fake image."""
    
    assert(np.squeeze(real).shape==np.squeeze(fake).shape)
    # A) Define the limit for the histogram. We avoid some corner cases
    # _, _, lim_real = peak_count_hist(backward(dataset.get_samples(dataset.N)), log=True, mean=False)
    # print(lim_real)
    lim = [1.5, 13]
    
    # B) Compute the histograms
    y_real, x_real, lim_real = peak_count_hist(real, log=True, mean=False, lim=lim)
    y_fake, x_fake, lim_fake = peak_count_hist(fake, log=True, mean=False, lim=lim)
    
    # C) Do some testing
    np.testing.assert_allclose(x_real,x_fake)
    np.testing.assert_allclose(lim_real,lim_fake)
    # This time the histogram is computed over the peak and their number may vary depending on the image
    # assert(np.sum(y_real)==np.sum(y_fake))
    
    # D) Normalize to get value between 0 an 1...
    npix = np.prod(real.shape[1:])
    y_real = y_real/npix
    y_fake = y_fake/npix

    
    # E) Define a lower limit and compute the log
    # nsamples = min(real.shape[0], fake.shape[0])
    low_lim = 1/npix
    y_real_log = np.log10(y_real+low_lim)
    y_fake_log = np.log10(y_fake+low_lim)
    
    # F) Compute the fd
    d = compute_fd(y_real_log, y_fake_log)
    
    return 1/d


def score_psd(real, fake):
    
    
    assert(np.squeeze(real).shape==np.squeeze(fake).shape)

    multiply=False
    box_ll=350 #(5*np.pi/180)
    bin_k=50
    confidence=None
    log_sampling=True
    cut=None

    psd_real, x_real = psd(X1=real, multiply=multiply, bin_k=bin_k, box_ll=box_ll, log_sampling=log_sampling, cut=cut)
    psd_gen, x_fake = psd(X1=fake, multiply=multiply, bin_k=bin_k, box_ll=box_ll, log_sampling=log_sampling, cut=cut)
    np.testing.assert_almost_equal(x_real, x_fake)
    n_fac = np.max(np.mean(psd_real, axis=0))
    y_real = psd_real/n_fac
    y_fake = psd_gen/n_fac
    low_lim = 0
    y_real_log = np.log10(y_real+low_lim)
    y_fake_log = np.log10(y_fake+low_lim)
    
    # F) Compute the fd
    d = compute_fd(y_real_log, y_fake_log)
    
    return 1/d