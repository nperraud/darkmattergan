"""List of metrics for the GAN models."""

from .core import StatisticalMetric, Statistic
import numpy as np
from scipy import stats


def gan_stat_list(subname=''):
    """Return a list of statistic for a GAN."""
    stat_list = []

    # While the code of the first statistic might not be optimal,
    # it is likely to be negligible compared to all the rest.

    # The order the statistic is important. If it is changed, the test cases
    # need to be adapted accordingly.

    if not (subname == ''):
        subname = '_' + subname

    # Mean
    def mean(x):
        return np.mean(x.flatten())

    stat_list.append(Statistic(mean, 'mean'+subname, 'descriptives'))

    # var
    def var(x):
        return np.var(x.flatten())

    stat_list.append(Statistic(var, 'var'+subname, 'descriptives'))

    # min
    def min(x):
        return np.min(x.flatten())

    stat_list.append(Statistic(min, 'min'+subname, 'descriptives'))

    # max
    def max(x):
        return np.max(x.flatten())

    stat_list.append(Statistic(max, 'max'+subname, 'descriptives'))

    # kurtosis
    def kurtosis(x):
        return stats.kurtosis(x.flatten())

    stat_list.append(Statistic(kurtosis, 'kurtosis'+subname, 'descriptives'))

    # skewness
    def skewness(x):
        return stats.skew(x.flatten())

    stat_list.append(Statistic(skewness, 'kurtosis'+subname, 'descriptives'))

    # median
    def median(x):
        return np.median(x.flatten())

    stat_list.append(Statistic(median, 'median'+subname, 'descriptives'))

    return stat_list


def gan_metric_list():
    """Return a metric list for a GAN."""

    stat_list = gan_stat_list()
    metric_list = [StatisticalMetric(statistic=stat) for stat in stat_list]
    return metric_list


def cosmo_stat_list():
    # mass_hist
    # peak_hist
    # psd

    # Peak stats?

    pass


def cosmo_metric_list():
    #cosmo_stat
    # gan_stat
    # wasserstein
    pass
