"""List of metrics for the GAN models."""

from .core import StatisticalMetric, Statistic, StatisticalMetricLim, MetricSum
import numpy as np
from scipy import stats
from .stats import mass_hist
from .stats import peak_count_hist as peak_hist
from .stats import power_spectrum_batch_phys as psd


def mean(x):
    """Compute the mean."""
    return np.mean(x.flatten())


def var(x):
    """Compute the variance."""
    return np.var(x.flatten())


def min(x):
    """Compute the minmum."""
    return np.min(x.flatten())


def max(x):
    """Compute the maximum."""
    return np.max(x.flatten())


def kurtosis(x):
    """Compute the kurtosis."""
    return stats.kurtosis(x.flatten())


def skewness(x):
    """Compute the skewness."""
    return stats.skew(x.flatten())


def median(x):
    """Compute the median."""
    return np.median(x.flatten())


def gan_stat_list(subname=''):
    """Return a list of statistic for a GAN."""
    stat_list = []

    # While the code of the first statistic might not be optimal,
    # it is likely to be negligible compared to all the rest.

    # The order the statistic is important. If it is changed, the test cases
    # need to be adapted accordingly.

    if not (subname == ''):
        subname = '_' + subname

    stat_list.append(Statistic(mean, name='mean'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(var, name='var'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(min, name='min'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(max, name='max'+subname, group='descriptives', stype=0))    
    stat_list.append(Statistic(kurtosis, name='kurtosis'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(median, name='median'+subname, group='descriptives', stype=0))

    return stat_list


def gan_metric_list(recompute_real=False):
    """Return a metric list for a GAN."""

    stat_list = gan_stat_list()
    metric_list = [StatisticalMetric(statistic=stat, recompute_real=False) for stat in stat_list]
    
    metric_list_t = []
    metric_list_t.append(StatisticalMetricLim(Statistic(mass_hist, name='mass_histogram_log', group='nomap'), log=True, recompute_real=recompute_real, stype=3))
    metric_list_t.append(StatisticalMetricLim(Statistic(peak_hist, name='peak_histogram_log', group='nomap'), log=True, recompute_real=recompute_real, stype=3))
    metric_list_t.append(StatisticalMetric(Statistic(psd_mean, name='psd_log', group='nomap'), log=True, recompute_real=recompute_real, stype=3))
    metric_list.append(MetricSum(metric_list_t, name ='global_score_log', group='nomap', recompute_real=recompute_real, stype=0))
   
    metric_list_t = []
    metric_list_t.append(StatisticalMetricLim(Statistic(mass_hist, name='mass_histogram', group='nomap'), log=False, recompute_real=recompute_real, stype=0))
    metric_list_t.append(StatisticalMetricLim(Statistic(peak_hist, name='peak_histogram', group='nomap'), log=False, recompute_real=recompute_real, stype=0))
    metric_list_t.append(StatisticalMetric(Statistic(psd_mean, name='psd', group='nomap'), log=False, recompute_real=recompute_real, stype=0))
    metric_list.append(MetricSum(metric_list_t, name ='global_score', group='nomap', recompute_real=recompute_real, stype=0))


    return metric_list


def psd_mean(*args,**kwargs):
    s = psd(*args, **kwargs)
    return (np.mean(s[0], axis=0), *s[1:])


def cosmo_metric_list(recompute_real=False):
    metric_list = []
    metric_list.append(StatisticalMetricLim(Statistic(mass_hist, name='mass_histogram', group='cosmology'), log=True, recompute_real=recompute_real, stype=3))
    metric_list.append(StatisticalMetricLim(Statistic(peak_hist, name='peak_histogram', group='cosmology'), log=True, recompute_real=recompute_real, stype=3))
    metric_list.append(StatisticalMetric(Statistic(psd_mean, name='psd', group='cosmology'), log=True, recompute_real=recompute_real, stype=3))
    # metric_list.append(MetricSum(metric_list[:3], name ='global_score', group='cosmology', recompute_real=recompute_real, stype=0))

    metric_list = [MetricSum(metric_list, name ='global_score', group='cosmology', recompute_real=recompute_real, stype=0)]

    # TODO: wasserstein

    return metric_list


def global_score(recompute_real=False):
    return cosmo_metric_list(recompute_real)[-1]
