"""This module define the base classes for the metrics of GANs."""

import numpy as np
import tensorflow as tf
from gantools.plot.plot_summary import PlotSummary, PlotSummaryLog

class TFsummaryHelper(object):
    """Helper class for tensorflow summaries."""

    def __init__(self, name, group=''):
        """Create a statistical object.

        Arguments
        ---------
        * name: name of the statistic (preferably unique)
        * group: group for the statistic
        """
        self._name = name
        self._group = group
        self._stype = None

    def add_summary(self, stype=0, collections=None):
        """Add a tensorflow summary.

        stype: summary type. 
               * 0 scalar
               * 1 image
               * 2 histogram
               * 3 curves
        """

        name = self.group + '/' + self.name
        self._stype = stype
        if stype == 0:
            self._placeholder = tf.placeholder(tf.float32, name=name)
            tf.summary.scalar(name, self._placeholder, collections=[collections])
        elif stype == 1:
            self._placeholder = tf.placeholder(
                tf.float32, shape=[None, None], name=name)
            tf.summary.image(name, self._placeholder, collections=[collections])
        elif stype == 2:
            self._placeholder = tf.placeholder(tf.float32, shape=[None], name=name)
            tf.summary.histogram(name, self._placeholder, collections=[collections])
        elif stype == 3:
            self._placeholder = tf.placeholder(tf.float32, name=name)
            tf.summary.scalar(name, self._placeholder, collections=[collections])
            if self._log:
                self._plot_summary = PlotSummaryLog(
                    self.name, self.group, collections=[collections])
            else:
                raise NotImplementedError()
        else:
            raise ValueError('Wrong summary type')

    def compute_summary(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class.")

    @property
    def group(self):
        return self._group
    
    @property
    def name(self):
        return self._name

class Statistic(TFsummaryHelper):
    """Base class for a statistic."""

    def __init__(self, func, name, group=''):
        """Create a statistical object.

        Arguments
        ---------
        * func: function to compute the statistic
        """
        super().__init__(name=name, group=group)
        self._func = func

    def compute_summary(self, dat, feed_dict={}):
        feed_dict[self._placeholder] = self(dat)
        return feed_dict

    def __call__(self, *args, **kwargs):
        """Compute the statistic."""
        return self._func(*args, **kwargs)


class Metric(TFsummaryHelper):
    """Base metric class."""

    def __init__(self, name, group='', recompute_real=True):
        """Initialize the statistic.

        Argument
        --------
        * name: name of the statistic (preferably unique)
        * recompute_real: recompute the real statistic (default True)
        """
        super().__init__(name=name, group=group)
        self._preprocessed = False
        self._recompute_real = recompute_real


    def preprocess(self, real):
        """Do the preprocessing.

        This function is designed to do all the precomputation that can be done with the real data.
        If this computaation can be done only once, i.e. the real data is always the same, then set recompute_real to False
        """
        self._preprocessed = True

    def __call__(self, fake, real=None):
        """Compute the metric."""
        if self._recompute_real or ((not self.preprocessed) and real):
            self.preprocess(real)
        elif (not self.preprocessed) and (not real):
            raise ValueError("The real data need to be preprocessed first!")
        return self._compute(fake, real)

    def _compute(self, fake, real):
        """Compute the metric."""
        raise NotImplementedError("This is an abstract class.")

    def compute_summary(self, fake, real, feed_dict={}):
        feed_dict[self._placeholder] = self(fake, real)
        return feed_dict

    @property
    def preprocessed(self):
        """Return True if the preprocessing been done."""
        return self._preprocessed


class StatisticalMetric(Metric):
    """Statistically based metric."""

    def __init__(self, statistic, order=2, log=False, recompute_real=True):
        """Initialize the StatisticalMetric.

        Arguments
        ---------
        * name: name of the metric (preferably unique)
        * statistic: a statistic object
        * order: order of the norm (default 2, Froebenius norm)
        * log: compute the log of the stat before the norm (default False)
        * recompute_real: recompute the real statistic (default True)
        """
        name = statistic.name + '_l' + str(order)
        if log:
            name += 'log'
        super().__init__(name, statistic.group, recompute_real)
        self._order = order
        self._log = log
        self.statistic = statistic
        self._saved_stat = None

    def preprocess(self, real):
        """Compute the statistic on the real data."""
        super().preprocess(real)
        self._saved_real_stat = self.statistic(real)

    def _compute_stats(self, fake, real=None):
        self._saved_fake_stat = self.statistic(fake)

    def _compute(self, fake, real):
        # The real is not vatiable is not used as the stat over real is
        # computed only once
        self._compute_stats(fake, real)
        fake_stat = self._saved_fake_stat
        real_stat = self._saved_real_stat
        if isinstance(real_stat, tuple):
            rs = real_stat[0]
            fs = fake_stat[0]
        else:
            rs = real_stat
            fs = fake_stat
        if self._log:
            rs = 10*np.log10(rs + 1e-2)
            fs = 10*np.log10(fs + 1e-2)
        diff = np.mean((rs - fs)**self._order)
        return diff

    def compute_summary(self, fake, real, feed_dict={}):
        super().compute_summary(fake, real, feed_dict)
        if self._stype == 3:
            feedict = self._plot_summary.compute_summary(
                self._saved_real_stat[1],
                self._saved_real_stat[0],
                self._saved_fake_stat[0],
                feed_dict=feed_dict)
        return feed_dict

    @property
    def real_stat(self):
        if self._saved_real_stat:
            return self._saved_real_stat
        else:
            raise ValueError("The statistic has not been computed yet")

    @property
    def fake_stat(self):
        if self._saved_fake_stat:
            return self._saved_fake_stat
        else:
            raise ValueError("The statistic has not been computed yet")


class StatisticalMetricLim(StatisticalMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lim = None

    def preprocess(self, real):
        """Compute the statistic on the real data."""
        super().preprocess(real)
        self._lim = self._saved_real_stat[2]

    def _compute_stats(self, fake, real=None):
        self._saved_fake_stat = self.statistic(fake, lim=self._lim)



