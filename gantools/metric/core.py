"""This module define the base classes for the metrics of GANs."""

import numpy as np
import tensorflow as tf

class Statistic(object):
    """Base class for a statistic."""

    def __init__(self, func, name, group=''):
        """Create a statistical object.

        Arguments
        ---------
        * name: name of the statistic (preferably unique)
        * func: function to compute the statistic
        """
        self._name = name
        self._group = group
        self._func = func

    def __call__(self, dat):
        """Compute the statistic."""
        return self._func(dat)

    def add_summary(self, stype=0, collections=None):
        """Add a tensorflow summary.

        stype: summary type. 0 scalar,
        """
        name = self.group+'/'+self.name
        self._placeholder = tf.placeholder(tf.float32, name=name)
        if stype==0:
            tf.summary.scalar(name, self._placeholder, collections=collections)
        else:
            ValueError('Wrong summary type')

    def compute_summary(self, dat, feed_dict={}):
        feed_dict[self._placeholder] = self(dat)
        return feed_dict

    @property
    def group(self):
        return self._group
    
    @property
    def name(self):
        return self._name

class Metric(object):
    """Base metric class."""

    def __init__(self, name, group='', recompute_real=True):
        """Initialize the statistic.

        Argument
        --------
        * name: name of the statistic (preferably unique)
        * recompute_real: recompute the real statistic (default True)
        """
        self._name = name
        self._group = group
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

    def add_summary(self, collections=None):
        """Add a tensorflow summary.

        stype: summary type. 0 scalar,
        """
        name = self.group+'/'+self.name
        self._placeholder = tf.placeholder(tf.float32, name=name)
        tf.summary.scalar(name, self._placeholder, collections=collections)

    def compute_summary(self, fake, real, feed_dict={}):
        feed_dict[self._placeholder] = self(fake, real)
        return feed_dict

    @property
    def preprocessed(self):
        """Return True if the preprocessing been done."""
        return self._preprocessed

    @property
    def group(self):
        return self._group
    
    @property
    def name(self):
        return self._name
    


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
        self._statistic = statistic
        self._saved_stat = None

    def preprocess(self, real):
        """Compute the statistic on the real data."""
        super().preprocess(real)
        self._saved_real_stat = self.statistic(real)
        if self._log:
            self._saved_real_stat = np.log(self._saved_real_stat)

    def statistic(self, dat):
        """Compute the statistics on some data."""
        return self._statistic(dat)

    def _compute(self, fake, real):
        # The real is not vatiable is not used as the stat over real is
        # computed only once
        real_stat = self._saved_real_stat
        fake_stat = self.statistic(fake)
        if self._log:
            fake_stat = np.log(fake_stat)
            # The log for the real is done in preprocess
        self._saved_fake_stat = fake_stat
        return (np.sum((real_stat - fake_stat)**self._order))**(1/self._order)


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
    