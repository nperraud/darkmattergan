"""This module define the base classes for the metrics of GANs."""

import numpy as np


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

    @property
    def group(self):
        return self._group
    
    @property
    def name(self):
        return self._name

class Metric(object):
    """Base metric class."""

    def __init__(self, name, group, recompute_real=True):
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
            self.preproces(real)
        elif (not self.preprocessed) and (not real):
            raise ValueError("The real data need to be preprocessed first!")
        self.compute(fake, real)

    def compute(self):
        """Compute the metric."""
        raise NotImplementedError("This is an abstract class.")

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
        super().__init__(name)
        self._order = order
        self._log = log
        self._statistic = statistic
        self._saved_stat = None

    def preprocess(self, real):
        """Compute the statistic on the real data."""
        super().preprocess(real)
        self._saved_real_stat = self.statistic(real)
        if self._log:
            real_stat = np.log(real_stat)

    def statistic(self, dat):
        """Compute the statistics on some data."""
        return self._statistic(dat)

    def compute(self, fake, real):
        real_stat = self._saved_real_stat
        fake_stat = self.statistic(real)
        if self._log:
            fake_stat = np.log(fake_stat)
            # The log for the real is done in preprocess
        self._saved_fake_stat = fake_stat
        return np.linalg.norm(real_stat - fake_stat, ord=self._order)


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
    