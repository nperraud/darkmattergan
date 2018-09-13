"""List of metrics for the GAN models."""

from .core import StatisticalMetric, Statistic
import numpy as np
from scipy import stats

def gan_stat_list():
	"""Return a list of statistic for a GAN."""

	stat_list = []

	# While the code of the first statistic might not be optimal, 
	# it is likely to be negligible compared to all the rest.

	# Mean
	def mean(x):
		return np.mean(np.flatten(x))
	stat_list.append(Statistic(mean,'mean', 'descriptives'))

	# var
	def var(x):
		return np.var(np.flatten(x))
	stat_list.append(Statistic(var,'var', 'descriptives'))

	# min
	def min(x):
		return np.min(np.flatten(x))
	stat_list.append(Statistic(min, 'min', 'descriptives'))

	# max
	def max(x):
		return np.min(np.flatten(x))
	stat_list.append(Statistic(max, 'max', 'descriptives'))

	# kurtosis
	def kurtosis(x):
		return stats.kurtosis(np.flatten(x))
	stat_list.append(Statistic(kurtosis, 'kurtosis', 'descriptives'))	

	# skewness
	def skewness(x):
		return stats.skew(np.flatten(x))
	stat_list.append(Statistic(skewness, 'kurtosis','descriptives'))

	# median
	def median(x):
		return stats.median(np.flatten(x))
	stat_list.append(Statistic(median, 'median','descriptives'))




def gan_metric_list():
	"""Return a metric list for a GAN."""

	stat_list = gan_stat_list()

	metric_list = []
	
