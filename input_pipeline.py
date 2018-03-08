import tensorflow as tf
import numpy as np
import os, random, pickle
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import utils

class input_pipeline(object):
	def __init__(self, dir_paths, batch_size, k=10, shuffle=False, buffer_size=1000):
		self.dir_paths = dir_paths
		self.k = k
		self._read_sample_file_paths()
		self.num_samples = len(self.sample_file_paths)

		if shuffle:
			random.shuffle(self.sample_file_paths)

		self.sample_file_paths = convert_to_tensor(self.sample_file_paths, name='sample_file_paths', dtype=dtypes.string)

		data = Dataset.from_tensor_slices(self.sample_file_paths)
		
		data = data.map(
				lambda sample_file_path : tf.py_func(
					self._py_read_file, [sample_file_path], [tf.float32]),
				num_threads=8, output_buffer_size=100*batch_size)

		if shuffle:
			data = data.shuffle(buffer_size=buffer_size)

		data = data.batch(batch_size)
		self.data = data

	def _read_sample_file_paths(self):
		'''
		store paths of all sample files in a list
		'''
		self.sample_file_paths = []
		for dir_path in self.dir_paths:
			for file_name in os.listdir(dir_path):
				file_path = os.path.join(dir_path, file_name)
				self.sample_file_paths.append(file_path)

	def _py_read_file(self, sample_file_path):
		'''
		parser function for samples of the dataset.
		for every file path in the dataset, this function gets called,
		where we load the data from the file. If only tensorflow ops here, then performance-wise more efficient
		'''
		sample = []
		with open(sample_file_path, 'rb') as pickle_file:
			sample = pickle.load(pickle_file)
			sample = utils.forward_map(sample, self.k)
			sample.resize([*sample.shape, 1])

		ret = sample.astype(np.float32)
		return ret

