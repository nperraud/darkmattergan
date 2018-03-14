import tensorflow as tf
import numpy as np
import os, random
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import utils

class input_pipeline(object):
	def __init__(self, dir_paths, batch_size, k=10, shuffle=False, buffer_size=1000, num_samples_in_each_file=100, image_size=[16,16,16]):
		self.dir_paths = dir_paths
		self.k = k
		self.image_size = image_size
		self._read_sample_file_paths()
		self.num_samples = len(self.sample_file_paths) * num_samples_in_each_file

		if shuffle:
			random.shuffle(self.sample_file_paths)

		dataset = tf.data.TFRecordDataset(self.sample_file_paths)

		def parser(serialized_example):
		    """Parses a single tf.Example into image"""
		    parsed_features = tf.parse_single_example(
		        serialized_example,
		        features={
		            'image': tf.FixedLenFeature([], tf.string)
		        })
		    
		    image = parsed_features['image']
		    image = tf.decode_raw(image, tf.float32)
		    image = tf.cast(image, tf.float32)
		    image = tf.reshape(image, [*self.image_size, 1]) # add the dimension for #channels
		    return image

		dataset = dataset.map(parser)
		dataset = dataset.batch(batch_size)

		if shuffle:
			dataset = dataset.shuffle(buffer_size=buffer_size)

		self.dataset = dataset

	def _read_sample_file_paths(self):
		'''
		store paths of all sample files in a list
		'''
		self.sample_file_paths = []
		for dir_path in self.dir_paths:
			for file_name in os.listdir(dir_path):
				file_path = os.path.join(dir_path, file_name)
				self.sample_file_paths.append(file_path)

