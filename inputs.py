import os
import tensorflow as tf
import numpy as np
from scipy import ndimage
ds = tf.contrib.distributions

def get_filename_queue(save_format,  input_pattern, path):
    if save_format == 'dat':
        file_ext = '.dat'
    else:
        file_ext = '.png'
    if input_pattern== '':
        print(" [!] Input pattern not specified, all images in the directory will be used")
    print('\n [*] Reading {}*{}*.{}'.format(path, input_pattern,save_format))
    queue = []
    print('a')
    print(input_pattern)
    print('b')
    for file in os.listdir(path):
        if file.endswith(file_ext) and (np.all([x in file for x in input_pattern.split("*")])):
            queue.append(os.path.join(path, file))
    N = len(queue)
    print(" [*] N = {}".format(N))
    return tf.train.string_input_producer(queue), N  # list of files to read


def gen_noise(z_dim,batch_size):
    return ds.Normal(tf.zeros(z_dim),
                      tf.ones(z_dim)).sample(batch_size)


def sample_latent(m, n, prior = "uniform"):
    if prior == "uniform":
        return np.random.uniform(-1., 1., size=[m, n])
    elif prior == "gaussian":
        return np.random.normal(0,1, size=[m,n])
    elif prior.startswith('student'):
        prior_ = prior.split('-')
        if len(prior_) == 2:
            df = int(prior_[1])
        else:
            df = 3
        return np.random.standard_t(df, size=[m,n])
    elif prior.startswith('chi2'):
        prior_ = prior.split('-')
        if len(prior_) >= 2:
            df = int(prior_[1])
            if len(prior_) >= 3:
                k = float(prior_[2])
            else:
                k = 7
        else:
            df, k = 3, 7
        return simple_numpy(np.random.chisquare(df, size=[m,n]), k)
    elif prior.startswith('gamma'):
        prior_ = prior.split('-')
        if len(prior_) >= 2:
            df = float(prior_[1])
            if len(prior_) >= 3:
                k = float(prior_[2])
            else:
                k = 4
        else:
            df, k = 1, 4
        return simple_numpy(np.random.gamma(df, size=[m,n]), k)
    else:
        raise ValueError(' [!] distribution not defined')

def gen_unif_noise(z_dim,batch_size):
    return tf.random_uniform(shape=[batch_size,z_dim], minval=-1, maxval=1)


def get_sim_data(filename_queue, params, flat_img = True):
    if params['save_format'] == 'dat':
        if params['format_float'] == "float32":
            m = 4
        elif params['format_float'] == "float16":
            m = 2
        elif params['format_float'] == "float64":
            m = 8
        else:
            raise ValueError('format_float not defined')

        reader = tf.FixedLengthRecordReader(record_bytes=params['x_dim'] * m)
        key, value = reader.read(filename_queue)

        my_img = tf.decode_raw(value, getattr(tf, params['format_float']))
    else:
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)

        my_img0 = tf.image.decode_png(value)
        my_img = tf.slice(my_img0, [0, 0, 0],
                          [params['size_image'], params['size_image'], params['dim_color']])
    if flat_img:
        img_flat = tf.reshape([my_img], [params['x_dim']])
    else:
        img_flat = tf.reshape([my_img], [params['size_image'], params['size_image'], 1])
    data = tf.cast(img_flat, dtype=getattr(tf, params['format_float']))

    if params['sigma_smooth_input'] is not None:
        # smooth images and round the output to the nearest integer

        def smooth(x):
            return ndimage.gaussian_filter(x, sigma=params['sigma_smooth_input'])

        data_ = tf.py_func(smooth, [data], getattr(tf, params['format_float']), name="smoothed_input")
        data_.set_shape(data.get_shape())
        data = data_
        data = tf.round(data)

    min_queue_examples = params['batch_size'] * 100
    num_preprocess_threads = 16
    batch_data = tf.train.shuffle_batch([data], batch_size=params['batch_size'], num_threads=num_preprocess_threads,
                                  capacity=min_queue_examples + 3 * params['batch_size'],
                                  min_after_dequeue=min_queue_examples)
    return batch_data


def get_inputs(data_type, params, path = None, mnist = None, flat_img = True):

    try:
        params = params.__flags
    except:
        pass

    if not params['is_train'] and params['viz_option'] != 'metrics' and params['viz_option'] != 'time': # return as a placeholder if we are only going to generate samples.
        return tf.placeholder(tf.float32,
                                    [params['batch_size'], params['size_image'], params['size_image'], 1]), None
    else:
        if data_type == "SIM":
            filename_queue, N = get_filename_queue(save_format=params['save_format'], input_pattern = params['input_pattern'], path=path)
            batch_data = get_sim_data(filename_queue, params, flat_img = flat_img)
        else:
            raise ValueError('No data set found...')

        return batch_data, N



def pre_process(X_raw, FLAGS):

    if FLAGS.pre_proc:
        if FLAGS.pre_proc == 'simple':

            k = tf.constant(FLAGS.simple_constant, dtype=tf.float32)

            # maps real positive numbers to a [-1,1] range  2 * (x/(x-10)) - 1
            X = tf.subtract(2.0 * (X_raw /  tf.add(X_raw, k)), 1.0, name="Input" + FLAGS.pre_proc)

        else:
            non_lin = getattr(tf, FLAGS.pre_proc)
            X = non_lin(X_raw, name="Input" + FLAGS.pre_proc)
    else:
        X = X_raw
    return X

def inv_pre_process(G, FLAGS):

    if FLAGS.pre_proc:
        inv_non_lin = get_inverse(FLAGS.pre_proc)

        #with tf.control_dependencies([tf.assert_greater_equal(G, -1.0), tf.assert_less_equal(G, 1.0)]):
        with tf.device('/cpu:0'):
            simple_max = simple_numpy(1e10, FLAGS.simple_constant)  # clipping the values to a max of 1e10 particles
            G_clipped = tf.clip_by_value(G, -1.0, simple_max, "G_clipped")
            G_raw = inv_non_lin(G_clipped,FLAGS.simple_constant)
    else:
        G_raw = G
    return G_raw


def get_inverse(non_lin):

    if non_lin == "log":
        inv = tf.exp
    elif non_lin == "sigmoid":
        # logit function
        def inv(x, k = None, name = "logit"):
            return tf.log(x / (1-x),name = name)
    elif non_lin == "tanh":
        def inv(x, k = None, name ="itanh"):
            k = tf.constant(0.5)
            return tf.multiply(k, tf.log( (1+x)/(1-x)), name = name)
    elif non_lin == "simple":
        def inv(x, k = None, name ="simple"):

            return tf.multiply((x + 1.0) / (1.0 - x),k, name)

    else:
        raise ValueError('non_linearity inverse not defined')
    return inv


def simple_numpy(x, k):
    k = float(k)
    return 2.0*x/(x+k) - 1.0

def log_numpy(x,k):
    return np.log(np.clip(x, k, 1e10))

