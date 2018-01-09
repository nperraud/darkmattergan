import tensorflow as tf
from numpy import prod



def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    if True:  # with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def numel(x):
    ''' Return the number of element in x '''
    return prod(tf.shape(x))  # tf.size?


def reshape2d(x, name=None):
    ''' Squeeze x into a 2d matrix '''
    return tf.reshape(x, [tf.shape(x)[0], prod(x.shape.as_list()[1:])], name=name)


def reshape4d(x, sx, sy, nc, name=None):
    ''' Squeeze x into a 2d matrix '''
    return tf.reshape(x, [tf.shape(x)[0], sx, sy, nc], name=name)


def lrelu(x, leak=0.2, name="lrelu"):
    ''' Leak relu '''
    return tf.maximum(x, leak*x, name=name)


def batch_norm(x, epsilon=1e-5, momentum=0.9, name="batch_norm", train=True):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(x,
                                          decay=momentum,
                                          updates_collections=None,
                                          epsilon=epsilon,
                                          scale=True,
                                          is_training=train,
                                          scope=name)

        return bn


def down_sampler(x, s=2):
    filt = tf.constant(1/(s*s), dtype=tf.float32, shape=[s, s, 1, 1])
    return tf.nn.conv2d(x, filt, strides=[1, s, s, 1], padding='SAME')


def up_sampler(x, s=2):
    filt = tf.constant(1/(s*s), dtype=tf.float32, shape=[s, s, 1, 1])
    bs = tf.shape(x)[0]
    shx2 = x.shape.as_list()[1:]
    output_shape = [bs, shx2[0]*s, shx2[1]*s, shx2[2]]
    return tf.nn.conv2d_transpose(x, filt, output_shape=output_shape,
                                  strides=[1, s, s, 1], padding='SAME')


def conv2d(imgs, nf_out, shape=[5, 5], stride=2, name="conv2d", summary=True):
    '''Convolutional layer for square images'''

    weights_initializer = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        w = _variable_on_cpu('w',
                             [shape[0], shape[1], imgs.get_shape()[-1], nf_out],
                             initializer=weights_initializer)
        conv = tf.nn.conv2d(imgs,
                            w,
                            strides=[1, stride, stride, 1],
                            padding='SAME')


        biases = _variable_on_cpu('biases', [nf_out], initializer=const)
        conv =tf.nn.bias_add(conv, biases)#  tf.reshape(, conv.get_shape())

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])

        return conv


def deconv2d(imgs, output_shape, shape=[5, 5], stride=2, name="deconv2d", summary=True):

    weights_initializer = tf.contrib.layers.xavier_initializer()
    # was
    # weights_initializer = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = _variable_on_cpu('w',
                             [shape[0],
                              shape[1], output_shape[-1], imgs.get_shape()[-1]],
                             initializer=weights_initializer)

        deconv = tf.nn.conv2d_transpose(imgs,
                                        w,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1])


        biases = _variable_on_cpu('biases', [output_shape[-1]], initializer=const)
        deconv = tf.nn.bias_add(deconv, biases) #tf.reshape(, deconv.get_shape())

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])
        return deconv


def linear(input_, output_size, scope=None, summary=True):
    shape = input_.get_shape().as_list()

    weights_initializer = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(scope or "Linear"):
        matrix = _variable_on_cpu("Matrix",
                                  [shape[1], output_size],
                                  initializer=weights_initializer)
        bias = _variable_on_cpu("bias", [output_size], initializer=const)
        if summary:
            tf.summary.histogram("Matrix_sum", matrix, collections=["metrics"])
            tf.summary.histogram("Bias_sum", bias, collections=["metrics"])
        return tf.matmul(input_, matrix) + bias




