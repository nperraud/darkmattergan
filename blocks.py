import tensorflow as tf
import numpy as np
from numpy import prod


def _tf_variable(name, shape, initializer):
    """Create a tensorflow variable.

    Arguments
    --------
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
    """Squeeze x into a 2d matrix."""
    return tf.reshape(
        x, [tf.shape(x)[0], prod(x.shape.as_list()[1:])], name=name)


def reshape4d(x, sx, sy, nc, name=None):
    """Squeeze x into a 2d matrix."""
    return tf.reshape(x, [tf.shape(x)[0], sx, sy, nc], name=name)


def lrelu(x, leak=0.2, name="lrelu"):
    """Leak relu."""
    return tf.maximum(x, leak * x, name=name)


def batch_norm(x, epsilon=1e-5, momentum=0.9, name="batch_norm", train=True):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(
            x,
            decay=momentum,
            updates_collections=None,
            epsilon=epsilon,
            scale=True,
            is_training=train,
            scope=name)

        return bn


def downsample(imgs, s, is_3d=False):
    # To be rewritten in numpy
    if is_3d:
        # 1 extra dim for channels
        imgs = np.expand_dims(imgs, axis=4)

        x = tf.placeholder(tf.float32, shape=imgs.shape, name='x')
        xd = down_sampler(x, s=s, is_3d=True)
        with tf.Session() as sess:
            img_d = sess.run(xd, feed_dict={x: imgs})

        return np.squeeze(img_d)

    else:
        if len(imgs.shape) < 4:
            imgs = np.expand_dims(imgs, axis=3)
       
        x = tf.placeholder(tf.float32, shape=[*imgs.shape[:3], 1], name='x')
        xd = down_sampler(x, s=s, is_3d=False)
        with tf.Session() as sess:
            img_d = []
            for i in range(imgs.shape[3]):
                curr_img = np.expand_dims(imgs[:, :, :, i], axis=3)
                img_d.append(sess.run(xd, feed_dict={x: curr_img}))
        return np.squeeze(np.concatenate(img_d, axis=3))



def down_sampler(x, s=2, is_3d=False):
    if is_3d:
        filt = tf.constant(1 / (s * s * s), dtype=tf.float32, shape=[s, s, s, 1, 1])
        return tf.nn.conv3d(x, filt, strides=[1, s, s, s, 1], padding='SAME')
    else:
        filt = tf.constant(1 / (s * s), dtype=tf.float32, shape=[s, s, 1, 1])
        return tf.nn.conv2d(x, filt, strides=[1, s, s, 1], padding='SAME')


def up_sampler(x, s=2, is_3d=False):
    bs = tf.shape(x)[0]
    dims = x.shape.as_list()[1:]

    if is_3d:
        filt = tf.constant(1, dtype=tf.float32, shape=[s, s, s, 1, 1])
        output_shape = [bs, dims[0] * s, dims[1] * s, dims[2] * s, dims[3]]
        return tf.nn.conv3d_transpose(
                        x,
                        filt,
                        output_shape=output_shape,
                        strides=[1, s, s, s, 1],
                        padding='SAME')
    else:
        filt = tf.constant(1, dtype=tf.float32, shape=[s, s, 1, 1])
        output_shape = [bs, dims[0] * s, dims[1] * s, dims[2]]
        return tf.nn.conv2d_transpose(
                        x,
                        filt,
                        output_shape=output_shape,
                        strides=[1, s, s, 1],
                        padding='SAME')


# # Testing up_sampler, down_sampler
# x = tf.placeholder(tf.float32, shape=[1,256,256,1],name='x')
# input_img = np.reshape(gen_sample[1], [1,256,256,1])
# xd = blocks.down_sampler(x, s=2)
# xdu = blocks.up_sampler(xd, s=2)
# xdud = blocks.down_sampler(xdu, s=2)
# xdudu = blocks.up_sampler(xdud, s=2)
# with tf.Session() as sess:
#     img_d, img_du = sess.run([xd,xdu], feed_dict={x: input_img})
#     img_dud, img_dudu = sess.run([xdud,xdudu], feed_dict={x: input_img})
# img_d = np.squeeze(img_d)
# img_du = np.squeeze(img_du)
# img_dud = np.squeeze(img_dud)
# img_dudu = np.squeeze(img_dudu)
# img = np.squeeze(input_img)

# img_d2 = np.zeros([128,128])

# for i in range(128):
#     for j in range(128):
#         img_d2[i,j] = np.mean(img[2*i:2*(i+1),2*j:2*(j+1)])
# print(np.linalg.norm(img_d2-img_d,ord='fro'))
# print(np.linalg.norm(img_dud-img_d,ord='fro'))
# print(np.linalg.norm(img_dudu-img_du,ord='fro'))


def conv2d(imgs, nf_out, shape=[5, 5], stride=2, name="conv2d", summary=True):
    '''Convolutional layer for square images'''

    weights_initializer = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        w = _tf_variable(
            'w', [shape[0], shape[1],
                  imgs.get_shape()[-1], nf_out],
            initializer=weights_initializer)
        conv = tf.nn.conv2d(
            imgs, w, strides=[1, stride, stride, 1], padding='SAME')

        biases = _tf_variable('biases', [nf_out], initializer=const)
        conv = tf.nn.bias_add(conv, biases)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])

        return conv


def conv3d(imgs,
           nf_out,
           shape=[5, 5, 5],
           stride=2,
           name="conv3d",
           summary=True):
    '''Convolutional layer for square images'''

    weights_initializer = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        w = _tf_variable(
            'w', [shape[0], shape[1], shape[2],
                  imgs.get_shape()[-1], nf_out],
            initializer=weights_initializer)
        conv = tf.nn.conv3d(
            imgs, w, strides=[1, stride, stride, stride, 1], padding='SAME')

        biases = _tf_variable('biases', [nf_out], initializer=const)
        conv = tf.nn.bias_add(conv, biases)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])

        return conv


def deconv2d(imgs,
             output_shape,
             shape=[5, 5],
             stride=2,
             name="deconv2d",
             summary=True):

    weights_initializer = tf.contrib.layers.xavier_initializer()
    # was
    # weights_initializer = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = _tf_variable(
            'w', [shape[0], shape[1], output_shape[-1],
                  imgs.get_shape()[-1]],
            initializer=weights_initializer)

        deconv = tf.nn.conv2d_transpose(
            imgs,
            w,
            output_shape=output_shape,
            strides=[1, stride, stride, 1])

        biases = _tf_variable(
            'biases', [output_shape[-1]], initializer=const)
        deconv = tf.nn.bias_add(deconv, biases)

        # If we are running on Leonhard we need to reshape in order for TF
        # to explicitly know the shape of the tensor. Machines with newer
        # TensorFlow versions do not need this.
        if tf.__version__ == '1.3.0':
            deconv = tf.reshape(deconv, output_shape)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])
        return deconv


def deconv3d(imgs,
             output_shape,
             shape=[5, 5, 5],
             stride=2,
             name="deconv3d",
             summary=True):

    weights_initializer = tf.contrib.layers.xavier_initializer()
    # was
    # weights_initializer = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        # filter : [depth, height, width, output_channels, in_channels]
        w = _tf_variable(
            'w', [
                shape[0], shape[1], shape[2], output_shape[-1],
                imgs.get_shape()[-1]
            ],
            initializer=weights_initializer)

        deconv = tf.nn.conv3d_transpose(
            imgs,
            w,
            output_shape=output_shape,
            strides=[1, stride, stride, stride, 1])

        biases = _tf_variable(
            'biases', [output_shape[-1]],
            initializer=const)  # one bias for each filter
        deconv = tf.nn.bias_add(deconv, biases)

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
        matrix = _tf_variable(
            "Matrix", [shape[1], output_size],
            initializer=weights_initializer)
        bias = _tf_variable("bias", [output_size], initializer=const)
        if summary:
            tf.summary.histogram(
                "Matrix_sum", matrix, collections=["metrics"])
            tf.summary.histogram("Bias_sum", bias, collections=["metrics"])
        return tf.matmul(input_, matrix) + bias


def mini_batch_reg(xin, n_kernels=300, dim_per_kernel=50):
    x = linear(xin, n_kernels * dim_per_kernel, scope="minibatch_reg")
    activation = tf.reshape(x, [tf.shape(x)[0], n_kernels, dim_per_kernel])
    abs_dif = tf.reduce_sum(
        tf.abs(
            tf.expand_dims(activation, 3) -
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
    C = tf.exp(-abs_dif)
    minibatch_features = (tf.reduce_sum(C, 2) - 1) / (
        tf.subtract(tf.cast(tf.shape(x)[0], tf.float32), 1.0))
    x = tf.concat([xin, minibatch_features], axis=1)

    return x


def tf_cdf(x, n_out, diff_weight=10):
    """Helping function to get correct histograms."""
    weights_initializer = tf.contrib.layers.xavier_initializer()
    w = _tf_variable(
        'cdf_weight',
        shape=[1, 1, n_out],
        initializer=weights_initializer)
    x = tf.expand_dims(reshape2d(x), axis=2)
    return tf.reduce_mean(tf.sigmoid(diff_weight * (w - x)), axis=1)


def tf_covmat(x, shape):
    if x.shape[-1] > 1:
        lst = []
        for i in range(x.shape[-1]):
            lst.append(tf_covmat(x[:,:,:,i:i+1], shape))
        return tf.stack(lst, axis=1)
    nel = np.prod(shape)
    bs = tf.shape(x)[0]

    sh = [shape[0], shape[1], 1, nel]
    # wi = tf.constant_initializer(0.0)
    # w = _tf_variable('covmat_var', sh, wi)
    w = tf.constant(np.eye(nel).reshape(sh), dtype=tf.float32)

    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
    nx = conv.shape[1]*conv.shape[2]
    conv_vec = tf.reshape(conv,shape=[bs, nx ,nel])
    m = tf.reduce_mean(conv_vec, axis=[1,2])
    conv_vec = tf.subtract(conv_vec,tf.expand_dims(tf.expand_dims(m,axis=1), axis=2))
    c = 1/tf.cast(nx, tf.float32)*tf.matmul(tf.transpose(conv_vec,perm=[0,2,1]), conv_vec)
    return c

