import tensorflow as tf
import numpy as np
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


def downsample(imgs, s):
    imgs = np.expand_dims(imgs, axis=3)
    x = tf.placeholder(tf.float32, shape=imgs.shape, name='x')
    xd = down_sampler(x, s=s)
    with tf.Session() as sess:
        img_d = sess.run(xd, feed_dict={x: imgs})
    return np.squeeze(img_d)

def down_sampler(x, s=2):
    filt = tf.constant(1/(s*s), dtype=tf.float32, shape=[s, s, 1, 1])
    return tf.nn.conv2d(x, filt, strides=[1, s, s, 1], padding='SAME')


def up_sampler(x, s=2):
    filt = tf.constant(1, dtype=tf.float32, shape=[s, s, 1, 1])
    bs = tf.shape(x)[0]
    shx2 = x.shape.as_list()[1:]
    output_shape = [bs, shx2[0]*s, shx2[1]*s, shx2[2]]
    return tf.nn.conv2d_transpose(x, filt, output_shape=output_shape,
                                  strides=[1, s, s, 1], padding='SAME')
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


# class minibatch_discrimination(pt.VarStoreMethod):
#     def __call__(self, input_layer, num_kernels, dim_per_kernel=5, name='minibatch_discrim'):
#         batch_size = input_layer.shape[0]
#         num_features = input_layer.shape[1]
#         W = self.variable('W', [num_features, num_kernels*dim_per_kernel],
#                           init=tf.contrib.layers.xavier_initializer())
#         b = self.variable('b', [num_kernels], init=tf.constant_initializer(0.0))
#         activation = tf.matmul(input_layer, W)
#         activation = tf.reshape(activation, [batch_size, num_kernels, dim_per_kernel])
#         tmp1 = tf.expand_dims(activation, 3)
#         tmp2 = tf.transpose(activation, perm=[1,2,0])
#         tmp2 = tf.expand_dims(tmp2, 0)
#         abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
#         f = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])
#         f = f + b
#         return f


# def mini_batch_reg(xin, batch_size, n_kernels=300, dim_per_kernel=50):
#     x = linear(xin, n_kernels * dim_per_kernel, scope="d_h")

#     activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

#     big = np.zeros((batch_size, batch_size), dtype='float32')
#     big += np.eye(batch_size)
#     big = tf.expand_dims(big, 1)

#     abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
#     mask = 1. - big
#     masked = tf.exp(-abs_dif) * mask

#     def half(tens, second):
#         m, n, _ = tens.get_shape()
#         m = int(m)
#         n = int(n)
#         return tf.slice(tens, [0, 0, second * batch_size], [m, n, batch_size])
#     # TODO: speedup by allocating the denominator directly instead of constructing it by sum
#     #       (current version makes it easier to play with the mask and not need to rederive
#     #        the denominator)
#     f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
#     f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))

#     minibatch_features = [f1, f2]
#     x = tf.concat(1, [xin] + minibatch_features)
#     return x
