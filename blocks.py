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


def selu(x, name="selu"):
    return tf.nn.selu(x, name=name)


def prelu(x, name="prelu"):
    """Parametrized Rectified Linear Unit, He et al. 2015, https://arxiv.org/abs/1502.01852"""
    return tf.keras.layers.PReLU(name=name)(x)


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

def np_downsample_2d(x, scaling):
    unique = False
    if len(x.shape)<2:
        raise ValueError('Too few dimensions')
    elif len(x.shape)==2:
        x = x.reshape([1,*x.shape])
        unique = True
    elif len(x.shape)>3:
        raise ValueError('Too many dimensions')
        
    n, sx, sy = x.shape 
    dsx = np.zeros([n,sx//scaling,sy//scaling])
    for i in range(scaling):
        for j in range(scaling):
            dsx += x[:,i::scaling,j::scaling]
    dsx /= (scaling**2)
    if unique:
        dsx = dsx[0]
    return dsx

def np_downsample_3d(x, scaling):
    unique = False
    if len(x.shape)<3:
        raise ValueError('Too few dimensions')
    elif len(x.shape)==3:
        x = x.reshape([1,*x.shape])
        unique = True
    elif len(x.shape)>4:
        raise ValueError('Too many dimensions')
        
    n, sx, sy, sz = x.shape 
    nx = sx//scaling
    ny = sy//scaling
    nz = sz//scaling
    dsx = np.zeros([n,nx,ny,nz])
    for i in range(scaling):
        for j in range(scaling):
            for k in range(scaling):
                dsx += x[:,i:scaling*nx+i:scaling,j:scaling*ny+j:scaling,k:scaling*nz+k:scaling]
    dsx /= (scaling**3)
    if unique:
        dsx = dsx[0]
    return dsx

def downsample(imgs, s, is_3d=False):
    if is_3d:
        return np_downsample_3d(imgs,s)
    else:
        return np_downsample_2d(imgs,s)




def down_sampler(x=None, s=2, is_3d=False):
    '''
    Op to downsample 2D or 3D images by factor 's'.
    This method works for both inputs: tensor or placeholder
    '''

    # The input to the downsampling operation is a placeholder.
    if x is None:
        placeholder_name = 'down_sampler_in_' + ('3d_' if is_3d else '2d_') + str(s)
        down_sampler_x = tf.placeholder(dtype=tf.float32, name=placeholder_name)
        op_name = 'down_sampler_out_' + ('3d_' if is_3d else '2d_') + str(s)
    
    # The input to the downsampling operation is the input tensor x.
    else: 
        down_sampler_x = x
        op_name = None


    if is_3d:
        filt = tf.constant(1 / (s * s * s), dtype=tf.float32, shape=[s, s, s, 1, 1])
        return tf.nn.conv3d(down_sampler_x, filt, strides=[1, s, s, s, 1], padding='SAME', name=op_name)

    else:
        filt = tf.constant(1 / (s * s), dtype=tf.float32, shape=[s, s, 1, 1])
        return tf.nn.conv2d(down_sampler_x, filt, strides=[1, s, s, 1], padding='SAME', name=op_name)


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




def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv2d(imgs, nf_out, shape=[5, 5], stride=2, use_spectral_norm=False, name="conv2d", summary=True):
    '''Convolutional layer for square images'''

    weights_initializer = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        w = _tf_variable(
            'w', [shape[0], shape[1],
                  imgs.get_shape()[-1], nf_out],
            initializer=weights_initializer)
        if use_spectral_norm:
            w = spectral_norm(w)
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
           use_spectral_norm=False,
           name="conv3d",
           summary=True):
    '''Convolutional layer for square images'''

    if use_spectral_norm:
        print("Warning spectral norm for conv3d set to True but may not be implemented!")

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


def inception_deconv(in_tensor, bs, sx, n_filters, stride, summary, num, is_3d=False, merge=False):
    if is_3d:
        output_shape = [bs, sx, sx, sx, n_filters]
        deconv = deconv3d
        shape = [[1, 1, 1], [3, 3, 3], [5, 5, 5]]
        concat_axis = 4
    else:
        output_shape = [bs, sx, sx, n_filters]
        deconv = deconv2d
        shape = [[1, 1], [3, 3], [5, 5]]
        concat_axis = 3

    tensor_1 = deconv(in_tensor,
                          output_shape=output_shape,
                          shape=shape[0],
                          stride=stride,
                          name='{}_deconv_1_by_1'.format(num),
                          summary=summary)

    tensor_3 = deconv(in_tensor,
                          output_shape=output_shape,
                          shape=shape[1],
                          stride=stride,
                          name='{}_deconv_3_by_3'.format(num),
                          summary=summary)

    tensor_5 = deconv(in_tensor,
                          output_shape=output_shape,
                          shape=shape[2],
                          stride=stride,
                          name='{}_deconv_5_by_5'.format(num),
                          summary=summary)

    out_tensor = tf.concat([tensor_1, tensor_3, tensor_5], axis=concat_axis)

    if merge:
        # do 1x1 convolution to reduce the number of output channels from (3 x n_filters) to n_filters
        out_tensor = deconv(out_tensor,
                          output_shape=output_shape,
                          shape=shape[0],
                          stride=1,
                          name='{}_deconv_1_by_1_merge'.format(num),
                          summary=summary)

    return out_tensor

def inception_conv(in_tensor, n_filters, stride, summary, num, is_3d=False, merge=False):
    if is_3d:
        conv = conv3d
        shape = [[1, 1, 1], [3, 3, 3], [5, 5, 5]]
        concat_axis = 4
    else:
        conv = conv2d
        shape = [[1, 1], [3, 3], [5, 5]]
        concat_axis = 3

    tensor_1 = conv(in_tensor,
                    nf_out=n_filters,
                    shape=shape[0],
                    stride=stride,
                    name='{}_conv_1_by_1'.format(num),
                    summary=summary)

    tensor_3 = conv(in_tensor,
                    nf_out=n_filters,
                    shape=shape[1],
                    stride=stride,
                    name='{}_conv_3_by_3'.format(num),
                    summary=summary)

    tensor_5 = conv(in_tensor,
                 nf_out=n_filters,
                 shape=shape[2],
                 stride=stride,
                 name='{}_conv_5_by_5'.format(num),
                 summary=summary)

    out_tensor = tf.concat([tensor_1, tensor_3, tensor_5], axis=concat_axis)

    if merge:
        # do 1x1 convolution to reduce the number of output channels from (3 x n_filters) to n_filters
        out_tensor = conv(out_tensor,
                        nf_out=n_filters,
                        shape=shape[0],
                        stride=1,
                        name='{}_conv_1_by_1_merge'.format(num),
                        summary=summary)

    return out_tensor


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


def tf_cdf(x, n_out, name='cdf_weight', diff_weight=10, use_first_channel=True):

    """Helping function to get correct histograms."""
    # limit = 4.
    # wi = tf.range(0.0, limit, delta=limit/n_out, dtype=tf.float32, name='range')

    # wl = tf.Variable(
    #     tf.reshape(wi, shape=[1, 1, n_out]),
    #     name='cdf_weight_left',
    #     dtype=tf.float32)
    # wr = tf.Variable(
    #     tf.reshape(wi, shape=[1, 1, n_out]),
    #     name='cdf_weight_right',
    #     dtype=tf.float32)
    weights_initializer = tf.contrib.layers.xavier_initializer()
    wr = _tf_variable(
        name+'_right',
        shape=[1, 1, n_out],
        initializer=weights_initializer)
    wl = _tf_variable(
        name+'_left',
        shape=[1, 1, n_out],
        initializer=weights_initializer)
    if use_first_channel:
        nc = len(x.shape)
        if nc == 4:
            x = x[:, :, :, 0]
        elif nc == 5:
            x = x[:, :, :, :, 0]
        else:
            raise ValueError('Wrong size')
    x = tf.expand_dims(reshape2d(x), axis=2)
    xl = tf.reduce_mean(tf.sigmoid(10 * (wl - x)), axis=1)
    xr = tf.reduce_mean(tf.sigmoid(10 * (x - wr)), axis=1)
    tf.summary.histogram("cdf_weight_right", wr, collections=["metrics"])
    tf.summary.histogram("cdf_weight_left", wl, collections=["metrics"])

    return tf.concat([xl, xr], axis=1)


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


def learned_histogram(x, params):
    """A learned histogram layer.

    The center and width of each bin is optimized.
    One histogram is learned per feature map.
    """
    # Shape of x: #samples x #nodes x #features.
    bins = params.get('bins', 20)
    initial_range = params.get('initial_range', 2)
    is_3d = params.get('is_3d', False)
    if is_3d:
        x = tf.reshape(x, [tf.shape(x)[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4]])
    else:
        x = tf.reshape(x, [tf.shape(x)[0], x.shape[1] * x.shape[2], x.shape[3]])
    n_features = int(x.get_shape()[2])
    centers = tf.linspace(float(0), initial_range, bins, name='range')
    centers = tf.expand_dims(centers, axis=1)
    centers = tf.tile(centers, [1, n_features])  # One histogram per feature channel.
    centers = tf.Variable(
        tf.reshape(tf.transpose(centers), shape=[1, 1, n_features, bins]),
        name='centers', dtype=tf.float32)
    # width = bins * initial_range / 4  # 50% overlap between bins.
    width = 1 / initial_range  # 50% overlap between bins.
    widths = tf.get_variable(
        name='widths', shape=[1, 1, n_features, bins], dtype=tf.float32,
        initializer=tf.initializers.constant(value=width, dtype=tf.float32))
    x = tf.expand_dims(x, axis=3)
    # All are rank-4 tensors: samples, nodes, features, bins.
    widths = tf.abs(widths)
    dist = tf.abs(x - centers)
    hist = tf.reduce_mean(tf.nn.relu(1 - dist * widths), axis=1)
    return tf.reshape(hist, [tf.shape(hist)[0], hist.shape[1] * hist.shape[2]])
