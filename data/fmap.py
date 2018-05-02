"""Different maps."""

# TODO: renaming!
import tensorflow as tf
import numpy as np
import scipy


def gauss_forward(x, shift=0):
    y = x + 1 + shift
    cp = (y-1)/y
    v = scipy.special.erfinv(cp)/np.sqrt(2)
    if not (shift==0):
        c = gauss_forward(shift,shift=0)
    else:
        c = 0.0
    return v - c

def gauss_backward(x, shift=0, clip_max=1e7):
    x_max = gauss_forward(clip_max, shift=shift)
    x = np.clip(x, 0.0, x_max)
    if not (shift==0):
        c = gauss_forward(shift,shift=0)
    else:
        c = 0.0
    cg = scipy.special.erf(np.sqrt(2)*(x + c))
    y = 1/(1-cg)
    return np.round(y - 1 - shift)

def log_forward(x):
    return np.log(x+np.e)

def log_backward(x):
    return np.round(np.exp(x)-np.e)

def shifted_log_forward(X, shift=1.0, minval=-1.0):
    return np.log(np.sqrt(X) + np.e**shift) - shift + minval


def shifted_log_backward(Xmap, max_value=2e5, shift=1.0,minval=-1.0):
    Xmap = np.clip(Xmap, minval, shifted_log_forward(max_value))
    tmp = np.exp(Xmap + shift - minval) - np.e**shift
    return np.round(tmp * tmp)


def nati_forward(X):
    return np.log(X**(1/2)+np.e)-2

def nati_backward(Xmap, max_value=2e5):
    Xmap = np.clip(Xmap, -1.0, nati_forward(max_value))
    tmp = np.exp((Xmap+2))-np.e
    return np.round(tmp*tmp)

def andres_forward(x, k=10., scale=1.):
    """Map real positive numbers to a [-scale, scale] range.

    Numpy version
    """
    return scale * (2 * (x / (x + k)) - 1)


def andres_backward(y, k=10., scale=1., real_max=1e8):
    """Inverse of the function forward map.

    Numpy version
    """
    simple_max = andres_forward(real_max, k, scale)
    simple_min = andres_forward(0, k, scale)
    y_clipped = np.clip(y, simple_min, simple_max) / scale
    return k * (y_clipped + 1) / (1 - y_clipped)


def pre_process(X_raw, k=10., scale=1.):
    """Map real positive numbers to a [-scale, scale] range.

    Tensorflow version
    """
    k = tf.constant(k, dtype=tf.float32)
    X = tf.subtract(2.0 * (X_raw / tf.add(X_raw, k)), 1.0) * scale
    return X


def inv_pre_process(X, k=10., scale=1., real_max=1e8):
    """Inverse of the function forward map.

    Tensorflow version
    """
    simple_max = andres_forward(real_max, k, scale)
    simple_min = andres_forward(0, k, scale)
    X_clipped = tf.clip_by_value(X, simple_min, simple_max) / scale
    X_raw = tf.multiply((X_clipped + 1.0) / (1.0 - X_clipped), k)
    return X_raw


forward = nati_forward
backward = nati_backward

