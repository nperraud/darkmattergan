"""Different maps."""

# TODO: renaming!
import tensorflow as tf
import numpy as np


def forward(X, shift=1.0):
    return np.log(np.sqrt(X) + np.e**shift) - shift


def backward(Xmap, max_value=2e5, shift=1.0):
    Xmap = np.clip(Xmap, 0, forward(max_value))
    tmp = np.exp(Xmap + shift) - np.e**shift
    return np.round(tmp * tmp)


def forward_old(X):
    return np.log(X**(1/2)+np.e)-2

def backward_old(Xmap, max_value=2e5):
    Xmap = np.clip(Xmap, -1.0, forward(max_value))
    tmp = np.exp((Xmap+2))-np.e
    return np.round(tmp*tmp)

def forward_map(x, k=10., scale=1.):
    """Map real positive numbers to a [-scale, scale] range.

    Numpy version
    """
    return scale * (2 * (x / (x + k)) - 1)


def backward_map(y, k=10., scale=1., real_max=1e8):
    """Inverse of the function forward map.

    Numpy version
    """
    simple_max = forward_map(real_max, k, scale)
    simple_min = forward_map(0, k, scale)
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
    simple_max = forward_map(real_max, k, scale)
    simple_min = forward_map(0, k, scale)
    X_clipped = tf.clip_by_value(X, simple_min, simple_max) / scale
    X_raw = tf.multiply((X_clipped + 1.0) / (1.0 - X_clipped), k)
    return X_raw
