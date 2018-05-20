"""Different maps."""

# TODO: renaming!
import tensorflow as tf
import numpy as np
import scipy
import functools

def gauss_forward(x, shift=0, a = 1):
    y = x + 1 + shift
    cp = (y-1)/y
    v = scipy.special.erfinv(cp)/np.sqrt(2)
    if not (shift==0):
        c = gauss_forward(shift,shift=0)
    else:
        c = 0.0
    return v - c

def gauss_backward(x, shift=0, clip_max=1e6):
    x_max = gauss_forward(clip_max, shift=shift)
    x = np.clip(x, 0.0, x_max)
    if not (shift==0):
        c = gauss_forward(shift,shift=0)
    else:
        c = 0.0
    cg = scipy.special.erf(np.sqrt(2)*(x + c))
    y = 1/(1-cg)
    return np.round(y - 1 - shift)

def log_forward(x, shift=6):
    return np.log(x+1 + shift) - np.log(1 + shift)

def log_backward(x, shift=6, clip_max=1e6):
    x_max = log_forward(clip_max, shift=shift)
    x = np.clip(x, 0.0, x_max)
    return np.round(np.exp(x+ np.log(1 + shift))-1-shift)

def shifted_log_forward(X, shift=1.0):
    return np.log(np.sqrt(X) + np.e**shift) - shift

def shifted_log_backard(Xmap, clip_max=1e6, shift=1.0):
    Xmap = np.clip(Xmap, 0, shifted_log_forward(clip_max))
    tmp = np.exp(Xmap + shift) - np.e**shift
    return np.round(tmp * tmp)


def nati_forward(X):
    return np.log(X**(1/2)+np.e)-2

def nati_backward(Xmap, clip_max=1e6):
    Xmap = np.clip(Xmap, -1.0, nati_forward(clip_max))
    tmp = np.exp((Xmap+2))-np.e
    return np.round(tmp*tmp)


def uniform_forward(X, shift=20):
    """Transform a power law distribution with k=2 into a uniform distribution."""
    return X/(X+1+shift)

def uniform_backward(X, shift=20, clip_max=1e6):
    """Inverse transform a power law distribution with k=2 into a uniform distribution."""
    X = np.clip(X, 0.0, uniform_forward(clip_max, shift=shift))
    return np.round((shift+1)*X/(1.0-X))

def tanh_forward(X, shift=20):
    """Transform a power law distribution with k=2 into a 2-2*tanh^2(x)."""
    y = uniform_forward(X, shift)
    return np.arctanh(0.5*(y+1.0))

def tanh_backward(X, shift=20, clip_max=1e6):
    """Inverse transform a power law distribution with k=2 into a 1-tanh^2(x)."""
    y = 2*np.tanh(X) - 1.0
    return uniform_backward(y)

def andres_forward(x, shift=10., scale=1.):
    """Map real positive numbers to a [-scale, scale] range.

    Numpy version
    """
    return scale * (2 * (x / (x + 1+shift)) - 1)


def andres_backward(y, shift=20., scale=1., real_max=1e8):
    """Inverse of the function forward map.

    Numpy version
    """
    simple_max = andres_forward(real_max, shift, scale)
    simple_min = andres_forward(0, shift, scale)
    y_clipped = np.clip(y, simple_min, simple_max) / scale
    return (shift+1) * (y_clipped + 1) / (1 - y_clipped)


def pre_process(X_raw, k=20., scale=1.):
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


def stat_forward_0(x, c=2e4):
    if not type(x).__module__ == np.__name__:
        x = np.array([x])
    res = np.zeros(shape=x.shape)
    mask = x>c
    maski = mask==False
    res[maski] = np.log(x[maski]+1)
    res[mask] = np.log(c+1) + (x[mask]/c-1)
    res *= 3/ np.log(c+1)
    return res

def stat_backward_0(x, c=2e4):
    if not type(x).__module__ == np.__name__:
        x = np.array([x])
    res = np.zeros(shape=x.shape)
    mc = np.log(c+1)
    x *= mc/3
    mask = x>mc
    maski = mask==False
    res[maski] = np.exp(x[maski])-1
    res[mask] = c*(x[mask] - np.log(c+1) + 1)
    return np.round(res)

def stat_forward(x, c=2e4, shift=3):
    return stat_forward_0(x+shift, c=c) - stat_forward_0(shift, c=c)


def stat_backward(x, c=2e4, shift=3):
    return stat_backward_0(x+stat_forward_0(shift, c=c), c=c) - shift


forward = stat_forward
backward = stat_forward

