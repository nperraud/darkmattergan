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

def gauss_backward(x, shift=0, clip_max=1e8):
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

def log_backward(x, shift=6, clip_max=1e8):
    x_max = log_forward(clip_max, shift=shift)
    x = np.clip(x, 0.0, x_max)
    return np.round(np.exp(x+ np.log(1 + shift))-1-shift)

def shifted_log_forward(X, shift=1.0):
    return np.log(np.sqrt(X) + np.e**shift) - shift


def shifted_log_backard(Xmap, max_value=2e5, shift=1.0):
    Xmap = np.clip(Xmap, 0, shifted_log_forward(max_value))
    tmp = np.exp(Xmap + shift) - np.e**shift
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



def power_law(x,k=2):
    """Power law for x>=1.
    
    p(x) = a x^(-k)
    """
    assert(k>1)
    a = k-1
    return a/(x**k)

def power_law_cdf(x, k=2):
    """CDF for power law for x>=1.
    
    c(x) = 1 - 1/(x^(k-1))
    """
    assert(k>1)
    a = k-1
    return 1 - 1/(x**a)

def power_law_cdf_inv(x, k=2):
    """Inverse CDF for power law.
    
    k=2 for now.
    c(x) = 1/(1-y)
    """
    assert(k==2)
    return 1/(1-x)


def power_law_wcf_cdf(x, c):
    """Power law with cutoff.
    
    H: x>=1
    Arguments:
    x : numpy array
    c : cuttoff
    """
    res = np.zeros(shape=x.shape)
    mask = x>c
    maski = mask==False
    res[maski] = power_law_cdf(x[maski],k=2)
    res[mask] = (c-1.0)/c + 1/c * cutoff(x[mask]/c-1)
    return res

def power_law_wcf_cdf_inv(x, c):
    """Inverse power law with cutoff.
    
    H: x>=1
    Arguments:
    x : numpy array
    c : cuttoff
    """
    res = np.zeros(shape=x.shape)
    mc = power_law_cdf(c,k=2)
    mask = x>mc
    maski = mask==False
    res[maski] = power_law_cdf_inv(x[maski],k=2)
#     res[mask] = np.round(c*(1 - np.log(c*(1-x[mask]))))    
    res[mask] = np.round( c*(cutoff_inv(c*(x[mask]-1)+1) +1) )
    return res

# def cutoff2(x):
#     return 1-1/np.sqrt(1+x)
# def cutoff2_inv(x):
#     return 1/((1-x)**2)-1

# def cutoff(x, p=2):
#     return 1-1/((1+x)**(1/p))

# def cutoff_inv(x, p=2):
#     return 1/((1-x)**p)-1

def cutoff(x):
    return 1 - np.exp(-x)
def cutoff_inv(x):
    return - np.log(1-x)

def laplacian_map_from_cdf_forward(x, pdf):
    cp = pdf(x)
    return -np.log(1-cp)

def laplacian_map_from_cdf_backward(x, pdf, pdf_inv, clip_max=1e6):
    v = np.array([clip_max])
    x_lim = laplacian_map_from_cdf_forward(v, pdf)
    x = np.clip(x,0,x_lim)
    cl = 1 - np.exp(-x)
    return pdf_inv(cl)



def stat_forward(x, c=1e5, shift=6):
    pdf = functools.partial(power_law_wcf_cdf, c=c)
    sv = laplacian_map_from_cdf_forward(np.array([1 + shift]), pdf)[0]
    return laplacian_map_from_cdf_forward(x+1 + shift, pdf) - sv

def stat_backward(x, c=1e5, shift=6):
    clip_max = c*20
    pdf = functools.partial(power_law_wcf_cdf, c=c)
    pdf_inv = functools.partial(power_law_wcf_cdf_inv, c=c)
    sv = laplacian_map_from_cdf_forward(np.array([1 + shift]), pdf)[0]
    return np.round(laplacian_map_from_cdf_backward(x+ sv, pdf, pdf_inv, clip_max=clip_max)-1-shift)


forward = nati_forward
backward = nati_backward

