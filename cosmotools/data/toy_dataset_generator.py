import numpy as np
from scipy.stats import multivariate_normal

def gaussian_2d(mu, sigma = 1.0):
    rv = multivariate_normal(mu, np.eye(2) * sigma)
    return rv

def generate_fake_image(x, y, sigma = 0.0001, n = 50, normalise=True):
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y

    # Generate random samples
    # TODO: n + 1 clusters are generated instead
    mu = np.random.rand(2)
    data = gaussian_2d(mu, sigma).pdf(pos)
    for i in range(n):
        mu = np.random.rand(2)
        data =  data + gaussian_2d(mu, sigma).pdf(pos)
    if normalise:
        data = data / np.max(data)
    return data

def generate_fake_images(nsamples=10, sigma=0.0001, N=100, image_shape=[32, 32], normalise=True):
    inter = 1.0 / np.array(image_shape)
    x, y = np.mgrid[0:1:inter[0], 0:1:inter[1]]
    return np.array([generate_fake_image(x, y, sigma, N, normalise) for i in range(nsamples)])

def generate_fake_dataset(sigma_int=[0.0001, 0.01], N_int=[10, 50], image_shape=[32, 32], nsamples=10000, normalise=True):
    inter = 1.0 / np.array(image_shape)
    x, y = np.mgrid[0:1:inter[0], 0:1:inter[1]]
    images = np.zeros((nsamples, image_shape[0], image_shape[1]))
    params = []
    for i in range(nsamples):

        # Get random parameters
        sigma = sigma_int[0] + np.random.rand() * (sigma_int[1] - sigma_int[0])
        N = np.random.randint(N_int[0], N_int[1])

        # Generate fake image
        images[i] = generate_fake_image(x, y, sigma, N, normalise)
        params.append([sigma, N])
    params = np.array(params)
    return [images, params]