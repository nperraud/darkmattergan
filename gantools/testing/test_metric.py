if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest

import numpy as np

from gantools import metric


def wasserstein_distance_jonathan(x_og, y, w):
    assert (x_og.shape == y.shape == w.shape)
    x = np.copy(x_og)
    loss = 0
    for idx in range(x.shape[0] - 1):
        d = y[idx] - x[idx]
        x[idx] = x[idx] + d
        x[idx + 1] = x[idx + 1] - d
        loss = loss + np.abs(d * (w[idx + 1] - w[idx]))
    return loss / (w[-1] - w[0])


class TestMetric(unittest.TestCase):
    def test_wasserstein_distance(self):
        w = np.arange(0, 10)
        x = np.arange(0, 10, 1)
        y = np.arange(9, -1, -1)

        x = x / np.sum(np.abs(x))
        y = y / np.sum(np.abs(y))

        a = metric.wasserstein_distance(x, y, w)
        b = wasserstein_distance_jonathan(x, y, w)

        np.testing.assert_almost_equal(a, b)

        w = np.cumsum(np.random.rand(10))
        x = np.random.rand(10)
        y = np.random.rand(10)

        x = x / np.sum(np.abs(x))
        y = y / np.sum(np.abs(y))

        a = metric.wasserstein_distance(x, y, w)
        b = wasserstein_distance_jonathan(x, y, w)

        np.testing.assert_almost_equal(a, b)


if __name__ == '__main__':
    unittest.main()