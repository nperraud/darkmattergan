if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from gantools.model import WGAN
from gantools.gansystem import GANsystem
from gantools.data.Dataset import Dataset
import numpy as np


class TestGANsystem(unittest.TestCase):
    def test_default_params(self):
        wgan = GANsystem(WGAN)

    def test_gansystem(self):
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        X = np.random.rand(101,16,16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        img = wgan.generate(500)
        assert(len(img)==500)

if __name__ == '__main__':
    unittest.main()