if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from gantools.model import WGAN
from gantools.gansystem import GANsystem
from gantools.data.Dataset import Dataset
import numpy as np


class TestGANmodels(unittest.TestCase):
    def test_default_params(self):
        wgan = GANsystem(WGAN)

    def test_2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 16, 1] # Shape of the image
        params['net']['generator']=dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 8 * 8]
        params['net']['generator']['nfilter'] = [2, 32, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn, bn]
        params['net']['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        params['net']['generator']['stride'] = [1, 2, 1, 1]
        params['net']['generator']['is_3d'] = False
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [5, 5], [5, 5], [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2, 2, 1]
        params['net']['discriminator']['is_3d'] = False

        X = np.random.rand(101,16,16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert(len(img)==2)
        assert(img.shape[1:]==(16,16,1))
        img = wgan.generate(500)
        assert(len(img)==500)

    def test_3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 1] # Shape of the image
        params['net']['generator']=dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 4 * 4 *4]
        params['net']['generator']['nfilter'] = [2, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3], [5, 5, 5]]
        params['net']['generator']['stride'] = [1, 2, 1]
        params['net']['generator']['is_3d'] = True
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5, 5], [3, 3, 3], [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2, 2]
        params['net']['discriminator']['is_3d'] = True


        X = np.random.rand(101,8,8,8)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert(len(img)==2)
        assert(img.shape[1:]==(8,8,8,1))
        img = wgan.generate(500)
        assert(len(img)==500)

if __name__ == '__main__':
    unittest.main()