if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from gantools.data import load, fmap
import numpy as np
from gantools.blocks import np_downsample_2d, np_downsample_3d, np_downsample_1d

class TestGANmodels(unittest.TestCase):
    def test_cosmo(self):
        forward = fmap.stat_forward
        # dataset = load.load_dataset(
        #     nsamples=None, spix=32, Mpch=350, forward_map=forward, patch=True)
        # it = dataset.iter(10)
        # print(next(it).shape)
        # assert (next(it).shape == (10, 32, 32, 4))
        # del it, dataset

        # dataset = load.load_dataset(
        #     nsamples=None,
        #     spix=32,
        #     Mpch=350,
        #     forward_map=forward,
        #     patch=True,
        #     is_3d=True)
        # it = dataset.iter(4)
        # print(next(it).shape)
        # assert (next(it).shape == (4, 32, 32, 32, 8))
        # del it, dataset

        # dataset = load.load_dataset(
        #     nsamples=None, spix=32, Mpch=70, forward_map=None, patch=False)
        # it = dataset.iter(10)
        # print(next(it).shape)

        # assert (next(it).shape == (10, 32, 32))
        # del it, dataset

        # dataset = load.load_dataset(
        #     nsamples=2, spix=256, Mpch=70, forward_map=forward, patch=False)
        # assert (dataset.get_all_data().shape[0] == 256 * 2)
        # del dataset

        # dataset = load.load_dataset(
        #     nsamples=2, spix=128, Mpch=350, forward_map=forward, patch=False)
        # it = dataset.iter(10)
        # print(next(it).shape)
        # assert (next(it).shape == (10, 128, 128))
        # del it, dataset

        # dataset = load.load_dataset(
        #     nsamples=2, spix=128, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=2)
        # it = dataset.iter(10)
        # s1 = next(it)
        # del it, dataset

        # dataset = load.load_dataset(
        #     nsamples=2, spix=32, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=8)
        # it = dataset.iter(10)
        # s2 = next(it)
        # del it, dataset
        # np.testing.assert_allclose(np_downsample_2d(s1,4), s2)

        dataset = load.load_dataset(
            nsamples=2, spix=128, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=2, is_3d=True)
        it = dataset.iter(10)
        s1 = next(it)
        del it, dataset

        dataset = load.load_dataset(
            nsamples=2, spix=32, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=8, is_3d=True)
        it = dataset.iter(10)
        s2 = next(it)
        del it, dataset
        np.testing.assert_allclose(np_downsample_3d(s1,4), s2)

    def test_medical(self):
        forward = fmap.medical_forward
        dataset = load.load_medical_dataset(spix=32, scaling=8, forward_map=forward, patch=False, augmentation=False)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 32, 32, 32))
        del it, dataset

        dataset = load.load_medical_dataset(spix=32, scaling=8, forward_map=None, patch=False, augmentation=True)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 32, 32, 32))
        del it, dataset

        dataset = load.load_medical_dataset(spix=16, scaling=8, forward_map=None, patch=True, augmentation=True)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 16, 16, 16, 8))
        del it, dataset

        dataset = load.load_medical_dataset(
            spix=128, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=2)
        it = dataset.iter(10)
        s1 = next(it)
        del it, dataset

        dataset = load.load_medical_dataset(
            spix=32, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=8)
        it = dataset.iter(10)
        s2 = next(it)
        del it, dataset
        np.testing.assert_allclose(np_downsample_3d(s1,4), s2)


    def test_nsynth(self):
        dataset = load.load_nsynth_dataset(scaling=64*4)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 250))
        del it, dataset

        dataset = load.load_nsynth_dataset(scaling=64)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 1000))
        del it, dataset

        dataset = load.load_nsynth_dataset(scaling=64*4, shuffle=False)
        it = dataset.iter(5)
        s1 = next(it)
        del it, dataset

        dataset = load.load_nsynth_dataset(scaling=64, shuffle=False)
        it = dataset.iter(5)
        s2 = next(it)
        del it, dataset
        np.testing.assert_allclose(np_downsample_1d(s2,4), s1)


if __name__ == '__main__':
    unittest.main()