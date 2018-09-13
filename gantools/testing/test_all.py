if __name__ == '__main__':
	import sys, os
	sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest

from gantools.testing import test_downsample
from gantools.testing import test_metric
from gantools.testing import test_split
from gantools.testing import test_fmap

loader = unittest.TestLoader()

suites = []
suites.append(loader.loadTestsFromModule(test_downsample))
suites.append(loader.loadTestsFromModule(test_metric))
suites.append(loader.loadTestsFromModule(test_split))
suites.append(loader.loadTestsFromModule(test_fmap))
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    os.environ["CUDA_VISIBLE_DEVICES"]=""

    run()