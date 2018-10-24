# Module to download the dataset.

import os

from gantools import utils

if __name__ == '__main__':
    # The dataset is availlable at https://doi.org/10.5281/zenodo.1303272
    
    url_nbody = 'https://zenodo.org/record/1464832/files/nbody-cubes.zip?download=1'
    url_readme = 'https://zenodo.org/record/1464832/files/README.md?download=1'

    md5_nbody = 'abc89d98e60d94fda703f5d176594dd9'
    md5_readme = '052c060c4f8e0e23699de76e65db557d'

    print('Download README')
    utils.download(url_readme, 'data/nbody')
    assert (utils.check_md5('data/nbody/README.md', md5_readme))

    print('Download nbody-cubes')
    utils.download(url_nbody, 'data/nbody/preprocessed_data')
    assert (utils.check_md5('data/nbody/preprocessed_data/nbody-cubes.zip', md5_nbody))
    print('Extract nbody-cubes')
    utils.unzip('data/nbody/preprocessed_data/nbody-cubes.zip', 'data/nbody/')
    os.remove('data/nbody/preprocessed_data/nbody-cubes.zip')


