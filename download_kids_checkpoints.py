# Module to download the dataset.

import os

import sys
if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import hashlib
import zipfile


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def check_md5(file_name, orginal_md5):
    # Open,close, read file and calculate MD5 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.md5()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (
                1 << 20))  # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    md5_returned = hasher.hexdigest()
    # Finally compare original MD5 with freshly calculated
    if orginal_md5 == md5_returned:
        print('MD5 verified.')
        return True
    else:
        print('MD5 verification failed!')
        return False


def unzip(file, targetdir):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(targetdir)


if __name__ == '__main__':
    # The checkpoints can be downloaded at https://zenodo.org/record/4651586
    
    url_checkpoints = 'https://zenodo.org/record/4651586/files/kids_checkpoints.zip?download=1'
    url_readme = 'https://zenodo.org/record/4651586/files/README.md?download=1'

    md5_checkpoints = 'c49df80d4067a4ca6744e3ff0af67ca7'
    md5_readme = '666da8da6ae55f2f35c4c186ef53ed1d'


    print('Download checkpoints')
    download(url_checkpoints, './')
    assert(check_md5('kids_checkpoints.zip', md5_checkpoints))
    print('Extract checkpoints')
    unzip('kids_checkpoints.zip', 'saved_results/')
    os.remove('kids_checkpoints.zip')
    
    print('Download README')
    download(url_readme, 'saved_results/Kids')
    assert (check_md5('saved_results/Kids/README.md', md5_readme))
