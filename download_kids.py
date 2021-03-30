# Module to download the dataset.

import os

import sys
if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import hashlib
import zipfile
import zlib
import shutil


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


def md5(file_name):
    # Open,close, read file and calculate MD5 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.md5()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (
                1 << 20))  # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    return hasher.hexdigest()
   
    

def check_md5(file_name, orginal_md5):

    # Finally compare original MD5 with freshly calculated
    md5_returned = md5(file_name)
    if orginal_md5 == md5_returned:
        print('MD5 verified.')
        return True
    else:
        print('MD5 verification failed!')
        return False


def unzip(file, targetdir):
    with zipfile.ZipFile(file, "r", zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.extractall(targetdir)

if __name__ == '__main__':
    # The dataset is availlable at https://zenodo.org/record/4564408
    
    base_url = "https://zenodo.org/record/4646764/files/"

    d_hash = {}
    d_hash["description.md"] = "8e87303f686e31ac977c55af9fa5a330"
    d_hash["cosmo.par"] = "674997a97f5f86b944c45793d378c0fc"
    d_hash['kappa_map_cosmo_0.292_0.966.zip'] = '94b6c8903f66d60f2a637e7deb2ecf28'
    d_hash['kappa_map_cosmo_0.196_1.225.zip'] = '2e3f120a09022899a662c5eaa24ead8b'
    d_hash['kappa_map_cosmo_0.161_1.119.zip'] = 'a6efa2a2221349beedd956954691271f'
    d_hash['kappa_map_cosmo_0.101_1.304.zip'] = 'da95c482bdd1c9afc2caa944d87f6035'
    d_hash['kappa_map_cosmo_0.402_0.570.zip'] = '03bc8316fb41e085dcc09c326fcbeff4'
    d_hash['kappa_map_cosmo_0.332_0.724.zip'] = '96e15807f1147d50e853b28e2d1ffeae'
    d_hash['kappa_map_cosmo_0.137_1.230.zip'] = '645258fe18db47df86392019d9c3ea63'
    d_hash['kappa_map_cosmo_0.233_0.791.zip'] = 'a19b345ef0b5415b3543fa5eb4d80e7d'
    d_hash['kappa_map_cosmo_0.188_1.019.zip'] = '39d9bbf55e107459f2f718dd085d05a0'
    d_hash['kappa_map_cosmo_0.431_0.475.zip'] = '9467db07a648f3e3fcde332b1248339e'
    d_hash['kappa_map_cosmo_0.395_0.730.zip'] = 'bfc8ec08e5f4a57f1ba2606151c032ff'
    d_hash['kappa_map_cosmo_0.225_1.129.zip'] = '83d18a5b619fc6afafe28c401504e8ba'
    d_hash['kappa_map_cosmo_0.250_0.658.zip'] = 'fdae86cf6e3a8e09e483cd03e937b6ab'
    d_hash['kappa_map_cosmo_0.314_0.487.zip'] = '349b16a4689d04dc63d25036e1906b63'
    d_hash['kappa_map_cosmo_0.291_0.601.zip'] = '5f5f8aec49b578fe00f50e7c4e314610'
    d_hash['kappa_map_cosmo_0.178_0.807.zip'] = '1808c9252bfd1b74b533756b17fb027a'
    d_hash['kappa_map_cosmo_0.273_0.910.zip'] = 'ee68213af4f9a505f37ee53107a04d30'
    d_hash['kappa_map_cosmo_0.311_0.842.zip'] = '79b77fffa9d1512dec7f414a9503cf74'
    d_hash['kappa_map_cosmo_0.123_1.006.zip'] = '19d0e32ad68f42045c93d3a115cf070c'
    d_hash['kappa_map_cosmo_0.154_1.281.zip'] = 'd11ef7960132feef8affed6a6153fd2b'
    d_hash['kappa_map_cosmo_0.421_0.628.zip'] = '92cf905af0cc531265e8fafef6739b93'
    d_hash['kappa_map_cosmo_0.269_0.534.zip'] = '0496fc2e1255e7333a40946dcf83d44b'
    d_hash['kappa_map_cosmo_0.169_0.961.zip'] = '734939b5a8c18e1fc527e205f9fc1ff2'
    d_hash['kappa_map_cosmo_0.291_0.783.zip'] = '844deb66cd99465e39bafba521a26817'
    d_hash['kappa_map_cosmo_0.171_1.331.zip'] = '9309f34f1ece074e3a76a49a6968103d'
    d_hash['kappa_map_cosmo_0.413_0.784.zip'] = '8b5ac4d799594c87902dae3cddfd8659'
    d_hash['kappa_map_cosmo_0.219_0.930.zip'] = 'f273ddf2abde38aabd79a3fafcc728ec'
    d_hash['kappa_map_cosmo_0.254_0.852.zip'] = 'c610d5472f1e0881a1868e94bf47531b'
    d_hash['kappa_map_cosmo_0.450_0.533.zip'] = 'ff7427919da3f6a6e2785bb9c59e1b7e'
    d_hash['kappa_map_cosmo_0.179_1.173.zip'] = '198787284b7b710e0586846a284f5c70'
    d_hash['kappa_map_cosmo_0.312_0.664.zip'] = '219bd199d383154a8e1e105cb4ef1b9e'
    d_hash['kappa_map_cosmo_0.370_0.838.zip'] = '8430d0f116e423c189d5b038184f4c02'
    d_hash['kappa_map_cosmo_0.212_0.727.zip'] = 'bf6bdab7a1d4c8634187cacc6a7898c7'
    d_hash['kappa_map_cosmo_0.238_0.988.zip'] = 'f0ee7f943b981adf3501fc93f750c7cd'
    d_hash['kappa_map_cosmo_0.257_1.043.zip'] = '6b6055c62b8c9a1b9138be75e17eacc9'
    d_hash['kappa_map_cosmo_0.487_0.643.zip'] = '90c5e503f48a5d4eebbd823dc445d8e6'
    d_hash['kappa_map_cosmo_0.271_0.723.zip'] = '6a0e886e16da84e40c3ddf85e9454eb8'
    d_hash['kappa_map_cosmo_0.356_0.614.zip'] = 'ca400fe14b99a5d3e760c9f32f45d438'
    d_hash['kappa_map_cosmo_0.142_1.063.zip'] = '7eb2bb20a53215cbb511caef3032fe51'
    d_hash['kappa_map_cosmo_0.335_0.552.zip'] = 'fa3ca459aef819098df488bb55451aa2'
    d_hash['kappa_map_cosmo_0.189_0.659.zip'] = '97333df361f8f0ca3c2305256ce3fd68'
    d_hash['kappa_map_cosmo_0.199_0.870.zip'] = 'c19667b163c02e912fb19434d49fa9b2'
    d_hash['kappa_map_cosmo_0.102_1.125.zip'] = '04b3ea416f4741f782f2b82990e7c562'
    d_hash['kappa_map_cosmo_0.469_0.589.zip'] = '92f0193ae3e9e72d83769dca60305d1e'
    d_hash['kappa_map_cosmo_0.440_0.683.zip'] = 'ebf1dd7cbe1ed4c4e6ebf30361cf3703'
    d_hash['kappa_map_cosmo_0.376_0.673.zip'] = '0a145ce16ed4729c1c46fde1ca0f9b3c'
    d_hash['kappa_map_cosmo_0.352_0.782.zip'] = '4de96da24e8e433a7340980181cd5646'
    d_hash['kappa_map_cosmo_0.207_1.075.zip'] = '5dba1638d24f0eec9b839adb0a9d876f'
    d_hash['kappa_map_cosmo_0.120_1.178.zip'] = '50a1f29162b6d54e00dc69f601fdcb86'
    d_hash['kappa_map_cosmo_0.156_0.741.zip'] = '183fed204b16ddfdd6cd973a697398d3'
    d_hash['kappa_map_cosmo_0.103_0.947.zip'] = 'bbfaf486647a65273de32f26aa622af7'
    d_hash['kappa_map_cosmo_0.148_0.900.zip'] = '102a59fd4d762f8e5c0c159af4493344'
    d_hash['kappa_map_cosmo_0.127_0.836.zip'] = '0b07e9807353e9c4dff4b6aada90a655'
    d_hash['kappa_map_cosmo_0.227_0.591.zip'] = '700c7a1589771a8644af256cee60dfae'
    d_hash['kappa_map_cosmo_0.330_0.898.zip'] = '7b6a142bc8210638ea95a836a3c4354d'
    d_hash['kappa_map_cosmo_0.458_0.737.zip'] = '11e57510d95d7e6d49cd9d26c3ce3af3'
    d_hash['kappa_map_cosmo_0.382_0.510.zip'] = 'b33891226c6870b039385ef23f7bf81b'

    for k, v in d_hash.items():
        print("Download: {}".format(k))
        url = base_url+ k + "?download=1"
        path = 'data/KiDs450_maps/'
        download(url, path)
        assert (check_md5(path + k, v))
        if k[-4:] == ".zip":
            print("  unzip {}".format(k))
            unzip(path + k, path)
            os.remove(path + k)
