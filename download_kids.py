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
    with zipfile.ZipFile(file, "r", zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.extractall(targetdir)

if __name__ == '__main__':
    # The dataset is availlable at https://zenodo.org/record/4564408
    
    base_url = "https://zenodo.org/record/4564408/files/"
    
    d_hash = {}
    d_hash["description.md"] = "8e87303f686e31ac977c55af9fa5a330"
    d_hash["cosmo.par"] = "674997a97f5f86b944c45793d378c0fc"
#     d_hash["kappa_map_cosmo_0.101_1.304.zip"] = "0314004a3aa54b9751cd14903fcd3875"		
#     d_hash["kappa_map_cosmo_0.102_1.125.zip"] = "17f25b7ab5a253cdc3f00e1c676f2ce2"		
#     d_hash["kappa_map_cosmo_0.103_0.947.zip"] = "e13271431aa7a0661ca95858b7718089"		
#     d_hash["kappa_map_cosmo_0.120_1.178.zip"] = "c5b32c223ac2cff83d253079c8d66849"		
#     d_hash["kappa_map_cosmo_0.123_1.006.zip"] = "1aa6c8273077ca080fed6b7c7024752c"		
#     d_hash["kappa_map_cosmo_0.127_0.836.zip"] = "d735f95a190ef80d8cf791641ba13460"		
#     d_hash["kappa_map_cosmo_0.137_1.230.zip"] = "b47fe6bdd91d03ef782699a04bfe530a"		
#     d_hash["kappa_map_cosmo_0.142_1.063.zip"] = "6b50ba35919aa8a17d1dc4ab4f4d6cb6"		
#     d_hash["kappa_map_cosmo_0.148_0.900.zip"] = "a3e6cd80af713d8480c3e9a8d6a4a290"		
#     d_hash["kappa_map_cosmo_0.154_1.281.zip"] = "44bc466ba1e4d7e3d1843f0ca8c851e4"		
#     d_hash["kappa_map_cosmo_0.156_0.741.zip"] = "aa395e6c05dc0a23720c8e46d2e3bf3e"		
#     d_hash["kappa_map_cosmo_0.161_1.119.zip"] = "281a7f0a901a5354ed87a7170c3dad4e"		
#     d_hash["kappa_map_cosmo_0.169_0.961.zip"] = "907eee7f02e64723960943e00487cd8c"		
#     d_hash["kappa_map_cosmo_0.171_1.331.zip"] = "f15a416f5c7b52e8a0d27b381db4aabd"		
#     d_hash["kappa_map_cosmo_0.178_0.807.zip"] = "103dadde2e0103ff6980349e793ba3b9"		
#     d_hash["kappa_map_cosmo_0.179_1.173.zip"] = "1c2a913e6d523f86e068cb05dbc31824"		
#     d_hash["kappa_map_cosmo_0.188_1.019.zip"] = "c7b13b39fb051f7ef402db817d03373b"		
#     d_hash["kappa_map_cosmo_0.189_0.659.zip"] = "426b9ff3d4d7d906178c328c9401f58c"		
#     d_hash["kappa_map_cosmo_0.196_1.225.zip"] = "a74d5b65fdf03273c995d569183d0f8a"		
#     d_hash["kappa_map_cosmo_0.199_0.870.zip"] = "b32ea074d5ee238fc1d0c80e691dc756"		
#     d_hash["kappa_map_cosmo_0.207_1.075.zip"] = "074cd4eb11cd45569b9d513b97c9d7a5"		
#     d_hash["kappa_map_cosmo_0.212_0.727.zip"] = "820e61bfba167ae70c9cbdc73f2d1c1e"		
#     d_hash["kappa_map_cosmo_0.219_0.930.zip"] = "61ca968d01cb00d845eabdfc9ba4a72c"		
#     d_hash["kappa_map_cosmo_0.225_1.129.zip"] = "e2779c3fd31c7d054b4f8bb171d91a0f"		
#     d_hash["kappa_map_cosmo_0.227_0.591.zip"] = "32ca7c8fec544f71cb72989fa9a3d318"		
#     d_hash["kappa_map_cosmo_0.233_0.791.zip"] = "6413ec686d60483faf451c44978201e2"		
    d_hash["kappa_map_cosmo_0.238_0.988.zip"] = "3320c0fde1d7a5c888a16c6362491941"		
#     d_hash["kappa_map_cosmo_0.250_0.658.zip"] = "573f09fb954ff6255b14246da6780e1f"		
#     d_hash["kappa_map_cosmo_0.254_0.852.zip"] = "4de10c2f07f5d0d9258870a94652b58d"		
#     d_hash["kappa_map_cosmo_0.257_1.043.zip"] = "dbe305f1ccc526983b8257f1ed5333c9"		
#     d_hash["kappa_map_cosmo_0.269_0.534.zip"] = "c6ce846b952761297e8fb78f21d1f3bb"		
#     d_hash["kappa_map_cosmo_0.271_0.723.zip"] = "b4e753f23c9037a6e7cc8e2ffb40779d"		
#     d_hash["kappa_map_cosmo_0.273_0.910.zip"] = "7ff826f9aaba8e47c39b84fbdae16506"		
#     d_hash["kappa_map_cosmo_0.291_0.601.zip"] = "1b6b6ab1dbd358b44c25d67fbf8d8e53"		
#     d_hash["kappa_map_cosmo_0.291_0.783.zip"] = "72a32fd2f05ba18b2c43619223b10b7e"		
#     d_hash["kappa_map_cosmo_0.292_0.966.zip"] = "9e5794b2572bb74ba857786552c0407b"		
#     d_hash["kappa_map_cosmo_0.311_0.842.zip"] = "d9d8a52f78547ecbe726978bafa250b7"		
#     d_hash["kappa_map_cosmo_0.312_0.664.zip"] = "521aa5baaafaaa439a0fe21d8a4e0c8a"		
#     d_hash["kappa_map_cosmo_0.314_0.487.zip"] = "6c2bc0f429b3f31c6624a122cc7d147d"		
#     d_hash["kappa_map_cosmo_0.330_0.898.zip"] = "e304f32f99de9bb968c38e4187810a42"		
#     d_hash["kappa_map_cosmo_0.332_0.724.zip"] = "8180cffa7bc4f0438282f3e85f80279f"		
#     d_hash["kappa_map_cosmo_0.335_0.552.zip"] = "3b296dbe4d9bc9eac59156d8e882d4eb"		
#     d_hash["kappa_map_cosmo_0.352_0.782.zip"] = "b26bd2519b4b4b2f65b05e814bc7c2c4"		
#     d_hash["kappa_map_cosmo_0.356_0.614.zip"] = "3920f06e1fb3c22608855c77e2470d10"		
#     d_hash["kappa_map_cosmo_0.370_0.838.zip"] = "7fd71938e2f12c175a2cf26143eca6c9"		
#     d_hash["kappa_map_cosmo_0.376_0.673.zip"] = "d76e5866a742408d3b9fc13cf1a39017"		
#     d_hash["kappa_map_cosmo_0.382_0.510.zip"] = "6c044451bb07ea90338f69b55060765d"		
#     d_hash["kappa_map_cosmo_0.395_0.730.zip"] = "b053d1bf345432a05f5d6cfbdf850744"		
#     d_hash["kappa_map_cosmo_0.402_0.570.zip"] = "92c57402906e5fcd7bc35a172742d33c"		
#     d_hash["kappa_map_cosmo_0.413_0.784.zip"] = "e08940c98769e46ae7988d1d4c9cb50f"		
#     d_hash["kappa_map_cosmo_0.421_0.628.zip"] = "234177dce763e0ec165f31949ba5a562"		
#     d_hash["kappa_map_cosmo_0.431_0.475.zip"] = "89f0dec87c4eee389db6a8d7736c8ff6"		
#     d_hash["kappa_map_cosmo_0.440_0.683.zip"] = "0a3a4236441cb66bb4457a5092e9eed4"		
#     d_hash["kappa_map_cosmo_0.450_0.533.zip"] = "bf681d3f7551093ab476101f67fcbeeb"		
#     d_hash["kappa_map_cosmo_0.458_0.737.zip"] = "b776a983f690b058487ca1e81b17244b"		
#     d_hash["kappa_map_cosmo_0.469_0.589.zip"] = "19eaf1d2740549e3dddb41a1e2dcef6c"		
#     d_hash["kappa_map_cosmo_0.487_0.643.zip"] = "e33737fd824d025a411096766f6c2ba4"

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
            shutil.move(path + "bigguy/nati/cosmogan/data/KiDs450_maps/*", path)
            shutil.rmtree(path + "bigguy")
