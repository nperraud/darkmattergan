import socket
import os

def data_path(spix=256):  
    return root_path() + 'size{}_splits1000_n500x3/'.format(spix)

def root_path():
    # Check if we are on pizdaint
    if 'nid' in socket.gethostname():
        # Mhis to the store folder to be able to all use it?
        # For reading it is ok.
        rootpath = '/scratch/snx3000/nperraud/nati-gpu/data/' 
    else:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        rootpath = utils_module_path + '/../../data/'
    return rootpath