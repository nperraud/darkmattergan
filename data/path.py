import socket
import os

# def data_path(spix=256):  
#     ''' Will be removed in the futur '''
#     return '/scratch/snx3000/nperraud/nati-gpu/data/size{}_splits1000_n500x3/'.format(spix)

def root_path():
    ''' Defining the different root path using the host name '''
    hostname = socket.gethostname()
    # Check if we are on pizdaint
    if 'nid' in hostname:
        # Mhis to the store folder to be able to all use it?
        # For reading it is ok.
        rootpath = '/scratch/snx3000/nperraud/pre_processed_data/' 
    elif 'omenx' in hostname:
        rootpath = '/store/nati/datasets/cosmology/pre_processed_data/'         
    else:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        rootpath = utils_module_path + '/../../pre_processed_data/'
    return rootpath