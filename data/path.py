import socket
import os

def data_path(spix=256):  
    ''' Will be removed in the futur '''
    return root_path() + 'size{}_splits1000_n500x3/'.format(spix)

def root_path():
    ''' Defining the different root path using the host name '''
    hostname = socket.gethostname()
    # Check if we are on pizdaint
    if 'nid' in hostname:
        # Mhis to the store folder to be able to all use it?
        # For reading it is ok.
        rootpath = '/scratch/snx3000/nperraud/pre_processed_data/' 
    elif 'omenx' in hostname:
        rootpath = '/store/nati/datasets/comsology/pre_processed_data/'         
    else:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        rootpath = utils_module_path + '/../../data/'
    return rootpath