import socket
import os

def root_path():
    ''' Defining the different root path using the host name '''
    hostname = socket.gethostname()
    # Check if we are on pizdaint
    if 'nid' in hostname:
        rootpath = '/scratch/snx3000/nperraud/' 
    elif 'omenx' in hostname:
        rootpath = '/store/nati/datasets/cosmology/'         
    else:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        rootpath = utils_module_path + '/../../data/nbody/'
    return rootpath