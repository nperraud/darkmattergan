import os

def root_path():
    ''' Defining the different root path for the nbody dataset'''
    # This should be done in a different way
    utils_module_path = os.path.dirname(__file__)
    rootpath = utils_module_path + '/../../data/nbody/'
    return rootpath


def root_path_kids():
    ''' Defining the different root path the kids dataset'''
    # This should be done in a different way
    utils_module_path = os.path.dirname(__file__)
    rootpath = utils_module_path + '/../../data/KiDs450_maps/'
    return rootpath