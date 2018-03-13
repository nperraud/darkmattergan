import numpy as np
import os
import pynbody
import h5py
import utils

def load_pynbody(fname_cuboid, PATH_READ, PATH_WRITE, cube_id):
    '''
    fname_cuboid = file within folders within each of the 10 boxes
    PATH_READ = path where the 10 boxes are stored
    PATH_WRITE = path where new data would be saved
    cube_id = could be any of the 10 boxes
    '''

    PATH_filename = os.path.join(PATH_READ, cube_id, fname_cuboid)
    if not os.path.exists(PATH_filename):
        raise ValueError("PATH to the nbody cube {} doesn't exist".format(PATH_filename))
    try:
        nbody = pynbody.load(PATH_filename)
    except:
        print("File {} is not a pynbody simulation".format(PATH_filename))
        return None, -1

    with open(os.path.join(PATH_WRITE, 'FLAGS_' + cube_id + '.txt'), 'a') as f:
        f.write('\n\n')
        f.write('Properties of file {}:\n'.format(fname_cuboid))
        for key in nbody.properties:
            value = nbody.properties[key]
            f.write('%s:%s\n' % (key, value))

    nbody.physical_units()
    nbody['pos'].convert_units('Mpc') ## nbody['pos'] = 3d coordinates of all the particles

    lbox = nbody.properties['boxsize'].in_units('Mpc')

    return nbody, lbox

def get_slice_edges(num_slices=10, lbox = 500):
    '''
    num_slices+1 cuts along each edge.
    return the begin and end index of each slice
    '''
    edge_locations = np.linspace(0, lbox, num_slices + 1, dtype=np.float64)
    ind_beg = np.arange(0, len(edge_locations)-1, 1)
    ind_end = ind_beg + 1
    edges_beg = edge_locations[ind_beg]
    edges_end = edge_locations[ind_end]
    return np.vstack( (edges_beg, edges_end) ).T

def slice_cuboid(edges, nbody_i, PATH_WRITE, cube_id):
    '''
    edges = (num_slices*2) : the beginning and ending of all the cuts
    slice the bigger nbody cube into smaller cubes, along all 3 axes, as per the cuts in edges array
    dump to disk: each smaller cube is represented by bottom left point. Insert the cubes with their key
    in the small_cubes dictionary. The smaller cuboids, within each nbody cube in the input, 
    has along y and z directions 500 Mpc, but along x direction, the thickness is less. We iteratively pass
    each cuboid to this function, and add to the file on disk, the points belonging to the sliced
    cubes, as per the key of the dictionary
    '''
    cuboid = nbody_i['pos']
    x_min = min(cuboid[:,0])
    x_max = max(cuboid[:,0])
    print("x_min={}  x_max={}".format(x_min, x_max))

    small_cubes = {}
    
    for x_edge in edges:
        if (x_edge[0] > x_max) or (x_edge[1] < x_min):
            continue
            
        for y_edge in edges:
            for z_edge in edges:
                key = (x_edge[0], y_edge[0], z_edge[0])

                small_cube_ind = ((x_edge[0] <= cuboid[:,0]) & (cuboid[:,0] < x_edge[1]) ) & \
                                ((y_edge[0] <= cuboid[:,1]) & (cuboid[:,1] < y_edge[1]) ) & \
                                ((z_edge[0] <= cuboid[:,2]) & (cuboid[:,2] < z_edge[1]) )
                small_cube = cuboid[small_cube_ind] #pull out the rows that belong to the current small cube

                if key in small_cubes:
                    small_cubes[key] = np.vstack( (small_cubes[key], small_cube) )
                else:
                    small_cubes[key] = np.array(small_cube) # take the points from the SimArray, form np array

    #if the directory does not exist, create it 
    dir_path = os.path.join(PATH_WRITE, cube_id)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print("start dumping small cubes to disk")

    #check if file with this key exists. If yes, append to it, else create the file
    for key, points in small_cubes.items():
        file_name = os.path.join(dir_path, str(key) + '.h5')

        if not os.path.exists(file_name):
            utils.save_hdf5(data=points, filename=file_name, dataset_name='data', mode='w')

        else:
            arr = utils.load_hdf5(filename=file_name, dataset_name='data', mode='r')
            arr = np.vstack((arr, points))
            utils.save_hdf5(data=arr, filename=file_name, dataset_name='data', mode='w')

def slice_cube(PATH_READ, PATH_WRITE, cube_id, num_slices=50, lbox=500):
    '''
    slice the cube with id cube_id, and write the smaller cubes to PATH_WRITE
    '''
    edges = get_slice_edges(num_slices, lbox)
    #print(edges)
    
    small_cubes={}
    # load cuboids one by one, and keep updating the smaller cubes
    cuboid_num = 0
    for filename in os.listdir( os.path.join(PATH_READ, cube_id) ):
        if not filename.endswith(".txt") and not filename.endswith(".dat") and not filename.endswith(".info"):
            print("---------------------------------------------------Current cuboid = {} num={}".format(filename, cuboid_num))
            cuboid_num += 1
            nbody_i, lbox_i = load_pynbody(fname_cuboid=filename, PATH_READ=PATH_READ, PATH_WRITE=PATH_WRITE, cube_id=cube_id)
            #print("nbody_i={}   lbox_i={} Mpc".format(nbody_i, lbox_i) )
            slice_cuboid(edges, nbody_i, PATH_WRITE, cube_id)

def small_cubes_to_3d_hist(PATH, cube_id, small_cube_dim=16):
    '''
    Returns a dictionary of 3d histograms for a single nbody bigger cube, where:
    key = bottom left point of the smaller cube
    value = ((small_cube_dim*small_cube_dim*small_cube_dim), list of 3 arrays, each of length small_cube_dim+1, denoting the edges of
              the bins along x,y, z axes respectively)
    '''
    path_write_hist = os.path.join(PATH, cube_id + 'hist')
    if not os.path.exists(path_write_hist):
        os.makedirs(path_write_hist)

    for filename in os.listdir( os.path.join(PATH, cube_id) ):
        file_path = os.path.join(PATH, cube_id, filename)
        points = utils.load_hdf5(filename=file_path, dataset_name='data', mode='r')
        hist_3d = np.histogramdd(sample=points, bins=small_cube_dim)
            
        hist_file_path = os.path.join(path_write_hist, filename)
        utils.save_hdf5(data=hist_3d[0], filename=hist_file_path, dataset_name='data', mode='w')

def read_small_cube_from_disk(PATH, cube_id, filename):
    file_path = os.path.join(PATH, cube_id, filename)
    points = utils.load_hdf5(filename=file_path, dataset_name='data', mode='r')
    return points

def read_3d_hist_from_disk(PATH, cube_id, filename):
    file_path = os.path.join(PATH, cube_id + 'hist', filename)
    hist_3d = utils.load_hdf5(filename=file_path, dataset_name='data', mode='r')
    return hist_3d

def main():
    PATH_WRITE = '../3d_smaller_cubes'
    #if the directory does not exist, create it
    if not os.path.exists(PATH_WRITE):
        os.makedirs(PATH_WRITE)
    
    cube_id_root = 'Box_350Mpch_'
    PATH_READ = '../../nbody/'

    for box_num in range(1):
        print("---------------------------------------------Current box = {}".format(box_num))
        cube_id = cube_id_root + str(box_num)
        slice_cube(PATH_READ, PATH_WRITE, cube_id, num_slices=10, lbox=500)

        # verify that nbody cubes were sliced properly
        nbody = []
        lbox = []
        for filename in os.listdir( os.path.join(PATH_READ, cube_id)):
            if not filename.endswith(".txt") and not filename.endswith(".dat") and not filename.endswith(".info"):
                nbody_i, lbox_i = load_pynbody(fname_cuboid=filename, PATH_READ=PATH_READ, PATH_WRITE=PATH_WRITE, cube_id=cube_id)
                nbody.append(nbody_i)
                lbox.append(lbox_i)

        t2 = 0
        for nbody_i in nbody:
            t2 += len(nbody_i['pos'])

        t1 = 0
        for filename in os.listdir( os.path.join(PATH_WRITE, cube_id)):
            points = read_small_cube_from_disk(PATH_WRITE, cube_id, filename)
            t1 += len(points)

        print("t1=", t1)
        print("t2=", t2)
        print("t1==t2: ", t1 == t2)

        print("Writing 3d Histogram to disk")
        cube_dim=16
        small_cubes_to_3d_hist(PATH_WRITE, cube_id, cube_dim)
        print("Done!")
    

if __name__ == '__main__':
    main()


