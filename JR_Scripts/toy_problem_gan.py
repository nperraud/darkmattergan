import dict_reader, utils,  sys
from Data_Generators import time_toy_generator
from model import TempConsGanModel
from gan import GAN
import numpy as np


def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year) + '_' + str(d.month)+ '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute)


def main():
    # Load parameters
    param_paths = sys.argv[1]
    params = dict_reader.read_gan_dict(param_paths)
    params['name'] = 'WGAN{}'.format(params['image_size'][0])
    time_str = current_time_str()
    params['summary_dir'] = 'tboard/' + params['name'] + '_' + time_str + 'summary/'
    params['save_dir'] = 'checkp/' + params['name'] + '_' + time_str + 'checkpoints/'
    params['num_classes'] = 4
    # Initialize model
    wgan = GAN(params, TempConsGanModel)
    # Generate data
    data = time_toy_generator.gen_dataset(images_per_time_step=5000, point_density_factor=75)
    data = np.asarray([data[0], data[3], data[6], data[9]])
    data = data.swapaxes(0,1)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
    data = data.astype(np.float32)
    data = utils.forward_map(data, params['k'])
    # Train model
    wgan.train(data)
    return 0


main()