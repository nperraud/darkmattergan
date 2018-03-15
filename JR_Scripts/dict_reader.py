import numpy as np


def string_no_newline(val):
    return val.rstrip()


# unfortunately the bool() function doesn't do what we need
def boolean(val):
    return 'True' == val


def integer_list(val):
    return [int(x) for x in val.split(',')]


def boolean_list(val):
    return [boolean(x) for x in val.split(',')]


def float_list(val):
    return [float(x) for x in val.split(',')]


def matrix_2d(val):
    return np.matrix(val).tolist()


parse_dict = {
    # net parameters
    'batch_norm': boolean_list,
    'full': integer_list,
    'latent_dim': int,
    'nfilter': integer_list,
    'non_lin': string_no_newline,
    'shape': matrix_2d,
    'stride': integer_list,
    'summary': boolean,
    # optimizer parameters
    'batch_size': int,
    'beta1': float,
    'beta2': float,
    'disc_learning_rate': float,
    'epsilon': float,
    'epoch': int,
    'gamma_gp': int,
    'gen_learning_rate': float,
    'learning_rate': float,
    'optimizer': string_no_newline,
    'disc_optimizer': string_no_newline,
    'gen_optimizer': string_no_newline,
    # general parameters
    'image_size': integer_list,
    'name': string_no_newline,
    'num_classes': int,
    'num_gaussians': int,
    'num_samples_per_class': int,
    'model_idx': int,
    'save_dir': string_no_newline,
    'save_every': int,
    'sum_every': int,
    'summary_dir': string_no_newline,
    'viz_every': int,
    'weight_l2': float,
    # reader parameters
    'discriminator_params_path': string_no_newline,
    'generator_params_path': string_no_newline,
    'optimizer_params_path': string_no_newline,
    'cosmology_params_path': string_no_newline,
    'params_path': string_no_newline,
    # cosmology parameters
    'clip_max_real': boolean,
    'k': int,
    'log_clip': float,
    'Npsd': int,
    'sigma_smooth': float,
}


def read_dict(path):
    param_dict = {}
    with open(path, "r") as in_file:
        for line in in_file:
            args = line.split('=')
            param_dict[args[0]] = parse_dict[args[0]](args[1])
    return param_dict


def read_gan_dict(path):
    paths = read_dict(path)
    gen_params = read_dict(paths['generator_params_path'])
    disc_params = read_dict(paths['discriminator_params_path'])
    opt_params = read_dict(paths['optimizer_params_path'])
    cosmo_params = read_dict(paths['cosmology_params_path'])
    params = read_dict(paths['params_path'])

    params['generator'] = gen_params
    params['discriminator'] = disc_params
    params['optimization'] = opt_params
    params['cosmology'] = cosmo_params
    return params