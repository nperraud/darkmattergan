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
    'shape': matrix_2d,
    'stride': integer_list,
    'nfilter': integer_list,
    'batch_norm': boolean_list,
    'full': integer_list,
    'latent_dim': int,
    'non_lin': string_no_newline,
    'summary': boolean,
    # optimizer parameters
    'gamma_gp': int,
    'batch_size': int,
    'optimizer': string_no_newline,
    'learning_rate': float,
    'disc_learning_rate': float,
    'gen_learning_rate': float,
    'beta1': float,
    'beta2': float,
    'epsilon': float,
    'epoch': int,
    # general parameters
    'name': string_no_newline,
    'image_size': integer_list,
    'sum_every': int,
    'viz_every': int,
    'save_every': int,
    'summary_dir': string_no_newline,
    'save_dir': string_no_newline,
    'clip_max_real': boolean,
    'log_clip': float,
    'sigma_smooth': float,
    'k': int,
    # reader parameters
    'generator_params_path': string_no_newline,
    'discriminator_params_path': string_no_newline,
    'optimizer_params_path': string_no_newline,
    'params_path': string_no_newline,
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
    params = read_dict(paths['params_path'])
    params['generator'] = gen_params
    params['discriminator'] = disc_params
    params['optimization'] = opt_params
    return params