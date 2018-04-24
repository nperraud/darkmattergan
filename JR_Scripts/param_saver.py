import sys
sys.path.insert(0, '../')

import utils, sys, os
from JR_Scripts import dict_reader, time_toy_generator


def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year) + '_' + str(d.month)+ '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute)


def main():
    # Load parameters

    params = dict()
    params['image_size'] = [128, 128]
    params['sum_every'] = 500
    params['viz_every'] = 500
    params['save_every'] = 8000
    params['name'] = 'TWGAN_Toy_4C'
    params['model_idx'] = 2
    params['num_classes'] = 4
    params['classes'] = [0,3,6,9]
    #params['class_weights'] = [0.91, 1]
    params['num_samples_per_class'] = 5000
    params['summary_dir'] = '/scratch/snx3000/rosenthj/results/summaries_A7'
    params['save_dir'] = '/scratch/snx3000/rosenthj/results/models_A7'
    params['num_gaussians'] = 42

    time_str = current_time_str()
    if 'save_dir' in params:
        params['summary_dir'] = params['summary_dir'] + '/' + params['name'] + '_' + time_str + '_summary/'
        params['save_dir'] = params['save_dir'] + '/' + params['name'] + '_' + time_str + '_checkpoints/'
    else:
        params['summary_dir'] = 'tboard/' + params['name'] + '_' + time_str + 'summary/'
        params['save_dir'] = 'checkp/' + params['name'] + '_' + time_str + 'checkpoints/'

    params['cosmology'] = dict()
    params['cosmology']['clip_max_real'] = False
    params['cosmology']['k'] = 100
    params['cosmology']['log_clip'] = 0.1
    params['cosmology']['Npsd'] = 500
    params['cosmology']['sigma_smooth'] = 1

    params['discriminator'] = dict()
    params['discriminator']['stride'] = [2, 2, 2, 2, 2, 1]
    params['discriminator']['nfilter'] = [16, 32, 64, 128, 256, 64]
    params['discriminator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [3, 3]]
    params['discriminator']['batch_norm'] = [False, False, False, False, False, False]
    params['discriminator']['full'] = [32]
    params['discriminator']['summary'] = True

    params['generator'] = dict()
    params['generator']['stride'] = [2, 2, 2, 2, 2, 1, 1]
    params['generator']['latent_dim'] = 25
    params['generator']['nfilter'] = [16, 64, 256, 128, 64, 32, 1]
    params['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [3, 3], [3, 3]]
    params['generator']['batch_norm'] = [False, False, False, False, False, False]
    params['generator']['full'] = [256]
    params['generator']['summary'] = True
    params['generator']['non_lin'] = None

    params['optimization'] = dict()
    params['optimization']['gamma_gp'] = 50
    params['optimization']['batch_size'] = 8
    params['optimization']['weight_l2'] = 0.1
    params['optimization']['disc_optimizer'] = 'rmsprop'
    params['optimization']['gen_optimizer'] = 'rmsprop'
    params['optimization']['disc_learning_rate'] = 3e-5
    params['optimization']['gen_learning_rate'] = 3e-5
    params['optimization']['beta1'] = 0.9
    params['optimization']['beta2'] = 0.99
    params['optimization']['epsilon'] = 1e-8
    params['optimization']['epoch'] = 150

    print("All params")
    print(params)
    print("\nDiscriminator Params")
    print(params['discriminator'])
    print("\nGenerator Params")
    print(params['generator'])
    print("\nOptimization Params")
    print(params['optimization'])
    print("\nCosmo Params")
    print(params['cosmology'])
    print()

    thesis_dir = '/home/jonathan/Documents/Master_Thesis/'
    utils.save_dict_pickle(thesis_dir + 'NNParamsDaint/params.pkl', params)
    utils.save_dict_for_humans(thesis_dir + 'NNParamsDaint/params.txt', params)

    return 0


main()
