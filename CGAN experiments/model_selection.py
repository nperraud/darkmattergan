import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.insert(0, '../')
from gantools import data
from gantools import utils
from gantools.model import ConditionalParamWGAN
from gantools.gansystem import GANsystem
from gantools import evaluation
from gantools import plot

# Default parameters of function select models
def non_lin(x):
    return tf.nn.relu(x)

def title_func(params):
    return "$\Omega_M$: " + str(params[0])[0:7] + ", $\sigma_8$: " + str(params[1])[0:7]

ylims = [[(1e-5, 1e-1), (0, 1)], (1e-3, 3e3), (1e-3, 1e5)]
fractional_difference = [True, False, False]
bin_k = 50
box_l=(5*np.pi/180)
cut = [100, 6000]
lim = (0, 0.4)


# Given the model specifications and a list of candidate checkpoints, returns the best model
# It also produces plots containing the results under the folder Outputs/ in the model summary folder
def select_model(global_path, name, checkpoints, dataset_train, dataset_test, N=2000, ns=256, non_lin=non_lin, title_func=title_func, ylims=ylims, fractional_difference=fractional_difference, bin_k=bin_k, box_l=box_l, cut=cut, lim=lim, save=True, lenstools=True):

    # Create result folders if they don't exist
    folder_out = os.path.join(global_path, name + '_summary/', 'Outputs/')
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    if not os.path.exists(os.path.join(global_path, name + '_checkpoints/', 'Generated/')):
        os.makedirs(os.path.join(global_path, name + '_checkpoints/', 'Generated/'))

    # Load parameters
    with open(os.path.join(global_path, name + '_checkpoints/', 'params.pkl'), 'rb') as f:
    	params = pickle.load(f)

    resume, params = utils.test_resume(True, params)

    # If a model is reloaded and some parameters have to be changed, then it should be done here.
    # For example, setting the number of epoch to 5 would be:
    params['summary_dir'] = os.path.join(global_path, name + '_summary/')
    params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')

    wgan = GANsystem(ConditionalParamWGAN, params)

    # Produce results
    scores = []
    correlations = []
    for checkpoint in checkpoints:

        # Accuracy on training set
        dataset = data.load.load_params_dataset(filename=dataset_train, batch=N, sorted=True, shape=[ns, ns])
        params_train = dataset.get_different_params()

        # Produce and save generated images
        file_name = os.path.join(global_path, name +'_checkpoints/', 'Generated/') + 'train_gen_' + str(checkpoint) + '.h5'
        if not os.path.isfile(file_name):
            first = True
            for p in params_train:
                gen_params = np.tile(p, [N, 1])
                gen_imgs = evaluation.generate_samples_params(wgan, p, nsamples=N, checkpoint=checkpoint)
                utils.append_h5(file_name, gen_imgs, gen_params, overwrite=first)
                first = False

        # Define getter functions for every parameter set
        # Note this is needed to save memory, as in this way every subset is loaded only when needed
        real_imgs = []
        fake_imgs = []
        dataset_generated = data.load.load_params_dataset(filename=file_name, batch=N, sorted=True, shape=[ns, ns])
        for p in params_train:
            real_imgs.append(lambda p1=p: dataset.get_data_for_params(p1, N=N)[0])
            fake_imgs.append(lambda p1=p: dataset_generated.get_data_for_params(p1, N=N)[0])

        # Compute plot scores
        plt.figure()
        fig, score_train = evaluation.compute_plots_for_params(params_train, real_imgs, fake_imgs, param_str=title_func, log=False, lim=lim, ylims=ylims, confidence='std', fractional_difference=fractional_difference, bin_k=bin_k, box_l=box_l, cut=cut, lenstools=lenstools)
        fig.savefig(folder_out + 'results_train_' + str(checkpoint) + '.png')

        # Compute plot correlations
        plt.figure()
        c_score_train = evaluation.compute_plot_correlations(real_imgs, fake_imgs, params_train, box_l=box_l, bin_k=bin_k, cut=cut, param_str=title_func, lenstools=lenstools)
        plt.savefig(folder_out + 'corr_train_' + str(checkpoint) + '.png')

        if not save:
            os.remove(filename)

        # Accuracy on test set
        dataset = data.load.load_params_dataset(filename=dataset_test, batch=N, sorted=True, shape=[ns, ns])
        params = dataset.get_different_params()

        # Produce and save generated images
        file_name = os.path.join(global_path, name +'_checkpoints/', 'Generated/') + 'test_gen_' + str(checkpoint) + '.h5'
        if not os.path.isfile(file_name):
            first = True
            for p in params:
                gen_params = np.tile(p, [N, 1])
                gen_imgs = evaluation.generate_samples_params(wgan, p, nsamples=N, checkpoint=checkpoint)
                utils.append_h5(file_name, gen_imgs, gen_params, overwrite=first)
                first = False

        # Define getter functions for every parameter set
        # Note this is needed to save memory, as in this way every subset is loaded only when needed
        real_imgs = []
        fake_imgs = []
        dataset_generated = data.load.load_params_dataset(filename=file_name, batch=N, sorted=True, shape=[ns, ns])
        for p in params:
            real_imgs.append(lambda p1=p: dataset.get_data_for_params(p1, N=N)[0])
            fake_imgs.append(lambda p1=p: dataset_generated.get_data_for_params(p1, N=N)[0])

        plt.figure()
        fig, score = evaluation.compute_plots_for_params(params, real_imgs, fake_imgs, param_str=title_func, ylims=ylims, log=False, confidence='std', lim=lim, fractional_difference=fractional_difference, bin_k=bin_k, box_l=box_l, cut=cut, lenstools=lenstools)
        fig.savefig(folder_out + 'results_test_' + str(checkpoint) + '.png')

        # Compute plot correlations
        plt.figure()
        c_score = evaluation.compute_plot_correlations(real_imgs, fake_imgs, params, box_l=box_l, bin_k=bin_k, cut=cut, param_str=title_func, lenstools=lenstools)
        plt.savefig(folder_out + 'corr_test_' + str(checkpoint) + '.png')

        if not save:
            os.remove(filename)

        # Save scores
        scores.append([score_train, score])
        correlations.append([c_score_train, c_score])

        # Accuracy heat map
        plt.figure()
        plot.plot_heatmap(score[:, 0, 4], params, score_train[:, 0, 4], params_train)
        plt.savefig(folder_out + 'fractional_difference_' + str(checkpoint) + '.png')

        plt.figure()
        plot.plot_heatmap(score[:, 0, 4], params, score_train[:, 0, 4], params_train, thresholds=[0.025, 0.05, 0.10, 0.15])
        plt.savefig(folder_out + 'fractional_difference_threshold_' + str(checkpoint) + '.png')

        plt.figure()
        plot.plot_heatmap(c_score.flatten(), params, c_score_train.flatten(), params_train, vmin=1, vmax=15 if lenstools else 25)
        plt.savefig(folder_out + 'correlations_' + str(checkpoint) + '.png')


    scores = np.array(scores)
    correlations = np.array(correlations)
    np.save(folder_out + 'scores.npy', scores)
    np.save(folder_out + 'corr_scores.npy', correlations)

    if not save:
        os.rmdir(os.path.join(global_path, name +'_checkpoints/', 'Generated/'))

    # Print model fractional difference accuracy on test set
    # Find best model
    best_score = np.inf
    best_model = None
    s_out = 'CHECKPOINT\tMEAN\tSTD\n'
    for i in range(len(scores)):
        curr_score = np.mean(scores[i][1][:, 0, 4])
        curr_std = np.std(scores[i][1][:, 0, 4])
        s_out = s_out + str(checkpoints[i]) + '\t' + str(curr_score) + '\t' + str(curr_std) + '\n'
        if curr_score < best_score:
            best_score = curr_score
            best_model = checkpoints[i]

    s_out = s_out + 'Suggested best model: ' + str(best_model)
    with open(folder_out + 'average_fd_out.txt', "w") as f:
        f.write(s_out)
    print("Suggested best model farc diff:", best_model)

    # Print model correlation on test set
    # Find best model
    best_score = np.inf
    best_model = None
    s_out = 'CHECKPOINT\tMEAN\tSTD\n'
    for i in range(len(correlations)):
        curr_score = np.mean(correlations[i][1])
        curr_std = np.std(correlations[i][1])
        s_out = s_out + str(checkpoints[i]) + '\t' + str(curr_score) + '\t' + str(curr_std) + '\n'
        if curr_score < best_score:
            best_score = curr_score
            best_model = checkpoints[i]

    s_out = s_out + 'Suggested best model: ' + str(best_model)
    with open(folder_out + 'correlations_out.txt', "w") as f:
        f.write(s_out)
    print("Suggested best model correlation:", best_model)

    return best_model