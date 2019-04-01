import tensorflow as tf
from model_selection import select_model

##### ADJUST THESE PARAMETERS #####

ns = 128 # Resolution of the image
try_resume = True # Try to resume previous simulation

def non_lin(x):
    return tf.nn.relu(x)

# Dataset names
dataset_train = 'kids_train.h5'
dataset_test = 'kids_test.h5'

# Model folder
time_str = '2D'

global_path = '/scratch/snx3000/smarcon/saved_results/'

name = 'KidsConditional{}'.format(ns) + '_smart_' + time_str

N = 1000 # Number of samples

# Checkpoints to be evaluated
checkpoints = [349163, 392217, 435271, 478325]

# Adjust plots
def title_func(params):
    return "$\Omega_M$: " + str(params[0])[0:7] + ", $\sigma_8$: " + str(params[1])[0:7]

lenstools = False
if lenstools:
    ylims = [[(1e-7, 1e-3), (0, 0.5)], [(1e-2, 1e3), (0, 0.5)], [(1e-2, 1e5), (0, 0.5)]]
else:
    ylims = [[(1e-4, 1e-1), (0, 0.5)], [(1e-2, 1e3), (0, 0.5)], [(1e-2, 1e5), (0, 0.5)]]
fractional_difference = [True, True, True]
lim = (0, 0.4)

##########

select_model(global_path, name, checkpoints, dataset_train, dataset_test, N=N, ns=ns, non_lin=non_lin, title_func=title_func, ylims=ylims, fractional_difference=fractional_difference, lim=lim, lenstools=lenstools)
