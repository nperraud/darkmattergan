This folder contains the script to train and evaluate the conditional GAN.

- launchEvaluation.sh: launch the model evaluation on CSCS. Please refer to model_selection_kids.py.
- launchModel.sh: launch the training on CSCS. Please refer to train.py.
- model_selection.py: contains the code used to perform model evalaution. 
					  Given a conditional GAN and a list of checkpoints, it computes the summary statistics of every checkpoint.
- model_selection_kids.py: python module that launches the model evaluation making use of model_selection.py.
						   Checkpoints and parameters have to be changed here.
- train.py: trains a model either from scratch or resuming a previous checkpoint. The model, the dataset and the parameters have to be specified here.
