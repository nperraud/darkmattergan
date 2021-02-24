# Overview of .py Files:

##### build_model_test.py
Used to test models and features locally. Can safely be deleted, however is useful to make sure everything is in working order.

##### dict_reader.py
Helper file for the legacy dict reader. Deleting it should be ok. Some of the oldest models may not have a pickl param dict.

##### notebook_helper.py
Utility source which is require by notebooks to run.

##### Stats_Generator.py
Used to remake statistics for trained models.

##### TGAN(128)-Generic.py
These are the files used to run models.

 - Depending on the model settings, the model will have a different name. Take a look at the get_model_name function definition to understand model names and get an understanding of what parameters they were run with.
 - The time_encoding string needs to be set to one of the three values following its definition. This defines how the latent variable is encoded with a continuous variable.
 - When calling the file, its arguments are the time steps with which to train the model.
 - Usage: python3 TGAN-Generic.py 0 2 4 6

##### time_toy_generator.py
Used to create the toy dataset.

##### Train_Encoder.py
Used to add and train an encoder with a model which has been trained without one.

##### TWGAN-Resumer.py
Used to resume training of a model. Usage: python3 TWGAN-Resumer.py MyModel

##### TWGAN-Toy64.py
Used to train a model on the toy problem dataset.

# Overview of .ipynb Files

##### GifMaker.ipynb
Used to generate gifs over time for a CCGAN

##### remote_model_viewer-(ce,sh,sf).ipynb
View statistics of a model trained with a specific latent encoding scheme. The three notebooks are identical except the channel encoding version has a bug that results in it selecting the same images to compare to every time.

##### remote_sts_model_comparer_CCGAN_vs_specialized-complexV3.ipynb
Compared models trained on a single redshift with a CCGAN with channel encoding.

##### remote_sts_model_comparer_trained_redshifts-complexV2.ipynb
View statistics of models with different latent encodings on time steps they were trained on.

##### remote_sts_model_comparer_untrained_redshifts-complexV2.ipynb
View statistics of models with different latent encodings on time steps they were not trained on.
