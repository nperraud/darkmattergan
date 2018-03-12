# CosmoGAN

Using Generative Adversarial Networks to generate replications of N-Body simulation images. This repository contains the code to solve different problems including (2D, 3D, 2D-time, generation part by parts)

## Required packages

1. pynbody (only for processing the simulations)
2. Pillow (PIL)
3. Tensorflow
4. h5py

## Dataset

The dataset consists of 10 N-body simulations.

TBD

## Processing the simulations

TBD

## Example
See the demo notebook for an example

https://github.com/nperraud/CodeGAN/blob/master/WGAN%20demo.ipynb

## Oganisation of the code

* gan.py : implement the basic GAN and CosmoGAN class
* modely.py : contains the different network architecture
* data.py : to load the different datasets
* blocks.py : basic tensorflow units
* utils.py : useful function
* default.py : default parameters
* metrics.py : computation of the different error functions
* plot.py : helper for the different plots

## Paths
* CSCS Project path `/store/sdsc/sd00/comosology/`
* Old path on plempy (Original PICOLA DATA) `/home/ipa/refreg/temp/Janis/AndresBoxes`
* Old path on plempy (Procesed data) `/home/ipa/refreg/temp/andresro/data/dat`

## Contributors

Perraudin Nathanaël, Rosenthal Jonathan, Srivastava Ankit

Some of the code is based on the work of Andres Rodriguez Escallon.

See the following repository https://github.com/dalab/msc_andres

## License

TBD

## TODO

* Re-organise the file into a small package
* Make the 3D dataset
* Remake the 2D dataset
* Merge the time datasets
* Include all dataset loading routines into the module `data.py`
