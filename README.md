# CosmoGAN

Using Generative Adversarial Networks to generate replications of N-Body simulation images. This repository contains the code to solve different problems including (2D, 3D, 2D-time, generation part by parts)

## Installation


### Required packages


*We hightly recommend to work in a virtual environnement.*

You can simply install those packages with the following command:
```
pip install -r requirements.txt
```
or if you have no gpu:
```
pip install -r requirements_nogpu.txt
```

For some operations, you may require `pynbody` as it was used to preprocess the simulation. If so, you need to install it separately.

## Dataset

The dataset consists of 30 N-body simulations at a scale of 500 MPch and 10 simulations at a scale of 100 Mpch. The dataset is availlable on Zenodo at: URL

To download the dataset, you can simply execute the code:
```
python download_nbody.py
```

## Processing the simulations

TBD

## Example
See the demo notebook for an example

https://github.com/nperraud/CodeGAN/blob/master/WGAN%20demo.ipynb

## Oganisation of the code

The code is composed of a package named *gantools*. It is composed of the following submodules:
* gansystem: implement the basic training and generating system for a gan
* modely: contains the different network architecture
* data: data module
* blocks: basic tensorflow units
* utils: useful functions
* metrics: computation of the different error functions
* plot: helper for the different plots

## Paths

### CSCS
* CSCS Project path `/store/sdsc/sd01/cosmology/`
* CSCS Raw data path `/store/sdsc/sd01/cosmology/data/nbody_raw_boxes/`
* CSCS Preprocessed data path `/store/sdsc/sd01/comosology/data/pre_processed_data/`

### Plempy
* Old path on plempy (Original PICOLA DATA) `/home/ipa/refreg/temp/Janis/AndresBoxes`
* Old path on plempy (Procesed data) `/home/ipa/refreg/temp/andresro/data/dat`

## Contributors

Perraudin Nathanaël, Rosenthal Jonathan, Srivastava Ankit

Some of the code is based on the work of Andres Rodriguez Escallon.

See the following repository https://github.com/dalab/msc_andres

## License

TBD

## TODO

* Put the data on zenodo
* Experiment paper
