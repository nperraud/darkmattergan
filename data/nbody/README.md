# 3Dcosmo: a benchmark dataset for large 3-dimensional generative models (and 2-dimensional as well)

This is the N-body simulations 3D images dataset used in the paper *Cosmological N-body simulations: a challenge for scalable generative models*,
Nathanaël Perraudin, Ankit Srivastava, Aurelien Lucchi, Tomasz Kacprzak, Thomas Hofmann, Alexandre Refregier

The dataset does not contain the Nbody simulations as they have a very large size. Instead, we sliced the space into 256 x 256 x 256 cubical areas and counted the number of particules in each area. The result are 3D histograms, where the number of particles is a proxy for matter density.

## 3DCosmo benchmark
This dataset can be used to evaluate 2D and 3D generative model. It is particularly suitable for large scale 3D images. Please check https://github.com/nperraud/3DcosmoGAN for more information.

Please consider citing our paper if you use it.

```
@inproceedings{perraudin2019cosmological,
  title = {Cosmological N-body simulations: a challenge for scalable generative models},
  author = {Nathana\"el, Perraudin and Ankit, Srivastava and Kacprzak, Tomasz and Lucchi, Aurelien and Hofmann, Thomas and R{\'e}fr{\'e}gier, Alexandre},
  year = {2019},
  archivePrefix = {arXiv},
  eprint = {....},
  url = {https://arxiv.org/abs/...},
}
```

While this data is associated to the paper *Cosmological N-body simulations: a challenge for scalable generative models*, note that a the same Nbody simulation were used in the paper *Fast Cosmic Web Simulations with Generative Adversarial Networks* (https://arxiv.org/abs/1801.09070v1), but with a different way of building the histogram. You may want to cite this work as well.

```
@article{rodriguez2018fast,
  title={Fast cosmic web simulations with generative adversarial networks},
  author={Rodr{\'\i}guez, Andres C and Kacprzak, Tomasz and Lucchi, Aurelien and Amara, Adam and Sgier, Rapha{\"e}l and Fluri, Janis and Hofmann, Thomas and R{\'e}fr{\'e}gier, Alexandre},
  journal={Computational Astrophysics and Cosmology},
  volume={5},
  number={1},
  pages={4},
  year={2018},
  publisher={Springer}
}
```

N-body simulation evolves a cosmological matter distribution over time, starting from soon after the big bang.
It represents matter density distribution as a finite set of massive particles, typically order of trillions.
The positions of these particles are modified due to gravitational forces and expansion of the cosmological volume due to cosmic acceleration.
N-body simulations use periodic boundary condition, where particles leaving the volume on one face enter it back from the opposite side.

## Short description of the data generation:

We created N-body simulations of cosmic structures in boxes of size 100 Mpc and 500 Mpc with 512^3 and 1,024^3 particles respectively.
We used L-PICOLA [21] to create 10 and 30 independent simulation boxes for both box sizes.
The cosmological model used was ΛCDM (Cold Dark Matter) with Hubble constant H0 = 500, h = 350 km s−1 Mpc−1,
dark energy density Omega_Lambda = 0.72 and matter density Omega_m = 0.28.
We used the particle distribution at redshift z = 0.

## The configuration used for L-Picola is as follows:

$ cat run_parameters.dat

% =============================== %
% This is the run parameters file %
% =============================== %

% Simulation outputs
% ==================
OutputDir                   /cluster/scratch/jafluri/AndresBoxes2/Box_350Mpch_0/                     % Directory for output.
FileBase                    out                             % Base-filename of output files (appropriate additions are appended on at runtime)
OutputRedshiftFile          /cluster/scratch/jafluri/AndresBoxes2/Box_350Mpch_0/output_redshift.dat           % The file containing the redshifts that we want snapshots for
NumFilesWrittenInParallel   16                                   % limits the number of files that are written in parallel when outputting.

% Simulation Specifications
% =========================
UseCOLA          1           % Whether or not to use the COLA method (1=true, 0=false).
Buffer           3           % The amount of extra memory to reserve for particles moving between tasks during runtime.
Nmesh            2048         % This is the size of the FFT grid used to compute the displacement field and gravitational forces.
Nsample          1024        % This sets the total number of particles in the simulation, such that Ntot = Nsample^3.
Box              350      % The Periodic box size of simulation.
Init_Redshift    9.0         % The redshift to begin timestepping from (redshift = 9 works well for COLA)
Seed             1020        % Seed for IC-generator
SphereMode       0           % If "1" only modes with |k| < k_Nyquist are used to generate initial conditions (i.e. a sphere in k-space),
                             % otherwise modes with |k_x|,|k_y|,|k_z| < k_Nyquist are used (i.e. a cube in k-space).

WhichSpectrum    2           % "0" - Use transfer function, not power spectrum
                             % "1" - Use a tabulated power spectrum in the file 'FileWithInputSpectrum'
                             % otherwise, Eisenstein and Hu (1998) parametrization is used
                             % Non-Gaussian case requires "0" and that we use the transfer function

WhichTransfer    0           % "0" - Use power spectrum, not transfer function
                             % "1" - Use a tabulated transfer function in the file 'FileWithInputTransfer'
                             % otherwise, Eisenstein and Hu (1998) parameterization used
                             % For Non-Gaussian models this is required (rather than the power spectrum)

FileWithInputSpectrum  files/input_spectrum.dat    % filename of tabulated input spectrum (if used)
                                                   % expecting k and Pk

FileWithInputTransfer  files/input_transfer.dat    % filename of tabulated transfer function (if used)
                                                   % expecting k and T (unnormalized)

% Cosmological Parameters
% =======================
Omega            0.276        % Total matter density (CDM + Baryons at z=0).
OmegaBaryon      0.045        % Baryon density (at z=0).
OmegaLambda      0.724        % Dark Energy density (at z=0)
HubbleParam      0.7         % Hubble parameter, 'little h' (only used for power spectrum parameterization).
Sigma8           0.811        % Power spectrum normalization (power spectrum may already be normalized correctly).
PrimordialIndex  0.961        % Used to tilt the power spectrum for non-tabulated power spectra (if != 1.0 and nongaussian, generic flag required)

% Timestepping Options
% ====================
StepDist         0           % The timestep spacing (0 for linear in a, 1 for logarithmic in a)
DeltaA           0           % The type of timestepping: "0" - Use modified COLA timestepping for Kick and Drift. Please choose a value for nLPT.
                             % The type of timestepping: "1" - Use modified COLA timestepping for Kick and standard Quinn timestepping for Drift. Please choose a value for nLPT.
                             % The type of timestepping: "2" - Use standard Quinn timestepping for Kick and Drift
                             % The type of timestepping: "3" - Use non-integral timestepping for Kick and Drift
nLPT             -2.5        % The value of nLPT to use for modified COLA timestepping


% Units
% =====
UnitLength_in_cm                3.085678e24       % defines length unit of output (in cm/h)
UnitMass_in_g                   1.989e43          % defines mass unit of output (in g/h)
UnitVelocity_in_cm_per_s        1e5               % defines velocity unit of output (in cm/sec)
InputSpectrum_UnitLength_in_cm  3.085678e24       % defines length unit of tabulated input spectrum in cm/h.
                                                  % Note: This can be chosen different from UnitLength_in_cm