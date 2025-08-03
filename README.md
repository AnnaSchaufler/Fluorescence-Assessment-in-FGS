# Fluorescence-Assessment-in-FGS
This repository contains the algorithms to train and test anomaly detection VAEs on the publically available FGS dataset (https://doi.org/10.5281/zenodo.15260349)

It contains:

1. A training sequence for a contarstive learning Variational Autoencoder for the detection of fluorescence in images of synthetic fluorescent samples. The sequence includes data preprocessing, composition of training and validation data sets, initialisation of the cl VAE and the training, as well as the training implementation. To execute the sequence run "synthFluorescenceDetection_clVAE_main.m"

2. A training sequence for a contarstive learning Variational Autoencoder for the detection of fluorescence in intraoperative neurosurgical images.
The sequence includes data preprocessing, composition of training and validation data sets, initialisation of the cl VAE and the training, as well as the training implementation. To execute the sequence run "fgsFluorescenceDetection_clVAE_main.m"

This script requires image and annotation data from the publicly available FGS imaging dataset (https://doi.org/10.5281/zenodo.15260349). Scripts, images and annotation masks need to be organized in a particular file structure. 

Store the MATLAB scripts in a main folder. Inside this folder, create a subfolder named "Images and Masks". Place all image data from the dataset directly into this "Images and Masks" folder. The corresponding annotation masks should also be placed inside this folder, organized into the following subfolders:
- masks ala
- masks liquid
- masks test
The resulting folder structure should look as follows:

- Main project folder/
- ├── synthFluorescenceDetection_clVAE_main.m
- ├── fgsFluorescenceDetection_clVAE_main.m
- ├── FD_contrastiveLossFunc.m
- ├── ...
- ├── Images and Masks/
- │   ├── masks ala/
- │   │   ├── P001_1_mask.mat
- │   │   └── ...
- │   ├── masks liquid/
- │   │   ├── mask_001ug.mat
- │   │   └── ...
- │   └── masks test/
- │   │   ├── mask_001ug_test.mat
- │   │   └── ... 

Adjustig the VAE models beta factor can be done by changing the beta_max variable in the "main" scrips "Initialize VAE net" section.
