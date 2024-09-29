#!/bin/bash

# Create and activate the conda environment
conda create --name sbi 
conda activate sbi

# Install dlib using conda-forge
conda install -c conda-forge dlib -y

# Install pip packages
pip install imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch
pip install -U retinaface_pytorch
pip install opencv-python-headless
pip install -U albumentations --no-binary qudida,albumentations
pip install -U scikit-learn scipy matplotlib
