#!/bin/bash

# Create and activate the conda environment
conda create --name td python=3.8 -y
source activate td  # or 'conda activate td' depending on your shell

# Install PyTorch with CUDA 11.1 support
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
  torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install OpenMIM and related packages
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose

# Clone and install mmaction2
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
cd ..

# Install additional pip packages
pip install -U albumentations --no-binary qudida,albumentations
pip install opencv-python-headless
pip install -U scikit-learn scipy matplotlib
pip install timm
