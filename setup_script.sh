#!/bin/bash

# Download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Give execute permissions
sudo chmod +x Miniconda3-latest-Linux-x86_64.sh

# Install miniconda by running bash script
./Miniconda3-latest-Linux-x86_64.sh # Install with yes, ENTER, yes

# Remove installation script
rm Miniconda3-latest-Linux-x86_64.sh

# Source bash 
source ~/.bashrc # or just open a new terminal

# Source conda
source miniconda3/etc/profile.d/conda.sh

# Cancel conda activating base environment 
conda config --set auto_activate_base false

# Source bash again
source ~/.bashrc

# Create conda environment named "book_spine_env" with python 3.6
conda create --name book_spine_env python=3.6

# Activate environment
conda activate book_spine_env

# Upgrade pip
pip install --upgrade pip

# Install tensorflow 1.6.0
pip install tensorflow==1.15.0

# Clone repo and cd
git clone https://gitlab.com/mabouseif/book_spine_detection && cd book_spine_detection/

# Install requirements
cd Mask_RCNN
pip install -r requirements.txt
python setup.py install

# Install some other required packages
sudo apt-get install -y libsm6 libxext6 libxrender1

# Run demo
cd ..
python demo.py



