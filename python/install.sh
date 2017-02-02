#!/bin/bash
#
# This is an installation script for installing all required python
# packages required for running the standalone lrp implementation /
# demo which is part of the LRP Toolbox v1.0 on the Ubuntu 14.04 LTS
# 64 bit distribution. This script should be considered a convenient
# solution for setting up the LRP toolbox for the afore mentioned OS
# and might have success installing all needed requirements on different
# releases of Ubuntu/Linux. There is, however, no guarantee for that.
# This is especially true for other Linux derivates or OSs in general.
#
# The installation of required python packages will by performed via 
# the reliable and convenient apt-get.
#
# Before executing this script, please read and modify the following commands 
# carefully in order to prevent unwanted changes to your system.
#
# This installation requires administrator level privileges.

# INSTALL REQUIRED PYTHON PACKAGES
apt-get install python-numpy python-numpy-dbg python-scipy python-matplotlib python-skimage
#pip install skimage #python skimage 0.12.3+ for newer ubuntu/windows compatibility



# DOWNLOAD MODELS AND DATA FOR THE DEMO APPLICATION
# go to toolbox root
cd .. 

# download and extract the MNIST hand written data set
if ! [[ -f data/MNIST/test_images.npy && -f data/MNIST/test_labels.npy ]]
then
  fname=data_mnist_npy.zip
  wget -nc http://heatmapping.org/files/lrp_toolbox/data/$fname
  unzip $fname
  rm $fname
fi

# download and extract the model required for successfully run the demo
if ! [ -f models/MNIST/long-rect.nn ]
then
  fname=models_mnist_nn.zip
  wget -nc http://heatmapping.org/files/lrp_toolbox/models/$fname
  unzip $fname
  rm $fname
fi

# go back to ./python
cd python



# RUN DEMO CODE
python lrp_demo.py

