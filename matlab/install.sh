#!/bin/bash
#
# This is an installation script for downloading the required models and
# data for the matlab standalone version of the LRP Toolbox and assumes
# a functioning installation of matlab.

# DOWNLOAD MODELS AND DATA FOR THE DEMO APPLICATION
# go to toolbox root
cd .. 

# download and extract the MNIST hand written data set if neccessary
if ! [[ -f data/MNIST/test_images.mat && -f data/MNIST/test_labels.mat ]] 
then
    fname=data_mnist_mat.zip
    wget -nc http://heatmapping.org/files/lrp_toolbox/data/$fname
    unzip $fname
    rm $fname
fi

# download and extract the model required for successfully run the demo
if  ! [ -f models/MNIST/long-rect.mat ]
then
    fname=models_mnist_mat.zip
    wget -nc http://heatmapping.org/files/lrp_toolbox/models/$fname
    unzip $fname
    rm $fname
fi

# go back to ./matlab
cd matlab


# RUN DEMO CODE
matlab -nodesktop -nosplash -r lrp_demo

