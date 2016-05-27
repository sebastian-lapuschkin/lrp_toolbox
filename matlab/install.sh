#!/bin/bash
#
# This is an installation script for downloading the required models and
# data for the matlab standalone version of the LRP Toolbox and assumes
# a functioning installation of matlab.

# DOWNLOAD MODELS AND DATA FOR THE DEMO APPLICATION
# go to toolbox root
cd .. 

# download and extract the MNIST hand written data set
fname=data_mnist_mat.tar.gz
wget http://heatmapping.org/files/lrp_toolbox/data/$fname
tar xvf $fname
rm $fname

# download and extract the model required for successfully run the demo
fname=models_mnist_mat.tar.gz
wget http://heatmapping.org/files/lrp_toolbox/models/$fname
tar xvf $fname
rm $fname

# go back to ./matlab
cd matlab


# RUN DEMO CODE
matlab -nodesktop -nosplash -r lrp_demo

