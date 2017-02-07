#!/bin/bash

echo 'preparing auxiliary model data for ilsvrc12 in ../data/ilsvrc12'

here=$PWD
cd ../data/ilsvrc12
#bash get_ilsvrc_aux.sh
#since dl.caffe.berkeleyvision.org seems to be down currently, and is quite slow in general, the files for the bvlc caffe reference model are hosted on heatmapping.org
echo "Downloading..."
wget http://heatmapping.org/files/lrp_toolbox/models/bvlc_caffe_reference_model/caffe_ilsvrc12.tar.gz
echo "Unzipping..."
tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz
echo "Done."

#old: downloads the model file from the caffemodel_url listet in the readme.md of the BVLC CaffeNet reference model
#now uses the same model as available from heatmapping.org
echo 'downloading model prototxt to ../../models/bvlc_reference_caffenet'
cd ../../models/bvlc_reference_caffenet
#wget -nc http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
wget -nc http://heatmapping.org/files/lrp_toolbox/models/bvlc_caffe_reference_model/bvlc_reference_caffenet.caffemodel
cd $here
