#!/bin/bash

echo 'preparing auxiliary model data for ilsvrc12 in ../data/ilsvrc12'
cd ../data/ilsvrc12
bash get_ilsvrc_aux.sh

#downloads the model file from the caffemodel_url listet in the readme.md of the BVLC CaffeNet reference model
echo 'downloading model prototxt to ../../models/bvlc_reference_caffenet'
cd ../../models/bvlc_reference_caffenet
wget -nc http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
