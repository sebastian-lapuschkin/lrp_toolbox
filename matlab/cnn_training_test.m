% @author: Sebastian Lapuschkin
% @maintainer: Sebastian Lapuschkin
% @contact: sebastian.lapuschkin@hhi.fraunhofer.de
% @date: 10.11.2015
% @version: 1.2+
% @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
% @license : BSD-2-Clause
%
%The purpose of this module is to demonstrate the process of obtaining pixel-wise explanations for given data points at hand of the MNIST hand written digit data set.
%
%The module first loads a pre-trained neural network model and the MNIST test set with labels and transforms the data such that each pixel value is within the range of [-1 1].
%The data is then randomly permuted and for the first 10 samples due to the permuted order, a prediction is computed by the network, which is then as a next step explained
%by attributing relevance values to each of the input pixels.
%
%finally, the resulting heatmap is rendered as an image and (over)written out to disk and displayed.%

clear
import model_io.*
import data_io.*
import render.*

%load neural network, as well as the MNIST test data and some labels
Xtrain = data_io.read('../data/MNIST/train_images.mat');
Xtest = data_io.read('../data/MNIST/test_images.mat');

Ytrain = data_io.read('../data/MNIST/train_labels.mat');
Ytest = data_io.read('../data/MNIST/test_labels.mat');

%transfer pixel values from [0 255] to [-1 1] to satisfy the expected input
%and training paradigm of the model
Xtrain = Xtrain / 127.5 - 1;
Xtest = Xtest / 127.5 - 1;

%reshape the vector representations in X to match the requirements of the
%CNN input
Xtrain = permute(reshape(Xtrain,[size(Xtrain,1), 28, 28, 1]),[1 3 2 4]);
Xtrain = padarray(Xtrain,[0 2 2],-1,'both');

Xtest = permute(reshape(Xtest,[size(Xtest,1), 28, 28, 1]),[1 3 2 4]);
Xtest = padarray(Xtest,[0 2 2],-1,'both');

%transform numeric class labels to vector indicators for uniformity.
%we assume all class labels to be present within the label set.
I = Ytrain+1;
Ytrain = zeros(size(Xtrain,1),numel(unique(Ytrain)));
Ytrain(sub2ind(size(Ytrain),1:size(Ytrain,1),I')) = 1;

I = Ytest+1;
Ytest = zeros(size(Xtest,1),numel(unique(Ytest)));
Ytest(sub2ind(size(Ytest),1:size(Ytest,1),I')) = 1;

%model network according to LeNet-5 architecture
lenet = modules.Sequential({
                            modules.Convolution([5 5 1 10],[1 1]),
                            modules.Rect(),
                            modules.SumPool([2 2],[2 2])
                            modules.Convolution([5 5 10 25],[1 1]),
                            modules.Rect(),
                            modules.SumPool([2 2],[2 2]),
                            modules.Convolution([4 4 25 100],[1 1]),
                            modules.Rect(),
                            modules.SumPool([2 2],[2 2]),
                            modules.Convolution([1 1 100 10],[1 1]),
                            modules.Flatten()
                            });

%train the net
lenet.train(Xtrain,Ytrain,Xtest,Ytest,25,10^6,0.0001);

%save the net
model_io.write(lenet,'../LeNet-5m.txt')



%a slight variation to test max pooling layers. this model should train faster.
maxnet = modules.Sequential({
                            modules.Convolution([5 5 1 10],[1 1]),
                            modules.Rect(),
                            modules.MaxPool([2 2],[2 2])
                            modules.Convolution([5 5 10 25],[1 1]),
                            modules.Rect(),
                            modules.MaxPool([2 2],[2 2]),
                            modules.Convolution([4 4 25 100],[1 1]),
                            modules.Rect(),
                            modules.MaxPool([2 2],[2 2]),
                            modules.Convolution([1 1 100 10],[1 1]),
                            modules.Flatten(),
                            modules.SoftMax()
                            });

%train the net
maxnet.train(Xtrain,Ytrain,Xtest,Ytest,25,10^6,0.001);

%save the net
model_io.write(maxnet,'../LeNet-5m-maxpooling.txt')

