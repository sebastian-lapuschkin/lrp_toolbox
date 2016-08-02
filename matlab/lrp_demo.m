% @author: Sebastian Lapuschkin
% @maintainer: Sebastian Lapuschkin
% @contact: sebastian.lapuschkin@hhi.fraunhofer.de
% @date: 14.08.2015
% @version: 1.0
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
nn = model_io.read('../models/MNIST/long-rect.mat');
X = data_io.read('../data/MNIST/test_images.mat');
Y = data_io.read('../data/MNIST/test_labels.mat');

%transfer pixel values from [0 255] to [-1 1] to satisfy the expected input
%and training paradigm of the model
X = X / 127.5 - 1;

%transform numeric class labels to vector indicators for uniformity.
%we assume all class labels to be present within the label set.
I = Y+1;
Y = zeros(size(X,1),numel(unique(Y)));
Y(sub2ind(size(Y),1:size(Y,1),I')) = 1;

%permute data order for demonstration. or not. your choice.
I = 1:size(X,1);
%I = randperm(size(X,1));

%predict and perform LRP for the first 10 samples
Ri = zeros(0,784);
for i = I(1:10)
    x = X(i,:);
    
    %forward pass and prediction
    ypred = nn.forward(x);
    [~,yt] = max(Y(i,:));
    [~,yp] = max(ypred);
    
    fprintf('True Class:      %d\n', yt-1);
    fprintf('Predicted Class: %d\n\n', yp-1);
    
    %compute first layer relevance according to prediction
    R = nn.lrp(ypred);                 %as Eq(56) from DOI: 10.1371/journal.pone.0130140
    %R = nn.lrp(ypred,'epsilon',100);   %as Eq(58) from DOI: 10.1371/journal.pone.0130140
    %R = nn.lrp(ypred,'alphabeta',2);    %as Eq(60) from DOI: 10.1371/journal.pone.0130140
    

    %R = nn.lrp(Y(i,:)); %compute first layer relevance according to the true class label
    
    %yselect = 4;
    %yselect = (1:size(Y,2) == yselect)*1.;
    %R = nn.lrp(yselect); %compute first layer relevance for an arbitrarily selected class


    %render input and heatmap as rgb images
    digit = render.digit_to_rgb(x,3);
    hm = render.hm_to_rgb(R,x,3,[],2);
    img = render.save_image({digit,hm},'../heatmap.png');
    imshow(img); axis off ; drawnow;
    input('hit enter!')
end

%note that modules.Sequential allows for batch processing of inputs
%ypred = nn.forward(X(1:10,:));
%R = nn.lrp(ypred);
%data_io.write(R,'../Rbatch.mat')

