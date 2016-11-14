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
%finally, the resulting heatmap is rendered as an image and (over)written out to disk and displayed.

clear
import model_io.*
import data_io.*
import render.*

%load neural network, as well as the MNIST test data and some labels
nn = model_io.read('../models/MNIST/LeNet-5.mat'); % 99.23% prediction accuracy
X = data_io.read('../data/MNIST/test_images.mat');
Y = data_io.read('../data/MNIST/test_labels.mat');

%transfer pixel values from [0 255] to [-1 1] to satisfy the expected input
%and training paradigm of the model
X = X / 127.5 - 1;

%reshape the vector representations in X to match the requirements of the
%CNN input
X = permute(reshape(X,[size(X,1), 28, 28, 1]),[1 3 2 4]);
X = padarray(X,[0 2 2],-1,'both');

%transform numeric class labels to vector indicators for uniformity.
%we assume all class labels to be present within the label set.
I = Y+1;
Y = zeros(size(X,1),numel(unique(Y)));
Y(sub2ind(size(Y),1:size(Y,1),I')) = 1;

%permute data order for demonstration. or not. your choice.
I = 1:size(X,1);
%I = randperm(size(X,1));

%predict and perform LRP for the first 10 samples
for i = I(1:10)
    x = X(i,:,:,:);

    %forward pass and prediction
    ypred = nn.forward(x);
    [~,yt] = max(Y(i,:));
    [~,yp] = max(ypred);

    fprintf('True Class:      %d\n', yt-1);
    fprintf('Predicted Class: %d\n\n', yp-1);

    %compute first layer relevance according to prediction
    %R = nn.lrp(ypred);                 %as Eq(56) from DOI: 10.1371/journal.pone.0130140
    R = nn.lrp(ypred,'epsilon',1.);   %as Eq(58) from DOI: 10.1371/journal.pone.0130140
    %R = nn.lrp(ypred,'alphabeta',2);    %as Eq(60) from DOI: 10.1371/journal.pone.0130140


    %R = nn.lrp(Y(i,:)); %compute first layer relevance according to the true class label

    %yselect = 4;
    %yselect = (1:size(Y,2) == yselect)*1.;
    %R = nn.lrp(yselect); %compute first layer relevance for an arbitrarily selected class
    
    
    % % you may also specify different decompositions for each layer, e.g. as below:
    % % first, set all layers (by calling set_lrp_parameters on the container module
    % % of class Sequential) to perform alpha-beta decomposition with alpha = 1.
    % % this causes the resulting relevance map to display excitation potential for the prediction
    % %
    % nn.set_lrp_parameters('alpha',1.)
    % %
    % %set the first layer (a convolutional layer) decomposition variant to 'w^2'. This may be especially
    % %usefill if input values are ranged [0 V], with 0 being a frequent occurrence, but one still wishes to know about
    % %the relevance feedback propagated to the pixels below the filter
    % %the result with display relevance in important areas despite zero input activation energy.
    % nn.modules{1}.set_lrp_parameters('ww'); % also try 'flat'
    % % compute the relevance map
    % R = nn.lrp(ypred); 

    
    %render input and heatmap as rgb images
    digit = render.digit_to_rgb(x,3);
    
    %turn digit back upright. the axis fliping is caused by the
    %vec2im-implementation with fortan indexing order in mind.
    digit = permute(digit,[2 1 3]);
    
    hm = render.hm_to_rgb(R,x,3,[],2);
    %same for the heatmap.
    hm = permute(hm,[2 1 3]);
    
    img = render.save_image({digit,hm},'../heatmap.png');
    imshow(img); axis off ; drawnow;
    input('hit enter!')
end

%note that modules.Sequential allows for batch processing of inputs
%ypred = nn.forward(X(1:10,:,:,:));
%R = nn.lrp(ypred);
%data_io.write(R,'../Rbatch.mat')
