% @author: Sebastian Lapuschkin
% @maintainer: Sebastian Lapuschkin
% @contact: sebastian.lapuschkin@hhi.fraunhofer.de
% @date: 21.09.2015
% @version: 1.0
% @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
% @license : BSD-2-Clause

% imports
clear
import model_io.*
import data_io.*
import render.*
% end of imports

nn = model_io.read('../models/MNIST/long-rect.mat'); % read model
X = data_io.read('../data/MNIST/test_images.mat'); % load MNIST test images
X = X(1,:) / 127.5 -1; % pick first data point, normalize to [-1 1]

Ypred  = nn.forward(X); % forward pass through network
R = nn.lrp(Ypred); % apply lrp to explain prediction of X

% render rgb images and save as image
digit = render.digit_to_rgb(X);
hm = render.hm_to_rgb(R,X); % render heatmap R, use X as outline
render.save_image({digit,hm},'../hm_m.png');





