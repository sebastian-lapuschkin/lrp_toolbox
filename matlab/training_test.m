% @author: Sebastian Lapuschkin
% @maintainer: Sebastian Lapuschkin
% @contact: sebastian.lapuschkin@hhi.fraunhofer.de
% @date: 30.09.2015
% @version: 1.0
% @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
% @license : BSD-2-Clause


import modules.*
import model_io.*

D = 2 ; N = 200000 ;

%this is the XOR problem.
X = rand(N,D); % we want [NxD] data
X = (X > 0.5)*1.0;
Y = X(:,1) == X(:,2); 
Y = [Y ~Y]; % and [NxC] labels

X = X + randn(N,D)*0.1; % add some noise to the data

%build a network
nn = modules.Sequential({modules.Linear(2,3), modules.Tanh(), modules.Linear(3,3),modules.Tanh(),modules.Linear(3,2),modules.SoftMax()});

%train the network.
nn.train(X,Y,[],[],5);

%save the network
model_io.write(nn, '../xor_net_small_1000m.txt')
