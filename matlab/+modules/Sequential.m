classdef Sequential < modules.Module
    % @author: Sebastian Bach
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Bach
    % @contact: sebastian.bach@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Bach, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %  Top level access point and incorporation of the neural network implementation.
    %  Sequential manages a sequence of computational neural network modules and passes
    %  along in- and outputs.
    
    properties
        % model parameters
        modules
    end
    
    methods
        function obj = Sequential(modules)
            %obj = Sequential(modules)
            %
            %Constructor
            %
            %Parameters
            %----------
            %modules : cell array
            %a cell array of instances of class Module
            obj.modules = modules;
        end
        
        
        function clean(obj)
            % Removes temporary variables from all network layers.
            for i = 1:length(obj.modules)
                obj.modules{i}.clean()
            end
        end
        
        function R = lrp(obj,R, lrp_var, param)
            %R = lrp(obj,R)
            %
            %Performs LRP using the network and temporary data produced by a forward call
            %
            %Parameters
            %----------
            %R : matrix
            %final layer relevance values. usually the network's prediction of some data points
            %for which the output relevance is to be computed
            %dimensionality should be equal to the previously computed predictions
            %
            %lrp_var : str
            %either 'none' or 'simple' or [] for standard Lrp ,
			%'epsilon' for an added epsilon slack in the denominator
			%'alphabeta' for weighting positive and negative contributions separately. param specifies alpha with alpha + beat = 1
            %
            %param : double
			%the respective parameter for the lrp method of choice
            %
            %Returns
            %-------
            %
            %R : matrix
            %the first layer relevances as produced by the neural net wrt to the previously forward
            %passed input data. dimensionality is equal to the previously into forward entered input data
            %
            %Note
            %----
            %
            %Requires the net to be populated with temporary variables, i.e. forward needed to be called with the input
            %for which the explanation is to be computed. calling clean in between forward and lrp invalidates the
            %temporary data
            if nargin < 4 || (exist('param','var') && isempty(param))
                param = 0;
            end
            if nargin < 3 || (exist('lrp_var','var') && isempty(lrp_var))
                lrp_var = [];
            end
            
            for i = length(obj.modules):-1:1
                R = obj.modules{i}.lrp(R,lrp_var,param);
            end
        end
        
        
        function X = forward(obj,X)
            %X = forward(obj,X)
            %
            %Realizes the forward pass of an input through the net
            %
            %Parameters
            %----------
            %X : matrix
            %a network input.
            %
            %Returns
            %-------
            %X : matrix
            %the output of the network's final layer
            
            for i = 1:length(obj.modules)
                X = obj.modules{i}.forward(X);
            end
        end
        
        
        function train(obj, X, Y, Xval, Yval, batchsize, iters, lrate, status, shuffle_data)
            if nargin < 10 || (exist('shuffle_data','var') && isempty(shuffle_data))
                shuffle_data = true;
            end
            
            if nargin < 9 || (exist('status','var') && isempty(status))
                status = 250;
            end
            
            if nargin < 8 || (exist('lrate','var') && isempty(lrate))
                lrate = 0.005;
            end
            
            if nargin < 7 || (exist('iters','var') && isempty(iters))
                iters = 10000;
            end
            
            if nargin < 6 || (exist('batchsize','var') && isempty(batchsize))
                batchsize = 25;
            end
            
            if nargin < 5 || (exist('Yval','var') && isempty(Yval)) || (exist('Xval','var') && isempty(Xval))
                Xval = X;
                Yval = Y;
            end
            
            [N,D] = size(X);
            if shuffle_data
               r = randperm(N);
               X = X(r,:);
               Y = Y(r,:);
            end
            
            for i = 0:(iters-1)
                samples = mod(i:i+batchsize-1,N)+1;
                Ypred = obj.forward(X(samples,:));
                obj.backward(Ypred - Y(samples,:));
                obj.update(lrate);
                
                if mod(i,status) == 0
                    Ypred = obj.forward(Xval);
                    [~, maxpred] = max(Ypred,[],2);
                    [~, maxtrue] = max(Yval,[],2);
                    acc = mean(maxpred == maxtrue);
                    fprintf('Accuracy after %i iterations: %f%%\n', i, acc*100);
                end
                
            end
            
        end
        
        
        function DY = backward(obj, DY)
            for i = length(obj.modules):-1:1
                DY = obj.modules{i}.backward(DY);   
            end
        end
        
        
        function update(obj, lrate)
            for i = 1:length(obj.modules)
                obj.modules{i}.update(lrate);   
            end
        end
        
    end
    
end