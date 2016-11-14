classdef Sequential < modules.Module
    % @author: Sebastian Lapuschkin
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.2+
    % @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
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
            obj = obj@modules.Module();
            obj.modules = modules;
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

        function clean(obj)
            % Removes temporary variables from all network layers.
            for i = 1:length(obj.modules)
                obj.modules{i}.clean()
            end
        end


        function train(obj, X, Y, Xval, Yval, batchsize, iters, lrate, lrate_decay, lfactor_initial, status, convergence, transform)
            % Sequential.train( X, Y, Xval, Yval, batchsize, iters, lrate, lrate_decay, lfactor_initial, status, convergence, transform)
            %
            % Provides a method for training the nerual net (obj) based on
            % given data. Parameters you do not wish to set manually should
            % be specified as []
            %
            % Parameters
            % ----------
            %
            % X : matrix
            % the training data, formatted as a [N,D]-shaped matrix, with N
            % being the number of samples and D their dimensionality.
            %
            % Y : matrix
            % the training labels, formatted as a [N,C] shaped label matrix,
            % with N being the number of samples and C being the training
            % classes
            %
            % Xval : matrix, default value is []
            % some optional validation data. used to measure network
            % performance during training. shaped [M,D]
            %
            % Yval : matrix, default value is []
            % the validation labels. shaped [m,C]
            %
            % batchsize : int, default value is 25
            % the batch size to use for training
            %
            % iters : int, default value is 10000
            % max number of training iterations
            %
            % lrate : float, default value is 0.005
            % the initial learning rate. the learning rate is adjusted
            % during training with increased model performance. See
            % lrate_decay
            %
            % lrate_decay : string, default value is []
            % controls if and how the learning rate is adjusted throughout
            % the training:
            % 'none' or [] disables learning rate adaption.
            % 'sublinear' adjusts the learning rate to lrate*(1-Accuracy^2)
            %   during an evaluation step often resulting in a better
            %   performing model
            % 'linear' adjusts the learning rate to lrate*(1-Accuracy)
            %   during an evaluation step, often resulting in a better
            %   performing model
            %
            % lfactor_initial : float, default value is 1.0
            % specifies an initial discount on the given learning rate, e.g. when retraining an established network in combination with a learning rate decay,
            % it might be undesirable to use the given learning rate in the beginning. this could have been done better. TODO: do better.
            %
            % status : int, default value is 250
            % number of iterations (i.e. the number of rounds of batch
            % forward pass, gradient backward pass, parameter update) of
            % silent training, until status print and evaluation on
            % validation data
            %
            % convergence : int, default value is -1
            % number of consecutive allowed sttatus evaluations with no
            % more morel improvements until we accept the model has
            % converged.
            % Set <=0 to disable.
            % Set to any value > 0  to control the maximal consecutive
            % number (status * convergence) iterations allowed without
            % model improvement, until convergence is accepted.
            %
            % transform : function handle, default value is []
            % a function taking as an input a batch of training data sized
            % [N,D] and returning a batch sized [N,D] with added noise or
            % other various data transformations. It's up to you!
            % the default value [] causes no data transformation.
            % expected syntax is, with size(X) == [N,D] == size(Xt)
            % function Xt = yourFunction(X):
            %    Xt = someStuff(X);

            %first, set default values whereever necessary
            if nargin < 13 || (exist('transform','var') && isempty(transform))
               transform = [];
            end

            if nargin < 12 || (exist('convergence','var') && isempty(convergence))
               convergence = -1;
            end

            if nargin < 11 || (exist('status','var') && isempty(status))
               status = 250;
            end

            if nargin < 10 || (exist('lfactor_initial','var') && isempty(lfactor_initial))
               lfactor_initial = 1.0;
            end

            if nargin < 9 || (exist('lrate_decay','var') && isempty(lrate_decay))
               lrate_decay = [];
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
                Xval = [];
                Yval = [];
            end

            %start training
            untilConvergence = convergence; learningFactor = lfactor_initial;
            bestAccuracy = 0.0;             bestLayers = obj.modules;

            N = size(X,1);
            for d = 1:iters

                %the actual training:
                %first, pick samples at random
                samples = randperm(N, batchsize);

                %transform batch data (maybe)
                if isempty(transform)
                    batch = X(samples,:,:,:);
                else
                    batch = transfor(X(samples,:,:,:));
                end

                %forward and backward propagation steps with parameter
                %update
                Ypred = obj.forward(batch);
                obj.backward(Ypred - Y(samples,:));
                obj.update(lrate*learningFactor);

                %periodically evaluate network and optionally adjust
                %learning rate or check for convergence
                if mod(d,status) == 0
                    
                    %if given, also evaluate on validation data
                    if ~isempty(Xval) && ~isempty(Yval)
                       Ypred = obj.forward(Xval);
                       [~,argmaxPred]  = max(Ypred,[],2);
                       [~,argmaxTruth] = max(Yval,[],2);
                       acc = mean(argmaxPred == argmaxTruth);
                       l1loss = sum(abs(Ypred(:) - Yval(:)))/size(Yval,1);
                       disp(' ')
                       disp(['Accuracy after ' num2str(d) ' iterations on validation set: ' num2str(acc*100) '% (l1-loss: '  num2str(l1loss) ')'])
                       
                    else %evaluate on training data only   
                        Ypred = obj.forward(X);
                        [~,argmaxPred]  = max(Ypred,[],2);
                        [~,argmaxTruth] = max(Y,[],2);
                        acc = mean(argmaxPred == argmaxTruth);
                        l1loss = sum(abs(Ypred(:) - Y(:)))/size(Y,1);
                        disp(' ')
                        disp(['Accuracy after ' num2str(d) ' iterations: ' num2str(acc*100) '% (l1-loss: '  num2str(l1loss) ')'])
                    end

                    %save current network parameters if we have improved
                    if acc > bestAccuracy
                        disp('    New optional parameter set encountered. saving...')
                        bestAccuracy = acc;
                        bestLayers = obj.modules;

                        %adjust learning rate
                        if isempty(lrate_decay) || strcmp(lrate_decay,'none')
                            %no adjustment
                        elseif strcmp(lrate_decay,'sublinear')
                            %slow down learning to better converge towards an optimum with increased network performance.
                            learningFactor = 1 - acc^2;
                            disp(['    Adjusting learning rate to ' num2str(learningFactor*lrate) ' ~ ' num2str(round(learningFactor*100,2)) '% of its initial value'])
                        elseif strcmp(lrate_decay,'linear')
                            learningFactor = 1 - acc;
                            disp(['    Adjusting learning rate to ' num2str(learningFactor*lrate) ' ~ ' num2str(round(learningFactor*100,2)) '% of its initial value'])
                        end

                        %refresh number of allowed search steps until
                        %convergence
                        untilConvergence = convergence;
                    else
                        untilConvergence = untilConvergence - 1;
                        if untilConvergence == 0 && convergence > 0
                            disp(['    No more recorded model improvements for ' num2str(convergence) ' evaluations. Accepting model convergence.'])
                            break
                        end
                    end

                elseif mod(d,status/10) == 0
                    % print 'alive' signal
                    % fprintf('.')
                    Ysamples = Y(samples,:);
                    l1loss = sum(abs(Ypred(:) - Ysamples(:)))/size(Ypred,1);
                    disp(['batch# ' num2str(d) ', lrate ' num2str(lrate) ', l1-loss ' num2str(l1loss)])
                end

            end

            %after training, either due to convergence or iteration limit:
            %set best encountered parameters as network parameters
            disp(['Setting network parameters to best encountered network state with ' num2str(bestAccuracy*100) '% accuracy.'])
            obj.modules = bestLayers;
        end




        function set_lrp_parameters(obj,lrp_var,param)
            if nargin < 3 || (exist('param','var') && isempty(param))
                param = [];
            end
            if nargin < 2 || (exist('lrp_var','var') && isempty(lrp_var))
                lrp_var = [];
            end
            
            for i = 1:length(obj.modules)
                obj.modules{i}.set_lrp_parameters(lrp_var,param);
            end
        end
        
        function R = lrp(obj,R, lrp_var, param)
            % Performs LRP by calling subroutines, depending on lrp_var and param or
            % preset values specified via Module.set_lrp_parameters(lrp_var,lrp_param)
            % 
            % If lrp parameters have been pre-specified (per layer), the corresponding decomposition
            % will be applied during a call of lrp().
            % 
            % Specifying lrp parameters explicitly when calling lrp(), e.g. net.lrp(R,lrp_var='alpha',param=2.),
            % will override the preset values for the current call.
            % 
            % How to use:
            % 
            % net.forward(X) #forward feed some data you wish to explain to populat the net.
            % 
            % then either:
            % 
            % net.lrp() #to perform the naive approach to lrp implemented in _simple_lrp for each layer
            % 
            % or:
            % 
            % for i = 1:length(net.modules)
            %     net.modules{i}.set_lrp_parameters(...)
            % end
            % net.lrp() #to preset a lrp configuration to each layer in the net
            % 
            % or:
            % 
            % net.lrp(somevariantname,someparameter) # to explicitly call the specified parametrization for all layers (where applicable) and override any preset configurations.
            % 
            % Parameters
            % ----------
            % 
            % R : matrix or tensor
            %     relevance input for LRP.
            %     should be of the same shape as the previously produced output by <Module>.forward
            % 
            % lrp_var : str
            %     either 'none' or 'simple' or None for standard Lrp ,
            %     'epsilon' for an added epsilon slack in the denominator
            %     'alphabeta' or 'alpha' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1
            %     'flat' projects an upper layer neuron's relevance uniformly over its receptive field.
            %     'ww' or 'w^2' only considers the square weights w_ij^2 as qantities to distribute relevances with.
            % 
            % param : double
            %     the respective parameter for the lrp method of choice
            % 
            % Returns
            % -------
            % R : the backward-propagated relevance scores.
            %     shaped identically to the previously processed inputs in <Module>.forward
            if nargin < 4 || (exist('param','var') && isempty(param))
                param = [];
            end
            if nargin < 3 || (exist('lrp_var','var') && isempty(lrp_var))
                lrp_var = [];
            end

            for i = length(obj.modules):-1:1
                R = obj.modules{i}.lrp(R,lrp_var,param);
            end
        end


        

    end

end