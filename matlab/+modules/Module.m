classdef Module < handle
    % @author: Sebastian Lapuschkin
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.2+
    % @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    % Superclass for all computation layer implementations

    properties
        %pre-set lrp parameters
        lrp_var
        lrp_param
    end

    methods
        function obj = Module()
            %implemented via inheriting classes
        end

        function update(obj,lrate)
            %implemented via inheriting classes
        end

        function clean(obj)
            %implemented via inheriting classes
        end

        function DY = backward(obj,DY)
             %implemented via inheriting classes
        end

        function train(obj, X, Y, varargin)
            error(['Training not implemented for class ' class(obj)])
        end

        function X = forward(obj,X)
            %implemented via inheriting classes
        end
        
        
        
        function set_lrp_parameters(obj,lrp_var,param)
            % pre-sets lrp parameters to use for this layer. see the documentation of Module.lrp for details
            
           if nargin < 3 || (exist('param','var') && isempty(param))
               param = [];
           end
           if nargin < 2 || (exist('lrp_var','var') && isempty(lrp_var))
               lrp_var = [];
           end
           
           obj.lrp_var = lrp_var;
           obj.lrp_param = param;
        end
        
        
        function R = lrp(obj,R,lrp_var, param)
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
           
           if isempty(lrp_var) && isempty(param)
               % module.lrp(R) has been called without further parameters.
               % set default values / preset values
               lrp_var = obj.lrp_var;
               param = obj.lrp_param;
           end

           if isempty(lrp_var) || strcmpi(lrp_var,'none') || strcmpi(lrp_var,'simple')
              R = obj.simple_lrp(R);
           elseif strcmpi(lrp_var,'flat')
              R = obj.flat_lrp(R);
           elseif strcmpi(lrp_var,'ww') || strcmpi(lrp_var,'w^2')
              R = obj.ww_lrp(R);
           elseif strcmpi(lrp_var,'epsilon')
              R = obj.epsilon_lrp(R,param);
           elseif strcmpi(lrp_var,'alphabeta') || strcmpi(lrp_var, 'alpha')
              R = obj.alphabeta_lrp(R,param);
           else
              error('unknown lrp variant %s\n',lrp_var)
           end

       end
        
        % ---------------------------------------------------------
        % Methods below should be implemented by inheriting classes
        % ---------------------------------------------------------
        
        function R = simple_lrp(R)
            error(['simple_lrp not implemented for class ' class(obj)])
        end
        
        function R = flat_lrp(R)
            error(['flat_lrp not implemented for class ' class(obj)])
        end
        
        function R = ww_lrp(R)
            error(['ww_lrp not implemented for class ' class(obj)])
        end

        function R = epsilon_lrp(R,epsilon)
            error(['epsilon_lrp not implemented for class ' class(obj)])
        end
        
        function R = alphabeta_lrp(R,alpha)
            error(['alpha_lrp not implemented for class ' class(obj)])
        end

        
    end

end