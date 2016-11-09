classdef Flatten < modules.Module
    % @author: Sebastian Lapuschkin
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 09.11.2016
    % @version: 1.0
    % @copyright: Copyright (c) 2016, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    % Rectification Layer

    properties
        %temporary variables
        Y
    end

    methods
        function obj = Flatten
            obj = obj@modules.Module();
            obj.inputshape = [];
        end

        function Y = forward(obj,X)
            obj.inputshape = size(X); % N x H x W x D
            Y = reshape(X,[obj.inputshape(1) prod(obj.inputshape(2:end))]);
        end
        
        function DY = backward(obj,DY)
           DY = reshape(DY,obj.inputshape);
        end
        
        function R = lrp(obj,R,varargin)
            R = reshape(R,obj.inputshape);
        end

    end

end