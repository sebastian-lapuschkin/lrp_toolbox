classdef Rect < modules.Module
    % @author: Sebastian Lapuschkin
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
    % @license : BSD-2-Clause
    %
    % Rectification Layer

    properties
        %temporary variables
        Y
    end

    methods
        function obj = Rect
            obj = obj@modules.Module();
        end

        function clean(obj)
            obj.Y = [];
        end

        function DY = backward(obj,DY)
           DY = DY .* (obj.Y ~= 0);
        end

        function Y = forward(obj,X)
            Y = max(0,X);
            obj.Y = Y;
        end

        function R = lrp(obj,R,varargin)
            % component-wise operations within this layer
            % ->
            % just propagate R further down.
            % makes sure subroutines never get called.
        end

    end

end