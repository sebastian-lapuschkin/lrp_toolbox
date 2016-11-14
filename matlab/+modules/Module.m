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
        
        function R = lrp(obj,R,varargin)
            %implemented via inheriting classes
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