classdef Module < handle
    % @author: Sebastian Lapuschkin
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
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

        function R = lrp(obj,R,lrpvar,param)
            %implemented via inheriting classes
        end

        
        function DY = backward(obj,DY)
             %implemented via inheriting classes 
        end
        
        function train(obj, X, Y, batchsize, iters, lrate, status, shuffle_data)
            
        end

        function X = forward(obj,X)
            %implemented via inheriting classes 
        end
     
    end

end