classdef Tanh < modules.Module
    % @author: Sebastian Bach
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Bach
    % @contact: sebastian.bach@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Bach, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    % Tanh Layer

    properties    
        %temporary variables
        Y
    end

    methods
        function obj = Tanh
        end

        function clean(obj)
            obj.Y = [];
        end

        function DY = backward(obj,DY)
           DY = DY.*(1.0 - obj.Y.^2); 
        end

        function Y = forward(obj,X)
            Y = tanh(X);
            obj.Y = Y;
        end

    end

end