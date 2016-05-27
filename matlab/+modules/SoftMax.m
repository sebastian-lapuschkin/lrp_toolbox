classdef SoftMax < modules.Module
    % @author: Sebastian Bach
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Bach
    % @contact: sebastian.bach@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Bach, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    % Softmax Layer


    properties    
        %temporary variables
        X
        Y
    end

    methods
        function obj = SoftMax
        end

        function clean(obj)
            obj.X = [];
            obj.Y = [];
        end

        function Y = forward(obj,X)
            eX = exp(X);
            Y = eX ./ repmat(sum(eX,2),1,size(eX,2)) ;
            obj.X = X;
            obj.Y = Y;
        end

        function R = lrp(obj, R, lrp_var, param)
            R = R .* obj.X;
        end

    end
end