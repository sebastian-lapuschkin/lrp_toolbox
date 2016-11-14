classdef Linear < modules.Module
    % @author: Sebastian Lapuschkin
    % @author: Gregoire Montavon
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    % This module implements a linear neural network layer

    properties
        % model parameters
        m
        n
        B
        W

        %temporary variables
        X
        Y
        dW
        dB
    end

    methods
        function obj = Linear(m,n)
            obj = obj@modules.Module();
            obj.m = m;
            obj.n = n;
            obj.B = zeros(1,n);
            obj.W = randn(m,n).* m .^ (-.5);
        end
        
        function Y = forward(obj,X)
            Y = X * obj.W + repmat(obj.B,size(X,1),1);
            obj.X = X;
            obj.Y = Y;
        end

        function DY = backward(obj,DY)
            obj.dW = obj.X'*DY;
            obj.dB = sum(DY,1);
            DY = (DY * obj.W') .* obj.m .^ .5 ./ obj.n .^ .5;
        end

        function update(obj, lrate)
           obj.W = obj.W - lrate .* obj.dW ./ obj.m .^ .5;
           obj.B = obj.B - lrate .* obj.dB ./ obj.n .^ .25;
        end

        function clean(obj)
            obj.X = [];
            obj.Y = [];
            obj.dW = [];
            obj.dB = [];
        end


        
        
        
       function R = simple_lrp(obj,R)
           % LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
           N = size(obj.X,1);
           Wr = repmat(reshape(obj.W,[1,obj.m,obj.n]),[N,1,1]);
           Xr = repmat(obj.X,[1,1,obj.n]);

           %localized preactivations
           Z = Wr .* Xr ;
           Zs = sum(Z,2) + repmat(reshape(obj.B,[1,1,obj.n]),[N,1,1]);

           Rr = repmat(reshape(R,[N,1,obj.n]),[1,obj.m,1]);
           R = sum((Z ./ repmat(Zs,[1,obj.m,1])) .* Rr,3);
       end
       
       
       function R = flat_lrp(obj,R)
           % distribute relevance for each output evenly to the output neurons' receptive fields.
           % note that for fully connected layers, this results in a uniform lower layer relevance map.
           N = size(obj.X,1);
           %localized preactivations
           Z = ones(N, obj.m, obj.n);
           Zs = sum(Z,2);

           Rr = repmat(reshape(R,[N,1,obj.n]),[1,obj.m,1]);
           R = sum((Z ./ repmat(Zs,[1,obj.m,1])) .* Rr,3);
       end
       
       function R = ww_lrp(obj,R)
           % LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
           N = size(obj.X,1);
           Wr = repmat(reshape(obj.W,[1,obj.m,obj.n]),[N,1,1]);

           %localized preactivations
           Z = Wr.^2 ;
           Zs = sum(Z,2);

           Rr = repmat(reshape(R,[N,1,obj.n]),[1,obj.m,1]);
           R = sum((Z ./ repmat(Zs,[1,obj.m,1])) .* Rr,3);
       end

       function R = epsilon_lrp(obj,R,epsilon)
           % LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
           N = size(obj.X,1);
           Wr = repmat(reshape(obj.W,[1,obj.m,obj.n]),[N,1,1]);
           Xr = repmat(obj.X,[1,1,obj.n]);

           %localized preactivations
           Z = Wr .* Xr ;
           Zs = sum(Z,2) + repmat(reshape(obj.B,[1,1,obj.n]),[N,1,1]);
           Zs = Zs + epsilon .* ((Zs >= 0)*2-1);

           Rr = repmat(reshape(R,[N,1,obj.n]),[1,obj.m,1]);
           R = sum((Z ./ repmat(Zs,[1,obj.m,1])) .* Rr,3);
       end


       function R = alphabeta_lrp(obj,R,alpha)
           % LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
           N = size(obj.X,1);
           Wr = repmat(reshape(obj.W,[1,obj.m,obj.n]),[N,1,1]);
           Xr = repmat(obj.X,[1,1,obj.n]);
           Rr = repmat(reshape(R,[N,1,obj.n]),[1,obj.m,1]);

           
           beta = 1 - alpha;
           Z = Wr .* Xr ; %localized preactivations
           
           if ~(alpha == 0)
                Zp = Z .* (Z > 0);
                Zsp = sum(Zp,2) + repmat(reshape(obj.B .* (obj.B > 0),[1,1,obj.n]),[N,1,1]);
                Ralpha = alpha .* sum((Zp ./ repmat(Zsp,[1,obj.m,1])) .* Rr,3);
           else
                Ralpha = 0; 
           end

           if ~(beta == 0)
                Zn = Z .* (Z < 0);
                Zsn = sum(Zn,2) + repmat(reshape(obj.B .* (obj.B < 0),[1,1,obj.n]),[N,1,1]);
                Rbeta = beta .* sum((Zn ./ repmat(Zsn,[1,obj.m,1])) .* Rr,3);
           else
                Rbeta = 0;
           end
           
           R = Ralpha + Rbeta;
       end




    end

end