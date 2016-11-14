classdef MaxPool < modules.Module
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
        %layer parameters
        stride
        pool
        
        %temporary variables
        Y
        X
    end

    methods
        function obj = MaxPool(pool,stride)
            % Constructor for the max pooling layer object
            % 
            % Parameters
            % ----------
            % 
            % pool : tuple (h,w)
            %     the size of the pooling mask in vertical (h) and horizontal (w) direction
            % 
            % stride : tuple (h,w)
            %     the vertical (h) and horizontal (w) step sizes between filter applications.
            
            obj = obj@modules.Module();

            if nargin < 2 || (exist('stride','var') && isempty(stride))
                obj.stride = [2,2];
            else
                obj.stride = stride;
            end
            if nargin < 1 || (exist('pool','var') && isempty(pool))
                obj.pool = [2,2];
            else
                obj.pool = pool;
            end
        end

        function Y = forward(obj,X)
            % Realizes the forward pass of an input through the max pooling layer.
            % 
            % Parameters
            % ----------
            % X : matrix
            %     a network input, shaped (N,H,W,D), with
            %     N = batch size
            %     H, W, D = input size in heigth, width, depth
            % 
            % Returns
            % -------
            % Y : matrix
            %     the max-pooled outputs, reduced in size due to given stride and pooling size
            
            obj.X = X;
            [N,H,W,D]= size(X);
            
            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);
            
            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;
            
            %initialize output
            obj.Y = zeros(N,Hout,Wout,D);
            for i = 1:Hout
               for j = 1:Wout
                  obj.Y(:,i,j,:) = max(max(X(:,(i-1)*hstride+1:(i-1)*hstride+hpool,(j-1)*wstride+1:(j-1)*wstride+wpool,:),[],2),[],3);
               end
            end
            Y = obj.Y; %'return'
        end
        
        function DX = backward(obj,DY)
            % Backward-passes an input error gradient DY towards the domintly ativating neurons of this max pooling layer.
            % 
            % Parameters
            % ----------
            % 
            % DY : matrix
            %     an error gradient shaped same as the output array of forward, i.e. (N,Hy,Wy,Dy) with
            %     N = number of samples in the batch
            %     Hy = heigth of the output
            %     Wy = width of the output
            %     Dy = output depth = input depth
            % 
            % 
            % Returns
            % -------
            % 
            % DX : matrix
            %     the error gradient propagated towards the input
            
            [N,H,W,D] = size(obj.X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;
            
            %distribute the gradient (1 * DY) towards all contributing
            %inputs evenly
            DX = zeros(N,H,W,D);
            for i = 1:Hout
                for j = 1:Wout
                    x = obj.X(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    y = repmat(obj.Y(:,i,j,:),[1 hpool wpool 1]);
                    
                    dy = repmat(DY(:,i,j,:),[1 hpool wpool 1]);
                    dx = DX(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                         
                    DX(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :) = dx + dy .* (x == y);
                end
            end     
        end
        
        function clean(obj)
           obj.X = [];
           obj.Y = [];
        end
        
          
        function Rx = simple_lrp(obj,R)
            % LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
            [N,H,W,D] = size(obj.X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;
            
            Rx = zeros(N,H,W,D);
            for i = 1:Hout
                for j = 1:Wout
                    x = obj.X(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    y = repmat(obj.Y(:,i,j,:),[1 hpool wpool 1]);
                    
                    Z = x == y;
                    Zs = sum(sum(Z,2),3);

                    rr = repmat(R(:,i,j,:),[1,hpool,wpool,1]);
                    zz = Z ./ repmat(Zs,[1,hpool,wpool,1]);
                    rx = Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    
                    Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :) = rx + rr .* zz;
                end
            end
        end
        
        function Rx = flat_lrp(obj,R)
            % distribute relevance for each output evenly to the output neurons' receptive fields.
            [N,H,W,D] = size(obj.X);

            hpool = obj.pool(1);        wpool = obj.pool(2);
            hstride = obj.stride(1);    wstride = obj.stride(2);

            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hpool)/hstride + 1;
            Wout = (W - wpool)/wstride + 1;
            
            Rx = zeros(N,H,W,D);
            for i = 1:Hout
                for j = 1:Wout              
                    Z = ones(N,hpool,wpool,D);
                    Zs = sum(sum(Z,2),3);

                    rr = repmat(R(:,i,j,:),[1,hpool,wpool,1]);
                    zz = Z ./ repmat(Zs,[1,hpool,wpool,1]);
                    rx = Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :);
                    
                    Rx(: , (i-1)*hstride+1:(i-1)*hstride+hpool , (j-1)*wstride+1:(j-1)*wstride+wpool , :) = rx + rr .* zz;
                end
            end
        end
        
        function Rx = ww_lrp(obj,R)
            % There are no weights to use. default to flat_lrp(R)
            Rx = obj.flat_lrp(R);
        end
        
        function Rx = epsilon_lrp(obj,R,epsilon)
            % Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
            Rx = obj.simple_lrp(R);
        end
        
        function Rx = alphabeta_lrp(obj,R,alpha)
            % Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
            Rx = obj.simple_lrp(R);
        end
    end
end