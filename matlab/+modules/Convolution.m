classdef Convolution < modules.Module
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
        %model parameters
        filtersize
        stride
        W
        B
        
        %temporary variables
        Y
        X
        DY
    end

    methods
        function obj = Convolution(filtersize,stride)
            % Constructor for a Convolution layer.
            % 
            % Parameters
            % ----------
            % 
            % filtersize : values (h,w,d,n), where
            %     h = filter heigth
            %     w = filter width
            %     d = filter depth
            %     n = number of filters = number of outputs
            % 
            % stride : (h,w), where
            %     h = step size for filter application in vertical direction
            %     w = step size in horizontal direction
            
            
            obj = obj@modules.Module();

            if nargin < 2 || (exist('stride','var') && isempty(stride))
                obj.stride = [2,2];
            else
                obj.stride = stride;
            end
            if nargin < 1 || (exist('filtersize','var') && isempty(filtersize))
                obj.filtersize = [5,5,3,32];
            else
                obj.filtersize = filtersize;
            end
            
            obj.W = randn(obj.filtersize) ./ (prod(obj.filtersize(1:3)) .^ .5);
            obj.B = zeros(filtersize(4));
        end

        function Y = forward(obj,X)
            % Realizes the forward pass of an input through the convolution layer.
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
            %     the layer outputs.
            
            obj.X = X;
            [N,H,W,D]= size(X);
            [hf,wf,df,Nf] = size(obj.W);
            
            hstride = obj.stride(1);    wstride = obj.stride(2);
            
            %assume the given pooling and stride parameters are carefully
            %chosen
            Hout = (H - hf)/hstride + 1;
            Wout = (W - wf)/wstride + 1;
            
            
            %prepare W and B for the loop below
            Wr = reshape(obj.W,[1 obj.filtersize]);
            Wr = repmat(Wr,[N 1 1 1 1]);
            Br = reshape(obj.B,[1 1 1 Nf]);
            Br = repmat(Br,[N Hout Wout 1]);
            
            %initialize output
            obj.Y = zeros(N,Hout,Wout,Nf);
            for i = 1:Hout
               for j = 1:Wout
                  x = X(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:);
                  xr = repmat(x,[1 1 1 1 Nf]);
                  
                  obj.Y(:,i,j,:) = sum(sum(sum(Wr .* xr,2),3),4);
               end
            end
            obj.Y = obj.Y + Br;
            Y = obj.Y; %'return'
        end
        
        function DX = backward(obj,DY)
            % Backward-passes an input error gradient DY towards the input neurons of this layer.
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
            
            obj.DY = DY;
            [N,H,D,W] = size(obj.X);
            [n,Hy,Wy,NF] = size(DY);
            [hf,wf,df,NF] = size(obj.W);
    
            hstride = obj.stride(1);    wstride = obj.stride(2);

            DX = zeros(N,H,D,W);
            for i = 1:Hout
                for j = 1:Wout
                    dx = DX(: , i:hstride:i+Hy , j:wstride:j+Hx, :);
                    DX(: , i:hstride:i+Hy , j:wstride:j+Wy, :) = dx + DY * permute(obj.W(i,j,:,:),[4 3 2 1]);
                end
            end     
        end

        
        function update(obj, lrate)
            
            [N,Hx,Wx,Dx] = size(obj.X);
            [N,Hy,Wy,Nf] = size(obj.DY);
            
            [hf,wf,df,Nf] = size(obj.W);
            hstride = obj.stride(1);    wstride = obj.stride(2);
            
            DW = zeros(hf,wf,df,Nf);
            for i = 1:Hy
                for j = 1:Wy              
                    x = obj.X(:,(i-1)*hstride+1:(i-1)*hstride+hf ,(j-1)*wstride+1:(j-1)*wstride+wf,:); % N x hf x wf x df
                    dy = obj.DY(:,i,j,:); % N x 1 x 1 x Nf
                    
                    %these next two lines are horribly ineffiecient.
                    x = repmat(x,[1 1 1 1 nf]); % N x hf x wf x df x Nf
                    dy = repmat(reshape(dy,[N 1 1 1 nf]), [1 hf wf df 1]); % N x hf x wf x df x Nf
                    
                    DW = DW + sum((x .* dy),1);
                end
            end
            
            DB = sum(sum(sum(self.DY,1),2),3);
            obj.W = obj.W - lrate .* DW;
            obj.B = obj.B - lrate .* DB;
            
        end
       
        
        function clean(obj)
           obj.X = [];
           obj.Y = [];
           obj.DY = [];
        end

        
        
       
        function Rx = lrp(obj,R,varargin)
            % LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
            
            [N,Hx,Wx,df] = size(obj.X);
            [N,Hout,Wout,Nf] = size(R);
            [hf,wf,df,Nf] = size(obj.W);
            hstride = obj.stride(1);    wstride = obj.stride(2);
            
            
            %prepare W and B for the loop below
            Wr = reshape(obj.W,[1 obj.filtersize]);
            Wr = repmat(Wr,[N 1 1 1 1]);
            Br = reshape(obj.B,[1 Nf]);
            Br = repmat(Br,[N 1]);
            
            Rx = zeros(N,Hx,Wx,df);
            for i = 1:Hout
                for j = 1:Wout
                    x = obj.X(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:);
                    x = repmat(x,[1 1 1 1 Nf]);
                    Z = Wr .* x; % N x hf x wf x df x Nf
                    
                    Zs = sum(sum(sum(Z,2),3),4);
                    Zs = Zs + reshape(Br, size(Zs)) ; % N x Nf
                    Zs = Zs + 1e-12*((Zs >= 0)*2 -1); %add a weak numerical stabilizer to cushion division by zero
                    Zs = repmat(reshape(Zs,[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                    
                    zz = Z ./ Zs ; % N x hf x wf x df x Nf
                    rr = repmat(reshape(R(:,i,j,:),[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                    rx = Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:); % N x hf x wf x df
                    
                    Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:) = rx +  sum(zz .* rr,5);
                end
            end
        end
    end
end