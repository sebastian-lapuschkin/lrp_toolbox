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
            obj.B = zeros(1,filtersize(4));
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
            %Wr = reshape(obj.W,[1 obj.filtersize]);
            %Wr = repmat(Wr,[N 1 1 1 1]);
            Br = reshape(obj.B,[1 1 1 Nf]);
            Br = repmat(Br,[N Hout Wout 1]);
            
            %initialize output
            obj.Y = zeros(N,Hout,Wout,Nf);
            for i = 1:Hout
               for j = 1:Wout
                  x = X(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:);
                  obj.Y(:,i,j,:) = reshape(x,[N (hf*wf*df)]) * reshape(obj.W, [(hf * wf * df) Nf]);
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
            [N,Hx,Wx,Dx] = size(obj.X);
            [N,Hy,Wy,Nf] = size(DY);
            [hf,wf,df,Nf] = size(obj.W);
    
            hstride = obj.stride(1);    wstride = obj.stride(2);

            DX = zeros(N,Hx,Wx,Dx);
           
            Wr = reshape(obj.W,[1 hf wf df Nf]);
            Wr = repmat(Wr,[N 1 1 1 1]); % N hf wf df Nf
            
            for i = 1:Hy
                for j = 1:Wy
                    dx = DX(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:); % N hf wf df
                    dy = DY(:,i,j,:); % N x 1 x 1 x Nf
                    dy = repmat(reshape(dy,[N 1 1 1 Nf]),[1 hf wf df 1]); % N hf wf df Nf
                    DX(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:) = dx + sum(Wr .* dy,5);
                end
            end
            
            DX = DX .* (hf*wf*df)^.5 ./ (Nf*Hy*Wy)^.5;
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
                    x = repmat(x,[1 1 1 1 Nf]); % N x hf x wf x df x Nf
                    dy = repmat(reshape(dy,[N 1 1 1 Nf]), [1 hf wf df 1]); % N x hf x wf x df x Nf
                    
                    DW = DW + reshape(sum((x .* dy),1),size(DW));
                end
            end
            
            DB = sum(sum(sum(obj.DY,1),2),3);
            obj.W = obj.W - lrate .* DW ./ (hf*wf*df)^.5 ;
            obj.B = obj.B - lrate .* reshape(DB,size(obj.B)) ./ (Nf*Hy*Wy)^.25;
            
        end
       
        
        function clean(obj)
           obj.X = [];
           obj.Y = [];
           obj.DY = [];
        end

        
        
        function R = lrp(obj,R,lrp_var, param)
           % performs LRP by calling subroutines, depending on lrp_var and param
           %
           % Parameters
           % ----------
           %
           % R : matrix
           % relevance input for LRP.
           % should be of the same shape as the previusly produced output by Convolution.forward
           %
           % lrp_var : str
           % either 'none' or 'simple' or None for standard Lrp ,
           % 'epsilon' for an added epsilon slack in the denominator
           % 'alphabeta' or 'alpha' for weighting positive and negative contributions separately. param specifies alpha with alpha + beat = 1
           % 'flat' projects an upper layer neuron's relevance uniformly over its receptive field.
           % 'ww' or 'w^2' only considers the square weights w_ij^2 as qantities to distribute relevances with.
           %
           % param : double
           % the respective parameter for the lrp method of choice
           %
           % Returns
           % -------
           % R : the backward-propagated relevance scores.
           % shaped identically to the previously processed inputs in Linear.forward

           if nargin < 4 || (exist('param','var') && isempty(param))
               param = 0;
           end
           if nargin < 3 || (exist('lrp_var','var') && isempty(lrp_var))
               lrp_var = [];
           end

           if isempty(lrp_var) || strcmpi(lrp_var,'none') || strcmpi(lrp_var,'simple')
              R = obj.simple_lrp(R);
           elseif strcmpi(lrp_var,'flat')
              R = obj.flat_lrp(R);
           elseif strcmpi(lrp_var,'ww') || strcmpi(lrp_var,'w^2')
              R = obj.ww_lrp(R);
           elseif strcmpi(lrp_var,'epsilon')
              R = obj.epsilon_lrp(R,param);
           elseif strcmpi(lrp_var,'alphabeta') || stcmpi(lrp_var, 'alpha')
              R = obj.alphabeta_lrp(R,param);
           else
              fprintf('unknown lrp variant %s\n',lrp_var)
           end

       end
        
       
        function Rx = simple_lrp(obj,R)
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
        
        function Rx = flat_lrp(obj,R)
            % distribute relevance for each output evenly to the output neurons' receptive fields.
            
            [N,Hx,Wx,df] = size(obj.X);
            [N,Hout,Wout,Nf] = size(R);
            [hf,wf,df,Nf] = size(obj.W);
            hstride = obj.stride(1);    wstride = obj.stride(2);
            
            Rx = zeros(N,Hx,Wx,df);
            for i = 1:Hout
                for j = 1:Wout
                    Z = ones(N,hf,wf,df,Nf);
                    
                    Zs = sum(sum(sum(Z,2),3),4);
                    Zs = repmat(reshape(Zs,[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                    
                    zz = Z ./ Zs ; % N x hf x wf x df x Nf
                    rr = repmat(reshape(R(:,i,j,:),[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                    rx = Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:); % N x hf x wf x df
                     
                    Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:) = rx +  sum(zz .* rr,5);
                end
            end
        end
        
        function Rx = ww_lrp(obj,R)
            % LRP according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
            
            [N,Hx,Wx,df] = size(obj.X);
            [N,Hout,Wout,Nf] = size(R);
            [hf,wf,df,Nf] = size(obj.W);
            hstride = obj.stride(1);    wstride = obj.stride(2);
            
            
            %prepare W for the loop below
            Wr = reshape(obj.W,[1 obj.filtersize]);
            Wr = repmat(Wr,[N 1 1 1 1]);

            
            Rx = zeros(N,Hx,Wx,df);
            for i = 1:Hout
                for j = 1:Wout       
                    Z = Wr; % N x hf x wf x df x Nf
                    
                    Zs = sum(sum(sum(Z,2),3),4);  % N x Nf
                    Zs = repmat(reshape(Zs,[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                    
                    zz = Z ./ Zs ; % N x hf x wf x df x Nf
                    rr = repmat(reshape(R(:,i,j,:),[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                    rx = Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:); % N x hf x wf x df
                     
                    Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:) = rx +  sum(zz .* rr,5);
                end
            end
        end
        
        
        function Rx = epsilon_lrp(obj,R,epsilon)
            % LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
            
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
                    Zs = Zs + epsilon*((Zs >= 0)*2 -1); %add a weak numerical stabilizer to cushion division by zero
                    Zs = repmat(reshape(Zs,[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                    
                    zz = Z ./ Zs ; % N x hf x wf x df x Nf
                    rr = repmat(reshape(R(:,i,j,:),[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                    rx = Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:); % N x hf x wf x df
                     
                    Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:) = rx +  sum(zz .* rr,5);
                end
            end
        end
        
        
        function Rx = alphabeta_lrp(obj,R,alpha)
            % LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
            
            [N,Hx,Wx,df] = size(obj.X);
            [N,Hout,Wout,Nf] = size(R);
            [hf,wf,df,Nf] = size(obj.W);
            hstride = obj.stride(1);    wstride = obj.stride(2);
            
            
            %prepare W and B for the loop below
            Wr = reshape(obj.W,[1 obj.filtersize]);
            Wr = repmat(Wr,[N 1 1 1 1]);
            Br = reshape(obj.B,[1 Nf]);
            Br = repmat(Br,[N 1]);
            
            beta = 1 - alpha;
            Rx = zeros(N,Hx,Wx,df);
            for i = 1:Hout
                for j = 1:Wout
                    x = obj.X(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:);
                    x = repmat(x,[1 1 1 1 Nf]);
                    rr = repmat(reshape(R(:,i,j,:),[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                                        
                    Z = Wr .* x; % N x hf x wf x df x Nf
                    
                    if ~(alpha == 0)
                        Zp = Z .* (Z > 0);
                        Brp = Br .* (Br > 0);
                        
                        Zsp = sum(sum(sum(Zp,2),3),4);
                        Zsp = Zsp + reshape(Brp, size(Zsp)) ; % N x Nf
                        Zsp = repmat(reshape(Zsp,[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                        
                        Ralpha = alpha .* sum(Zp ./ Zsp .* rr,5);
                    else
                        Ralpha = 0;
                    end
                    
                    if ~(beta == 0)
                        Zn = Z .* (Z < 0);
                        Brn = Br .* (Br < 0);
                        
                        Zsn = sum(sum(sum(Zn,2),3),4);
                        Zsn = Zsn + reshape(Brn, size(Zsn)) ; % N x Nf
                        Zsn = repmat(reshape(Zsn,[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
                        
                        Rbeta = beta .* sum(Zn ./ Zsn .* rr,5);
                    else
                        Rbeta = 0;
                    end
                    
                    
                    rx = Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:); % N x hf x wf x df   
                    Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:) = rx + Ralpha + Rbeta;
                end
            end
        end
    end
end