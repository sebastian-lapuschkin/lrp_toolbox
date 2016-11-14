function out = enlarge_image(img,scaling)
    % @author: Sebastian Lapuschkin
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %out = enlarge_image(img,scaling)
    %
    %Enlarges a given input matrix by replicating each pixel value scaling times in horizontal and vertical direction.
    %
    %Parameters
    %----------
    %
    %img : matrix or vector of shape [H x W x D] or [H x W]
    %
    %scaling : int
    %positive integer value > 0
    %
    %Returns
    %-------
    %
    %out : two-dimensional matrix or array of shape [scaling*H x scaling*W]

    if nargin < 2 || (exist('scaling','var') && isempty(scaling))
        scaling = 3;
    end

    scaling = round(scaling);
    if scaling < 1
       fprintf('scaling factor needs to be an integer >= 1\n')
    end
    
    if length(size(img)) == 2
        [H,W] = size(img);
        out = zeros(H*scaling,W*scaling);
        for h = 1:H
          fh = (h-1)*scaling+1;

          for w = 1:W
              fw = (w-1)*scaling+1;
              out(fh:fh+scaling-1,fw:fw+scaling-1) = img(h,w);
          end
        end
    
    elseif length(size(img)) == 3
        [H,W,D] = size(img);
        out = zeros(H*scaling, W*scaling, D);
        for h = 1:H
            fh = (h-1)*scaling+1;
            
            for w = 1:W
                fw = (w-1)*scalling+1;
                out(fh:fh+scaling-1,fw:fw+scaling-1) = img(h,w,:);
            end
        end
    end

end

