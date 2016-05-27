function rgbimg = repaint_corner_pixels(rgbimg, scaling)
    % @author: Sebastian Bach
    % @maintainer: Sebastian Bach
    % @contact: sebastian.bach@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Bach, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %rgbimg = repaint_corner_pixels(rgbimg, scaling)
    %
    %Recolors the top left and bottom right pixel (groups) with the average rgb value of its three neighboring pixel (groups).
    %The recoloring visually masks the opposing pixel values which are a product of stabilizing the scaling.
    %Assumes those image ares will pretty much never show evidence.
    %     
    %TODO: find a smarter way to do this. I know a smarter way, yet am too lazy to bother.
    %     
    %Parameters
    %----------
    %     
    %rgbimg : matrix or vector shape [H x W x 3]
    %         
    %scaling : int
    %positive integer value > 0
    %     
    %Returns
    %-------
    %     
    %rgbimg : three-dimensional matrix or vector of shape [scaling*H x scaling*W x 3] 

    if nargin < 2 || (exist('scaling','var') && isempty(scaling))
        scaling = 3;
    end
    
    %top left corner
    fill = (rgbimg(1,scaling+1,:)+rgbimg(scaling+1,1,:)+rgbimg(scaling+1,scaling+1,:))/3.;
    rgbimg(1:scaling,1:scaling,:) = repmat(fill,[scaling, scaling, 1]);
    %bottom right corner
    fill =  (rgbimg(end,end-scaling,:)+rgbimg(end-scaling,end,:)+rgbimg(end-scaling,end-scaling,:))/3.;
    rgbimg(end-scaling+1:end,end-scaling+1:end,:) = repmat(fill, [scaling,scaling,1]);
end

