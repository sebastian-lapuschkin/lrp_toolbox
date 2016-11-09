function rgbimg = hm_to_rgb(R,X,scaling,shape,sigma,cmap,normalize)
    % @author: Sebastian Lapuschkin
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %rgbimg = hm_to_rgb(R,X,scaling,shape,sigma,cmap,normalize)
    %
    %Takes as input an intensity array and produces a rgb image for the represented heatmap.
    %optionally draws the outline of another input on top of it.
    %
    %Parameters
    %----------
    %
    %R : matrix shaped [M x N]
    %the heatmap to be visualized
    %
    %X : matrix shaped [M x N]
    %optional. usually the data point for which the heatmap R is for, which shall serve
    %as a template for a black outline to be drawn on top of the image
    %
    %scaling: int
    %factor, on how to enlarge the heatmap (to control resolution and as a inverse way to control outline thickness)
    %after reshaping it using shape.
    %
    %shape: [1 x 2]
    %optional. if not given, X is reshaped to be square.
    %
    %sigma : double
    %optional. sigma-parameter for the canny algorithm used for edge detection. the found edges are drawn as outlines.
    %
    %cmap : str
    %optional. color map of choice
    %
    %normalize : bool
    %optional. whether to normalize the heatmap to [-1 1] prior to colorization or not.
    %
    %Returns
    %-------
    %
    %rgbimg : matrix of shape [scaling*H x scaling*W x 3], where H*W == M*N

    if nargin < 7 || (exist('normalize','var') && isempty(normalize))
        normalize = true;
    end
    if nargin < 6 || (exist('cmap','var') && isempty(cmap))
       cmap = jet(255);
    end
    if nargin < 5 || (exist('sigma','var') && isempty(sigma))
        sigma = 2;
    end
    if nargin < 4 || (exist('shape','var') && isempty(shape))
        shape = [];
    end
    if nargin < 3 || (exist('scaling','var') && isempty(scaling))
        scaling = 3;
    end
    if nargin < 2 || (exist('X','var') && isempty(X))
        X = [];
    end

    if normalize
        R = R/max(abs(R(:)));
    end
    R(1) = 1 ; R(end) = -1;

    R = (R - min(R(:))); %we need this anyway for ind2rgb to work, so normlizing will be ignored.
    R = round((R ./ max(R(:)))*255);
    R(1,1) = 1; R(end,end) = 0;

    R = render.vec2im(R,shape);
    R = render.enlarge_image(R,scaling);
    rgbimg = ind2rgb(R,cmap);
    rgbimg = render.repaint_corner_pixels(rgbimg,scaling);

    if ~isempty(X) && any(exist('edge') == [2 6]) %draw outline of X on top of the image, if function edge from the image processing toolbox exists
        X = render.vec2im(X,shape);
        X = render.enlarge_image(X,scaling);

        xdims = size(X);
        rdims = size(R);
        if ~all(xdims == rdims)
            fprintf('transformed heatmap and data dimension mismatch. data dimensions differ?\n')
            fprintf('size(R) = %d %d size(X) = %d %d\n', rdims, xdims)
            fprintf('skipping drawing of outline\n\n')
        else
            %caution: 'edge' requires the image processing toolbox
            edges = repmat(~edge(X,'canny',[],sigma),[1,1,3])*1.;
            rgbimg = rgbimg .* edges; %set outline pixels in black color
        end
    end
end

