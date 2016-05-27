function image = digit_to_rgb(X,scaling,shape,cmap)
    % @author: Sebastian Bach
    % @maintainer: Sebastian Bach
    % @contact: sebastian.bach@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Bach, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %image = digit_to_rgb(X,scaling,shape,cmap)
    %
    %Takes as input an intensity array and produces a rgb image due to some color map
    %     
    %Parameters
    %----------
    %     
    %X : intensity matrix of shape [M x N]
    %         
    %scaling : int
    %optional. positive integer value > 0
    %         
    %shape: tuple or list of its output shape , length = 2
    %optional. if not given, X is reshaped to be square.
    %         
    %cmap : str
    %name of color map of choice. default is 1-gray(255)
    %which is the equivalent to the 'binary' color map as
    %available with python'S matplotlib
    %         
    %Returns
    %-------
    %     
    %image : three-dimensional matrix of shape [scaling*H x scaling*W x 3] , where H*W == M*N

    if nargin < 4 || ( exist('cmap','var') && isempty(cmap))
       cmap = 1-gray(255) ;
    end
    if nargin < 3 || ( exist('shape','var') && isempty(shape))
        shape = [];
    end
    if nargin < 2 || ( exist('scaling','var') && isempty(scaling))
       scaling = 3; 
    end
    
    X = (X - min(X(:)));
    X = round((X ./ max(X(:)))*255);
    
    X = render.vec2im(X,shape);
    X = render.enlarge_image(X,scaling);
    image = ind2rgb(X,cmap);
end

