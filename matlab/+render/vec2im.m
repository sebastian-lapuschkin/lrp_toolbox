function im = vec2im( V, shape )
    % @author: Sebastian Lapuschkin
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
    % @license : BSD-2-Clause
    %
    %im = vec2im( V, shape )
    %
    %Transform an array V into a specified shape - or if no shape is given assume a square output format.
    %
    %Parameters
    %----------
    %
    %V : some vector or matrix sized [M x N]
    %an array either representing a matrix or vector to be reshaped into an two-dimensional image
    %
    %shape : [1 x 2]
    %optional. containing the shape information for the output array if not given, the output is assumed to be square
    %
    %Returns
    %-------
    %
    %W : reshaped vector or matrix
    %with size(W) = shape or size(W,i) = sqrt(size(V)) for i = 1,2


    if nargin < 2 || (exist('shape','var') && isempty(shape))
        shape = repmat(sqrt(numel(V)),[1,2]);
    end

    im = reshape(V,shape)';
end

