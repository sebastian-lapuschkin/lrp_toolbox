function image = save_image(rgb_images, path, gap)
    % @author: Sebastian Lapuschkin
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.2+
    % @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %save_image(rgb_images, path, gap)
    %
    %Takes as input a list of rgb images, places them next to each other with a gap and writes out the result.
    %
    %Parameters
    %----------
    %
    %rgb_images : cell array of three-dimensional rgb images
    %each item in the collection is expected to be an rgb image of dimensions [H x _ x 3]
    %where the width is variable
    %
    %path : str
    %the output path of the assembled image
    %
    %gap : int
    %optional. sets the width of a black area of pixels realized as an image shaped [H x gap x 3] in between the input images
    %
    %Returns
    %-------
    %
    %image : rgb image
    %the assembled image as written out to path

    if nargin < 3 || (exists('gap','var') && isempty(gap))
        gap = 2;
    end

    sz = [];
    image = [];
    for i = 1:length(rgb_images)
       if isempty(sz)
          sz = size(rgb_images{i});
          image = rgb_images{i};
          gap = zeros(sz(1),gap,sz(3));
          continue;
       end
       isz = size(rgb_images{i});
       if ~ all(sz([1,3]) == isz([1,3]))
          fprintf('image %i differs in size. unable to perform horizontal alignment\n',i)
          fprintf('expected: Hx_xD = %i_x_%i\n',sz([1,3]))
          fprintf('got: Hx_xD = %i_x_%i\n',isz([1,3]))
          fprintf('skipping image\n\n')
       else
           image = horzcat(image, gap, rgb_images{i});
       end
    end

    fprintf('saving image to %s\n\n',path)
    imwrite(image, path)


end

