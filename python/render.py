'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause

provides methods to draw heatmaps beautifully.
'''

import numpy as np
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import skimage.io
try:
    from skimage.feature import canny
except:
    from skimage.filter import canny


def vec2im(V, shape = () ):
    '''
    Transform an array V into a specified shape - or if no shape is given assume a square output format.

    Parameters
    ----------

    V : numpy.ndarray
        an array either representing a matrix or vector to be reshaped into an two-dimensional image

    shape : tuple or list
        optional. containing the shape information for the output array if not given, the output is assumed to be square

    Returns
    -------

    W : numpy.ndarray
        with W.shape = shape or W.shape = [np.sqrt(V.size)]*2

    '''

    if len(shape) < 2:
        shape = [int(np.sqrt(V.size))]*2

    return np.reshape(V, shape)


def enlarge_image(img, scaling = 3):
    '''
    Enlarges a given input matrix by replicating each pixel value scaling times in horizontal and vertical direction.

    Parameters
    ----------

    img : numpy.ndarray
        array of shape [H x W] OR [H x W x D]

    scaling : int
        positive integer value > 0

    Returns
    -------

    out : numpy.ndarray
        two-dimensional array of shape [scaling*H x scaling*W]
        OR
        three-dimensional array of shape [scaling*H x scaling*W x D]
        depending on the dimensionality of the input
    '''

    if scaling < 1 or not isinstance(scaling,int):
        print('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H,W = img.shape

        out = np.zeros((scaling*H, scaling*W))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling] = img[h,w]

    elif len(img.shape) == 3:
        H,W,D = img.shape

        out = np.zeros((scaling*H, scaling*W,D))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling,:] = img[h,w,:]

    return out


def repaint_corner_pixels(rgbimg, scaling = 3):
    '''
    DEPRECATED/OBSOLETE.

    Recolors the top left and bottom right pixel (groups) with the average rgb value of its three neighboring pixel (groups).
    The recoloring visually masks the opposing pixel values which are a product of stabilizing the scaling.
    Assumes those image ares will pretty much never show evidence.

    Parameters
    ----------

    rgbimg : numpy.ndarray
        array of shape [H x W x 3]

    scaling : int
        positive integer value > 0

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3]
    '''


    #top left corner.
    rgbimg[0:scaling,0:scaling,:] = (rgbimg[0,scaling,:] + rgbimg[scaling,0,:] + rgbimg[scaling, scaling,:])/3.0
    #bottom right corner
    rgbimg[-scaling:,-scaling:,:] = (rgbimg[-1,-1-scaling, :] + rgbimg[-1-scaling, -1, :] + rgbimg[-1-scaling,-1-scaling,:])/3.0
    return rgbimg


def digit_to_rgb(X, scaling=3, shape = (), cmap = 'binary'):
    '''
    Takes as input an intensity array and produces a rgb image due to some color map

    Parameters
    ----------

    X : numpy.ndarray
        intensity matrix as array of shape [M x N]

    scaling : int
        optional. positive integer value > 0

    shape: tuple or list of its , length = 2
        optional. if not given, X is reshaped to be square.

    cmap : str
        name of color map of choice. default is 'binary'

    Returns
    -------

    image : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    #create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    image = enlarge_image(vec2im(X,shape), scaling) #enlarge
    image = cmap(image.flatten())[...,0:3].reshape([image.shape[0],image.shape[1],3]) #colorize, reshape

    return image



def hm_to_rgb(R, X = None, scaling = 3, shape = (), sigma = 2, cmap = 'jet', normalize = True):
    '''
    Takes as input an intensity array and produces a rgb image for the represented heatmap.
    optionally draws the outline of another input on top of it.

    Parameters
    ----------

    R : numpy.ndarray
        the heatmap to be visualized, shaped [M x N]

    X : numpy.ndarray
        optional. some input, usually the data point for which the heatmap R is for, which shall serve
        as a template for a black outline to be drawn on top of the image
        shaped [M x N]

    scaling: int
        factor, on how to enlarge the heatmap (to control resolution and as a inverse way to control outline thickness)
        after reshaping it using shape.

    shape: tuple or list, length = 2
        optional. if not given, X is reshaped to be square.

    sigma : double
        optional. sigma-parameter for the canny algorithm used for edge detection. the found edges are drawn as outlines.

    cmap : str
        optional. color map of choice

    normalize : bool
        optional. whether to normalize the heatmap to [-1 1] prior to colorization or not.

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    R = enlarge_image(vec2im(R,shape), scaling)

    if cmap in custom_maps:
        rgb =  custom_maps[cmap](R)
    else:
        if normalize:
            R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
            R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping

        #create color map object from name string
        cmap = eval('matplotlib.cm.{}'.format(cmap))

        # apply colormap
        rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    #rgb = repaint_corner_pixels(rgb, scaling) #obsolete due to directly calling the color map with [0,1]-normalized inputs

    if not X is None: #compute the outline of the input
        X = enlarge_image(vec2im(X,shape), scaling)
        xdims = X.shape
        Rdims = R.shape

        if not np.all(xdims == Rdims):
            print('transformed heatmap and data dimension mismatch. data dimensions differ?')
            print('R.shape = ',Rdims, 'X.shape = ', xdims)
            print('skipping drawing of outline\n')
        else:
            edges = canny(X, sigma=sigma)
            edges = np.invert(np.dstack([edges]*3))*1.0
            rgb *= edges # set outline pixels to black color

    return rgb


def save_image(rgb_images, path, gap = 2):
    '''
    Takes as input a list of rgb images, places them next to each other with a gap and writes out the result.

    Parameters
    ----------

    rgb_images : list , tuple, collection. such stuff
        each item in the collection is expected to be an rgb image of dimensions [H x _ x 3]
        where the width is variable

    path : str
        the output path of the assembled image

    gap : int
        optional. sets the width of a black area of pixels realized as an image shaped [H x gap x 3] in between the input images

    Returns
    -------

    image : numpy.ndarray
        the assembled image as written out to path
    '''

    sz = []
    image = []
    for i in range(len(rgb_images)):
        if not sz:
            sz = rgb_images[i].shape
            image = rgb_images[i]
            gap = np.zeros((sz[0],gap,sz[2]))
            continue
        if not sz[0] == rgb_images[i].shape[0] and sz[1] == rgb_images[i].shape[2]:
            print('image',i, 'differs in size. unable to perform horizontal alignment')
            print('expected: Hx_xD = {0}x_x{1}'.format(sz[0],sz[1]))
            print('got     : Hx_xD = {0}x_x{1}'.format(rgb_images[i].shape[0],rgb_images[i].shape[1]))
            print('skipping image\n')
        else:
            image = np.hstack((image,gap,rgb_images[i]))

    image *= 255
    image = image.astype(np.uint8)

    print('saving image to ', path)
    skimage.io.imsave(path,image)
    return image


# ################## #
# custom color maps: #
# ################## #

def gregoire_gray_red(R):
    basegray = 0.8 #floating point gray

    maxabs = np.max(R)
    RGB = np.ones([R.shape[0], R.shape[1],3]) * basegray #uniform gray image.

    tvals = np.maximum(np.minimum(R/maxabs,1.0),-1.0)
    negatives = R < 0

    RGB[negatives,0] += tvals[negatives]*basegray
    RGB[negatives,1] += tvals[negatives]*basegray
    RGB[negatives,2] += -tvals[negatives]*(1-basegray)

    positives = R>=0
    RGB[positives,0] += tvals[positives]*(1-basegray)
    RGB[positives,1] += -tvals[positives]*basegray
    RGB[positives,2] += -tvals[positives]*basegray

    return RGB


def gregoire_black_green(R):
    maxabs = np.max(R)
    RGB = np.zeros([R.shape[0], R.shape[1],3])

    negatives = R<0
    RGB[negatives,2] = -R[negatives]/maxabs

    positives = R>=0
    RGB[positives,1] = R[positives]/maxabs

    return RGB


def gregoire_black_firered(R):
    R = R / np.max(np.abs(R))
    x = R

    hrp  = np.clip(x-0.00,0,0.25)/0.25
    hgp = np.clip(x-0.25,0,0.25)/0.25
    hbp = np.clip(x-0.50,0,0.50)/0.50

    hbn = np.clip(-x-0.00,0,0.25)/0.25
    hgn = np.clip(-x-0.25,0,0.25)/0.25
    hrn = np.clip(-x-0.50,0,0.50)/0.50

    return np.concatenate([(hrp+hrn)[...,None],(hgp+hgn)[...,None],(hbp+hbn)[...,None]],axis = 2)


def gregoire_gray_red2(R):
    v = np.var(R)
    R[R > 10*v] = 0
    R[R<0] = 0
    R = R / np.max(R)
    #(this is copypasta)
    x=R

    # positive relevance
    hrp = 0.9 - np.clip(x-0.3,0,0.7)/0.7*0.5
    hgp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4
    hbp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4

    # negative relevance
    hrn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
    hgn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
    hbn = 0.9 - np.clip(-x-0.3,0,0.7)/0.7*0.5

    hr = hrp*(x>=0)+hrn*(x<0)
    hg = hgp*(x>=0)+hgn*(x<0)
    hb = hbp*(x>=0)+hbn*(x<0)


    return np.concatenate([hr[...,None],hg[...,None],hb[...,None]],axis=2)



def alex_black_yellow(R):

    maxabs = np.max(R)
    RGB = np.zeros([R.shape[0], R.shape[1],3])

    negatives = R<0
    RGB[negatives,2] = -R[negatives]/maxabs

    positives = R>=0
    RGB[positives,0] = R[positives]/maxabs
    RGB[positives,1] = R[positives]/maxabs

    return RGB


#list of supported color map names. the maps need to be implemented ABOVE this line because of PYTHON
custom_maps = {'gray-red':gregoire_gray_red,\
'gray-red2':gregoire_gray_red2,\
'black-green':gregoire_black_green,\
'black-firered':gregoire_black_firered,\
'blue-black-yellow':alex_black_yellow}
