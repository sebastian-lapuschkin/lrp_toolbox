'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.0
@copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
@license : BSD-2-Clause

provides methods to draw heatmaps beautifully.
'''

import numpy as np
from matplotlib.cm import ScalarMappable
from skimage import filter, io


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
        shape = [np.sqrt(V.size)]*2
    
    return np.reshape(V, shape)


def enlarge_image(img, scaling = 3):
    '''
    Enlarges a given input matrix by replicating each pixel value scaling times in horizontal and vertical direction.
    
    Parameters
    ----------
    
    img : numpy.ndarray
        array of shape [H x W]
        
    scaling : int
        positive integer value > 0
    
    Returns
    -------
    
    out : numpy.ndarray
        two-dimensional array of shape [scaling*H x scaling*W]     
    '''
    
    if scaling < 1 or not isinstance(scaling,int):
        print 'scaling factor needs to be an int >= 1'
        
    H,W = img.shape
    
    out = np.zeros((scaling*H, scaling*W))
    for h in range(H):
        fh = scaling*h
        for w in range(W):
            fw = scaling*w
            out[fh:fh+scaling, fw:fw+scaling] = img[h,w]
            
    return out


def repaint_corner_pixels(rgbimg, scaling = 3):
    '''
    Recolors the top left and bottom right pixel (groups) with the average rgb value of its three neighboring pixel (groups).
    The recoloring visually masks the opposing pixel values which are a product of stabilizing the scaling.
    Assumes those image ares will pretty much never show evidence.
    
    TODO: find a smarter way to do this. I know a smarter way, yet am too lazy to bother.
    
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
    
    sm = ScalarMappable(cmap = cmap)
    image = sm.to_rgba(enlarge_image(vec2im(X,shape), scaling))[:,:,0:3]
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
    
    sm = ScalarMappable(cmap = cmap)  #prepare heatmap -> rgb image conversion
    
    if normalize:
        R = R / np.max(np.abs(R))
    R[0,0] = 1; R[-1,-1] = -1; # anchors for controlled color mapping, to be drawn over later. 
    
    R = enlarge_image(vec2im(R,shape), scaling)
    rgbimg = sm.to_rgba(R)[:,:,0:3]
    rgbimg = repaint_corner_pixels(rgbimg, scaling)
    ''' TODO: shift heatmap values to [0,1] after normalizing, then use matplotlib.cm.<cmapname> directly. removes the need for repaint_corner_pixels. '''
    
    
    if not X is None: #compute the outline of the input     
        X = enlarge_image(vec2im(X,shape), scaling)
        xdims = X.shape
        Rdims = R.shape

        if not np.all(xdims == Rdims):
            print 'transformed heatmap and data dimension mismatch. data dimensions differ?'
            print 'R.shape = ',Rdims, 'X.shape = ', xdims
            print 'skipping drawing of outline\n'
        else:
            edges = filter.canny(X, sigma=sigma) 
            edges = np.invert(np.dstack([edges]*3))*1.0
            rgbimg *= edges # set outline pixels to black color
    
    return rgbimg


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
    for i in xrange(len(rgb_images)):
        if not sz:
            sz = rgb_images[i].shape
            image = rgb_images[i]
            gap = np.zeros((sz[0],gap,sz[2]))
            continue
        if not sz[0] == rgb_images[i].shape[0] and sz[1] == rgb_images[i].shape[2]:
            print 'image',i, 'differs in size. unable to perform horizontal alignment'
            print 'expected: Hx_xD = {0}x_x{1}'.format(sz[0],sz[1])
            print 'got     : Hx_xD = {0}x_x{1}'.format(rgb_images[i].shape[0],rgb_images[i].shape[1])
            print 'skipping image\n'
        else:
            image = np.hstack((image,gap,rgb_images[i]))
        
    print 'saving image to ', path
    io.imsave(path,image)
    return image 

            
            
        
