# short todo:
# write code which receives a path of raw heatmap scores (find out formatting on the fly)
# and a string descriptor for a heatmapping scheme (a name)
#
# loads the heatmap, converts it to a 2d array if necessary
# heatmaps the data (manually set limits ? zero-centered most probably. let's go with that by default.) according to colormap of choice.
# produces jpg or png heatmap file. image is not cached. use cached hmfile.
#
# caching (david!) : incorporate target class into cache name somehow.
#
# resolving heatmap name: 1) check custom heatmap table. each custom heatmap needs to be realized as afunction.
#
# if no custom heatmap can be found for given name: try to fall back using the matplotlib built-ins

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import imsave
import cv2

#this script expects command line parameters as
#
# python apply_heatmap.py rawheatmappath cmapname <outputpath>
#
# rawheatmappath it the path to a numeric raw heatmap. that may be plain text (alex,  ascii block matrix), mat, npy or npz
#
# cmapname is a string name describing the color mapping.
#
# outputpath is the output path of the produced rgb image INCLUDING file extension


def produce_supported_maps():
     #return a list of names and extreme color values.
    for map in custom_maps.keys() + matplotlib_maps:
        print map


def colorize_matplotlib(R,cmapname):
    #fetch color mapping function by string
    cmap = eval('cm.{0}'.format(cmapname))

    #bring data to [-1 1]
    R = R / np.max(np.abs(R))

    #push data to [0 1] to avoid automatic color map normalization
    R = (R + 1)/2

    H,W = R.shape

    return cmap(R.flatten())[:,0:3].reshape([H,W,3])



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


def hm_as_alpha_channel(R,inputimagepath,gamma):
    img = cv2.imread(inputimagepath)[...,::-1]/255.
    R *= R >= 0
    R = R / np.max(R)

    #add a bit of gamma correction to R, else we wont see much
    R = R**gamma

    #background = np.zeros_like(img) # black background
    return img*R[...,None]# + np.ones_like(img)*(1-R[...,None])




def load_alex_format(lines):
    D = int(lines[0].split(' ')[0])
    H,W = [int(v) for v in lines[1].split(' ') if len(v) > 0]
    R = np.zeros([H,W,D])

    OFFSET = 2
    for d in xrange(D):
        for h in xrange(H):
            row = [float(v) for v in lines[d*H + h + OFFSET].split(' ') if len(v) > 0]
            R[h,:,d] = np.array(row)

    return R

#methods and functions
def load_relevance_scores(heatmappath, aggregate='sum'):
    if heatmappath.endswith('.npy') or heatmappath.endswith('.npz'):
        R = np.load(heatmappath)
    elif heatmappath.endswith('.mat'):
        import scipy.io as matio
        R = matio.loadmat(heatmappath)
        # pick first "proper" variable
        for k in R.keys():
            if not k.startswith('__') and not k.startswith('MATLAB '):
                R = R[k]
                break
    elif heatmappath.endswith('.txt') or heatmappath.endswith('.alex'):
        with open(heatmappath, 'rb') as f:
            #heuristically check for alex' format
            content = f.read().split('\n')
            line1 = [c for c in content[0].split(' ') if len(c) > 0]
            line2 = [c for c in content[1].split(' ') if len(c) > 0]
            if len(line1) == 1 and line1[0] == '3' and len(line2) == 2: # we do probably have an alex-formatted file.
                R = load_alex_format(content)
            else: # assume block matrix
                R = np.loadtxt(heatmappath)


    #we now have loaded some heatmaps. those may ha(R):ve relevance values for eac color channe, e.g. be H x W x 3 dimensional.
    #we flatten by summing over the color channels.
    #FLATTENING CAN LATER BE DONE PER COLOR MAPPING: THIS MIGHT BE USED FOR ACTUALLY MAPPING STUFF
    if aggregate == 'sum':
        R = R.sum(axis=2)
    elif aggregate == 'l2':
        R = np.linalg.norm(R, ord=2, axis=2)
    elif aggregate == 'linf':
        R = np.linalg.norm(R, ord=np.Inf, axis=2)
    else:
        raise ValueError('Invalid aggregation mode ' + aggregate)
    return R

#list of supported color map names. the maps need to be implemented ABOVE this line because of PYTHON
custom_maps = {'gray-red':gregoire_gray_red,\
'gray-red2':gregoire_gray_red2,\
'black-green':gregoire_black_green,\
'black-firered':gregoire_black_firered,\
'blue-black-yellow':alex_black_yellow}

matplotlib_maps = ['afmhot','jet','seismic','bwr']

def apply_colormap(R,cmapname,rendermode = 'normal' ,inputimagepath = None , gamma = '0.5', gaussblur = 0):

    HM = 0

    if rendermode == 'normal' or rendermode == 'overlay':
        if cmapname in custom_maps:
            HM =  custom_maps[cmapname](R)
        elif cmapname in matplotlib_maps:
            HM =  colorize_matplotlib(R,cmapname)
        else:
            #produce_supported_maps()
            raise Exception('You have somehow managed to smuggle in the unsupported colormap {0} into method apply_colormap. Supported mappings above'.format(cmapname))

    elif rendermode == 'alpha':
        HM = hm_as_alpha_channel(R,inputimagepath,gamma)
    else:
        raise Exception('You have chosen an unsupported render mode. supported modes are: (1) normal , (2) alpha and (3) overlay')

    if rendermode == 'overlay':
        img = cv2.imread(inputimagepath)[...,::-1]/255.
        alpha = np.abs(R)[...,None]
        alpha = (alpha / np.max(alpha))**gamma
        HM = alpha*HM + (1-alpha)*img

    if gaussblur > 0: #there is a bug in opencv which causes an error with this command
        HM = cv2.GaussianBlur(HM,(gaussblur,gaussblur),0)

    return HM


def write_heatmap_image(RGB,outputpath):
    #RGB is still filled with floating point values.
    #convert, then save.
    RGB *= 255.
    RGB = RGB.astype(np.uint8)
    imsave(outputpath,RGB)


if __name__ == '__main__':
     if len(sys.argv) < 2:
        print 'USAGE: python apply_heatmap.py <pathtofile> <cmapname>'
        exit()
        
     if len(sys.argv) > 2:
        cmapname = sys.argv[2]
     else:
        cmapname = 'black-firered'

     if len(sys.argv) > 3:
        agg = sys.argv[3]
     else:
        agg = 'sum'
        
     inputpath = sys.argv[1]
        
     R = load_relevance_scores(inputpath, aggregate=agg)
     RGB = apply_colormap(R,cmapname)
     
     outputpath = inputpath + '_' + cmapname + '_' + 'heatmap.png'
     print 'writing result to', outputpath
     write_heatmap_image(RGB,outputpath)
     
     

