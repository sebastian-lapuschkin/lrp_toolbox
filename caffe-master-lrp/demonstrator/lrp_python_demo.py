import caffe
import numpy as np
import Image
import matplotlib           as mpl
import matplotlib.pyplot    as plt

# global variables to make this demo more convient to use
IMAGENET_MEAN_LOCATION  = '../python/caffe/imagenet/ilsvrc_2012_mean.npy'
EXAMPLE_IMAGE_PATH      = 'someimages/cat18.jpg'
MODEL                   = 'caffenet'                                        # options: ('googlenet' | 'caffenet')
                                                                            # - for caffenet, execute download_model.sh prior to this script
                                                                            # - for googlenet, a pre-trained model can be downloaded from http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

def main():
    simple_lrp_demo()

def simple_lrp_demo():
    """
    Simple example to demonstrate the LRP methods using the Caffe python interface.
    Calculates the prediction and epsilon LRP heatmap for an example image.
    """

    # load the pre-trained model
    net = load_model(model = MODEL)

    if MODEL == 'googlenet':
        in_hei = 224
        in_wid = 224
    else:
        # default: caffenet
        in_hei = 227
        in_wid = 227

    # load imagenet mean and crop it to fit the networks input dimensions
    cropped_mean = cropped_imagenet_mean(in_hei, in_wid)

    # load example iamge
    example_image = Image.open(EXAMPLE_IMAGE_PATH)

    # preprocess image to fit caffe input convention (subtract mean, swap input dimensions (input blob convention is NxCxHxW), transpose color channels to BGR)
    transformed_input = transform_input(example_image, True, True, in_hei = in_hei, in_wid = in_wid, mean=cropped_mean)

    # classification (forward pass)
    net.blobs['data'].data[...] = transformed_input[None, :]
    out = net.forward()
    top_class = np.argmax(out['prob'])

    lrp_type    = 'epsilon'
    lrp_param   =  0.00001
    classind = top_class

    # LRP
    backward = net.lrp(classind, lrp_opts(lrp_type, lrp_param))

    # post-process the relevance values
    heatmap = process_raw_heatmap(backward)

    # stretch input to input dimensions (only for visualization, same as preprocessing but without the channel swaps needed to pass it to the model)
    stretched_input = transform_input(example_image, False, False, in_hei = in_hei, in_wid = in_wid, mean=cropped_mean)

    # presentation
    plt.subplot(1,2,1)
    plt.title('Prediction: {}'.format(top_class))
    plt.imshow(stretched_input)
    plt.axis('off')

    # normalize heatmap for visualization
    max_abs = np.max(np.absolute(heatmap))
    norm = mpl.colors.Normalize(vmin = -max_abs, vmax = max_abs)

    plt.subplot(1,2,2)
    plt.title('{}-LRP heatmap for class {}'.format(lrp_type, classind))
    plt.imshow(heatmap, cmap='seismic', norm=norm)
    plt.axis('off')
    plt.show()


## ############### ##
# Helper Functions: #
## ############### ##

def process_raw_heatmap(rawhm, normalize=False):
    """
    Process raw heatmap as outputted by the caffe network.
    Inverts channel swap to RGB and brings the heatmap back to the (height, width, channels) format

    Parameters
    ----------
    rawhm:      numpy.ndarray
                raw heatmap as outputted by the lrp method

    normalize:  bool
                flag indicating whether to normalize the heatmap to the [-1, 1] interval (divide each pixel value by the highest absolute value)

    Returns
    -------
    heatmap:    numpy.ndarray
                processed heatmap
    """

    rawhm = rawhm[0].transpose(1,2,0)
    heatmap = rawhm.mean(2)

    if normalize:
        heatmap = heatmap / np.max(np.absolute(heatmap))

    return heatmap

def lrp_opts(method = 'epsilon', param = 0.):
    """
    Simple function to make standard lrp and epsilon lrp available more conveniently, something similar could as well be implemented directly in the python wrapper (pycaffe.py)

    Parameters
    ----------
    method:     str
                method type, either 'epsilon' or 'alphabeta', indicating the LRP method to use

    param:      float
                method parameter value (epsilon for the epsilon method, beta for the alphabeta method)
    Returns
    -------
    heatmap:    RelPropOpts object
                the RelPropOpts object that will be given to the lrp function
    """

    lrp_opts = caffe.RelPropOpts()

    if method == 'epsilon':
        lrp_opts.relpropformulatype = 0
        lrp_opts.epsstab            = param

    elif method == 'alphabeta':
        lrp_opts.relpropformulatype = 2
        lrp_opts.alphabeta_beta     = param

    else:
        print('unknown method name in lrp_opts helper function, currently only epsilon and alphabeta are supported')
        return None

    # standard values for the rest of the parameters, this function is for demonstration purposes only
    lrp_opts.numclasses = 1000
    lrp_opts.lastlayerindex = -2
    lrp_opts.firstlayerindex = 0
    lrp_opts.codeexectype = 0
    lrp_opts.lrn_forward_type = 0
    lrp_opts.lrn_backward_type = 1
    lrp_opts.maxpoolingtoavgpoolinginbackwardpass = 0
    lrp_opts.biastreatmenttype = 0

    return lrp_opts

def transform_input(input_image, transpose_input_dimensions = False, channel_swap = False, in_hei = 227, in_wid = 227, mean = None):
    """
    Preprocessing function to input an image to a caffe model.
    Resizes the image to the input height and width of the model, transposes common (H,W,D) shape to (D,H,W) and swaps the collor channels from RGB to BGR

    Parameters
    ----------
    input_image:                numpy.ndarray
                                input image

    transpose_input_dimensions: bool
                                flag whether to change (H,W,D) to (D,H,W)

    channel_swap:               bool
                                flag whether to swap input channels from RGB to BGR

    in_hei:                     int
                                input height of the network

    in_wid:                     int
                                input width of the network

    mean:                       numpy.ndarray
                                dataset mean to substract, set to None if not desired

    Returns
    -------
    input_image:                preprocessed image
    """

    input_image = input_image.resize((in_hei, in_wid), Image.ANTIALIAS)

    input_image = np.array(input_image)
    print 'shape:',  input_image.shape

    if transpose_input_dimensions:
        # suppose data in (H,W,D) format, change to (D,H,W) for caffe
        input_image = input_image.transpose(2,0,1)

    if channel_swap:
        # suppose rgb input, change to bgr
        input_image = input_image[[2,1,0],:,:]

    if mean is not None:
        # subtract dataset mean
        input_image = input_image - mean

    return input_image


def load_model(model = 'caffenet'):
    """
    Load pre-trained model from file

    Parameters
    ----------
    model:  str
            name of the model to load, currently supported are 'caffenet' and 'googlenet'

    Returns
    -------
    net:    net
            pycaffe net object
    """

    if model == 'googlenet':
        inwid = 224
        inhei = 224
    else:# ('bvlc_reference_caffenet' - default)
        inwid = 227
        inhei = 227


    if model == 'googlenet':
        net = caffe.Net('../models/bvlc_googlenet/deploy.prototxt',
                        '../models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                        caffe.TEST)
    else:# ('bvlc_reference_caffenet' - default)
        net = caffe.Net('../models/bvlc_reference_caffenet/deploy.prototxt',
                        '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                        caffe.TEST)

    net.blobs['data'].reshape(1,3,inwid,inhei)

    return net

def cropped_imagenet_mean(inhei, inwid):
    """
    Loads mean from file and crops it to fit given input dimensions

    Parameters
    ----------
    inhei:  height of the cropped mean
    inwid:  width  of the cropped mean

    Returns
    -------
    mean:   cropped mean file
    """

    mean_from_file = np.array(np.load(IMAGENET_MEAN_LOCATION))

    # mean file is shape (3, 256, 256), however both googlenet and caffenet have smaller inputs -> crop the mean file
    if ((256 < inhei) or (256 < inwid)):
        print 'ERROR: Net Input size too large. Now using the pixel/colorchannel mean (scalar)'
        mean = np.mean(mean_from_file)
    else:
        w_offset = (256 - inwid) / 2
        h_offset = (256 - inhei) / 2
        mean = mean_from_file[:,w_offset:w_offset+inwid, h_offset:h_offset+inhei]



if __name__ == '__main__':
    main()
