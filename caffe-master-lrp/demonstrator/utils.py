import numpy as np
import caffe
from PIL import Image


def normalize_color_hm(hm):
    norm_hm = hm /  np.max(np.absolute(hm))
    return norm_hm / 2. + 0.5


def split_into_batches(data, batch_size):

    """
    Split given batch of data into smaller batches of max batch_size data.

    Parameters
    ----------
    data:   numpy.ndarray of shape (N, ...)
            large batch of data
            first dimension needs to be the samples (data[i] gives i-th data point), a possible shape is (N, H, W, D) for image data

    batchs_size:
            int
            batch size needed

    Returns
    -------
    batches:    list[numpy.ndarray]
                a list of all batches of maximum batch size batch_size
    """


    N = data.shape[0]

    num_batches     = N // batch_size + int(N % batch_size != 0)

    batches = []

    for batch_idx in range(num_batches):
        batches.append(data[batch_idx * batch_size: np.min([N, (batch_idx + 1) * batch_size])])

    return batches

def get_target_indices(target_indices, predictions):
    '''
    check the format of the target_indices and translate negative indices to top prediction indices (-x is top x prediction)
    '''

    batch_size = predictions.shape[0]
    target_indices = np.array(target_indices, dtype='int32')

    # assure that target_indices is either a single value (used for all images) or a vector of length batch_size
    target_indices_vector_length = len(target_indices.shape)

    if target_indices_vector_length > 1:
        print('Error: target_indices vector needs to be either a single value or a 1d numpy array')

    elif target_indices_vector_length == 0:
        target_indices = target_indices[None]

    target_vec_dim = target_indices.shape[0]

    if  target_vec_dim == 1:
        target_indices = target_indices[0]
        target_indices = np.ones(batch_size, dtype='int32') * target_indices
        # print('single target class given -> use the same target class for all images in the batch')

    elif target_vec_dim == batch_size:
        # print('individual target class given for each image')
        pass

    else:
        print('ERROR: several target classes given but shape does not match the batch size.')
        return None

    pos_inds = target_indices >= 0
    neg_inds = target_indices <  0


    if np.any(neg_inds):
        top_preds  = np.argsort(predictions)

        target_vec = np.zeros(batch_size, dtype='int32')
        target_vec[pos_inds] = target_indices[pos_inds]
        target_vec[neg_inds] = top_preds[neg_inds, target_indices[neg_inds]]
        target_indices = target_vec

    return target_indices

def lrp_hm(net, input_images, lrp_method = 'epsilon', lrp_param = 0.0000001, target_class_inds = -1, switch_layer = -1, single_mode=False, verbose_output= False):

    input_shape = input_images.shape
    network_batch_shape = net.blobs['data'].data.shape

    if isinstance( target_class_inds, int):
        print('Using the same target class {} for all inputs in the batch.'.format(target_class_inds))
    else:
        assert(target_class_inds.shape[0] == input_images.shape[0])
        print('Individual classind given for each input')

    if single_mode:
        og_batch_size = int(network_batch_shape[0])
        net.blobs['data'].reshape(1, network_batch_shape[1], network_batch_shape[2], network_batch_shape[3])
        net.reshape()
        # print('Changed batchsize to {}'.format(1))

    batch_size = net.blobs['data'].shape[0]
    input_batches = split_into_batches(input_images, batch_size)
    output = []

    num_batches = len(input_batches)

    if not single_mode:
        print('...start batch processing with batchsize {}'.format(batch_size))

    for b_i, input_image_batch in enumerate(input_batches):

        if not single_mode and verbose_output:
            print('batch ({}/{})'.format(b_i+1, num_batches))

        # test prediction:
        original_shape = net.blobs['data'].shape

        # last batch can be smaller: adapt network input size for this batch
        tmp_reduced_batchsize = False
        if original_shape[0] != input_image_batch.shape[0]:
            net.blobs['data'].reshape(input_image_batch.shape[0], original_shape[1], original_shape[2], original_shape[3])
            net.reshape()
            tmp_reduced_batchsize = True
            # print('Changed batchsize to {}'.format(input_image_batch.shape[0]))

        net.blobs['data'].data[...] = input_image_batch
        out = net.forward()

        target_class = get_target_indices(target_class_inds, out['prob'])

        if single_mode:

            if switch_layer > 0 and lrp_method != 'epsilon' and lrp_method != 'alphabeta':
                lrpopts =  lrp_opts(lrp_method, lrp_param, switch_layer = switch_layer) 
                if lrpopts is None:
                    print('Invalid lrp parameter setting, check lrp_type and lrp_param')
                    return None
                relevance = net.lrp_single(int(target_class[0]), lrpopts)
            else:
                lrpopts =  lrp_opts(lrp_method, lrp_param) 
                if lrpopts is None:
                    print('Invalid lrp parameter setting, check lrp_type and lrp_param')
                    return None
                relevance = net.lrp_single(int(target_class[0]), lrpopts)

        else:

            if switch_layer > 0 and lrp_method != 'epsilon' and lrp_method != 'alphabeta':
                lrpopts =  lrp_opts(lrp_method, lrp_param, switch_layer = switch_layer) 
                if lrpopts is None:
                    print('Invalid lrp parameter setting, check lrp_type and lrp_param')
                    return None
                relevance = net.lrp(target_class, lrpopts)
            else:
                lrpopts =  lrp_opts(lrp_method, lrp_param) 
                if lrpopts is None:
                    print('Invalid lrp parameter setting, check lrp_type and lrp_param')
                    return None
                relevance = net.lrp(target_class, lrpopts)

        output.append(relevance)

        # revert network input change for the smaller batch
        if tmp_reduced_batchsize:
            net.blobs['data'].reshape(original_shape[0], original_shape[1], original_shape[2], original_shape[3])
            net.reshape()
            # print('Changed batchsize to {}'.format(original_shape[0]))

    output = np.concatenate(output)

    if single_mode:
        net.blobs['data'].reshape(og_batch_size, network_batch_shape[1], network_batch_shape[2], network_batch_shape[3])
        net.reshape()
        # print('Changed batchsize to {}'.format(og_batch_size))

    return output

def process_raw_heatmaps(rawhm_batch, normalize=False, sum_over_channels=True):
    """
    Process raw heatmap as outputted by the caffe network.
    Inverts channel swap to RGB and brings the heatmap back to the (num_images, height, width, channels) format

    Parameters
    ----------
    rawhm:      numpy.ndarray
                raw heatmap batch as outputted by the lrp_hm method

    normalize:  bool
                flag indicating whether to normalize the heatmap to the [-1, 1] interval (divide each pixel value by the highest absolute value)

    sum_over_channels: bool
                flag indicating whether to sum the heatmap values over the last image dimension (color channels)

    Returns
    -------
    heatmap:    numpy.ndarray
                processed heatmap batch
    """

    # n,h,w,c format
    rawhm_batch = rawhm_batch.transpose(0,2,3,1)
    heatmap = rawhm_batch

    if sum_over_channels:
        # color information not used, average over the channels dimension
        heatmap = heatmap.sum(3, keepdims=True)

    else:
        # color information important, invert caffe BGR channels wap to RGB
        heatmap = heatmap[:,:,:, [2,1,0]]

    if normalize:
        heatmap = heatmap / np.max(np.absolute(heatmap, axis=[1, 2, 3], keepdims=True), axis=[1, 2, 3], keepdims=True)

    return heatmap

def lrp_opts(method = 'epsilon', param = 0., switch_layer = -1):
    """
    Simple function to make some standard lrp varaints available more conveniently

    Parameters
    ----------
    method:         str
                    method type, either 'epsilon' or 'alphabeta', indicating the LRP method to use

    param:          float
                    method parameter value (epsilon for the epsilon method, beta for the alphabeta method)

    switch_layer:   int
                    layer in which to switch between lrp formulas (only used for some methods). The given int is the first layer for which the second formula is used

    Returns
    -------
    heatmap:    RelPropOpts object
                the RelPropOpts object that will be given to the lrp function to execute the desired lrp method
    """

    lrp_opts = caffe.RelPropOpts()

    if method == 'epsilon':
        lrp_opts.relpropformulatype = 0
        lrp_opts.epsstab            = param

    elif method == 'alphabeta':
        lrp_opts.relpropformulatype = 2
        lrp_opts.alphabeta_beta     = param

    elif method == 'eps_n_flat':
        lrp_opts.relpropformulatype = 54
        lrp_opts.epsstab             = param
        lrp_opts.auxiliaryvariable_maxlayerindexforflatdistinconv = switch_layer

    elif method == 'eps_n_wsquare':
        lrp_opts.relpropformulatype = 56
        lrp_opts.epsstab             = param
        lrp_opts.auxiliaryvariable_maxlayerindexforflatdistinconv = switch_layer

    elif method == 'ab_n_flat':
        lrp_opts.relpropformulatype = 58
        lrp_opts.alphabeta_beta     = param
        lrp_opts.auxiliaryvariable_maxlayerindexforflatdistinconv = switch_layer

    elif method == 'ab_n_wsquare':
        lrp_opts.relpropformulatype = 60
        lrp_opts.alphabeta_beta     = param
        lrp_opts.auxiliaryvariable_maxlayerindexforflatdistinconv = switch_layer

    elif method == 'eps_n_ab':

        if isinstance(param, tuple) and len(param) == 2:
            epsilon = param[0]
            beta    = param[1]
        else:
            epsilon = 1e-10
            beta    = 0.

        lrp_opts.alphabeta_beta     = beta
        lrp_opts.epsstab            = epsilon 
        lrp_opts.relpropformulatype = 114
        lrp_opts.auxiliaryvariable_maxlayerindexforflatdistinconv = switch_layer

    elif method == 'layer_dep':

        if isinstance(param, tuple) and len(param) == 2:
            epsilon = param[0]
            beta    = param[1]
        else:
            epsilon = 1e-10
            beta    = 0.

        lrp_opts.alphabeta_beta     = beta
        lrp_opts.epsstab            = epsilon 

        lrp_opts.relpropformulatype = 100
        lrp_opts.auxiliaryvariable_maxlayerindexforflatdistinconv = switch_layer

    elif method == 'layer_dep_n_flat':
        
        if isinstance(param, tuple) and len(param) == 2:
            epsilon = param[0]
            beta    = param[1]
        else:
            epsilon = 1e-10
            beta    = 0.

        lrp_opts.alphabeta_beta     = beta
        lrp_opts.epsstab            = epsilon 
        
        lrp_opts.relpropformulatype = 102
        lrp_opts.auxiliaryvariable_maxlayerindexforflatdistinconv = switch_layer

    elif method == 'layer_dep_n_wsquare':
        if isinstance(param, tuple) and len(param) == 2:
            epsilon = param[0]
            beta    = param[1]
        else:
            epsilon = 1e-10
            beta    = 0.

        lrp_opts.alphabeta_beta     = beta
        lrp_opts.epsstab            = epsilon 
        
        lrp_opts.relpropformulatype = 104
        lrp_opts.auxiliaryvariable_maxlayerindexforflatdistinconv = switch_layer

    elif method == 'deconv':
        lrp_opts.relpropformulatype = 26

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


def load_model(model = 'caffenet', batchsize=16):
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

    net.blobs['data'].reshape(batchsize,3,inwid,inhei)

    return net

def cropped_imagenet_mean(mean_file_path,inhei, inwid):
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

    mean_from_file = np.array(np.load(mean_file_path))

    # mean file is shape (3, 256, 256), however both googlenet and caffenet have smaller inputs -> crop the mean file
    if ((256 < inhei) or (256 < inwid)):
        print ('ERROR: Net Input size too large. Now using the pixel/colorchannel mean (scalar)')
        mean = np.mean(mean_from_file)
    else:
        w_offset = (256 - inwid) // 2
        h_offset = (256 - inhei) // 2

        print(w_offset, h_offset)

        mean = mean_from_file[:,w_offset:w_offset+inwid, h_offset:h_offset+inhei]


