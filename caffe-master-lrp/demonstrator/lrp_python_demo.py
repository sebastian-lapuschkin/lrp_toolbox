import os
import numpy as np
from PIL import Image
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import copy
import sys

# temporarily append the directory of the pycaffe wrapper (lrp_toolbox/caffe-master-lrp/python) to the PYTHONPATH
sys.path.append('../python')
import caffe

from utils import load_model, cropped_imagenet_mean, transform_input, lrp_hm, normalize_color_hm, process_raw_heatmaps

# global variables to make this demo more convient to use
IMAGENET_MEAN_LOCATION  = '../python/caffe/imagenet/ilsvrc_2012_mean.npy'
EXAMPLE_IMAGE_FOLDER    = 'someimages'
MODEL                   = 'caffenet'                                        # options: ('googlenet' | 'caffenet')
                                                                            # - for caffenet, execute download_model.sh prior to this script
                                                                            # - for googlenet, a pre-trained model can be downloaded from http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

def main():
    simple_lrp_demo()

def simple_lrp_demo(num_images = 3):
    """
    Simple example to demonstrate the LRP methods using the Caffe python interface.
    Calculates the prediction and LRP heatmap for num_images of example imaages from the EXAMPLE_IMAGE_FOLDER
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
    cropped_mean = cropped_imagenet_mean(IMAGENET_MEAN_LOCATION, in_hei, in_wid)

    # load example iamge
    image_paths = [os.path.join(EXAMPLE_IMAGE_FOLDER, EXAMPLE_IMAGE_PATH) for EXAMPLE_IMAGE_PATH in os.listdir(EXAMPLE_IMAGE_FOLDER)[:num_images]]
    example_images = [Image.open(img_pth) for img_pth in image_paths]

    # preprocess image to fit caffe input convention (subtract mean, swap input dimensions (input blob convention is NxCxHxW), transpose color channels to BGR)
    transformed_input = np.array([transform_input(example_image, True, True, in_hei = in_hei, in_wid = in_wid, mean=cropped_mean)for example_image in example_images])

    # adapt caffe batchsize to avoid unnecessary computations
    net.blobs['data'].reshape(*transformed_input.shape)

    # classification (forward pass)
    # the lrp_hm convenience method always performs a forward pass anyways, the output here is only used to output the top predictions later
    net.blobs['data'].data[...] = transformed_input
    out = net.forward()
    top_predictions = np.argmax(out['prob'], axis=1)


    ## ############# ##
    # LRP parameters: #
    ## ############# ##
    lrp_type    = 'epsilon'         
    # lrp_type              | meaning of lrp_param  | uses switch_layer | description 
    # ---------------------------------------------------------------------------
    # epsilon               | epsilon               | no                | epsilon lrp
    # alphabeta             | beta                  | no                | alphabeta lrp, alpha = 1-beta
    # eps_n_flat            | epsilon               | yes               | epsilon lrp until switch_layer,   wflat lrp for all layers below
    # eps_n_wsquare         | epsilon               | yes               | epsilon lrp until switch_layer,   wsquare lrp for all layers below
    # ab_n_flat             | beta                  | yes               | alphabeta lrp until switch_layer, wflat lrp for all layers below
    # ab_n_wsquare          | beta                  | yes               | alphabeta lrp until switch_layer, wsquare lrp for all layers below
    # std_n_ab              | beta                  | yes               | standard lrp (epsilon with eps=0) until switch_layer, alphabeta lrp for all layers below
    # layer_dep             | -                     | no                | standard lrp (epsilon with eps=0) for all fully-connected layers, alphabeta lrp with alpha=1 for all convolution layerrs
    # layer_dep_n_flat      | -                     | yes               | layer_dep (see above) until switch_layer, wflat lrp for all layers below
    # layer_dep_n_wsquare   | -                     | yes               | layer_dep (see above) until switch-layer, wsquare lrp for all layers below

    lrp_param   =  0.000001         # (epsilon | beta      | epsilon    | epsilon      | beta    )
    classind    =  -1              # (class index  | -1 for top_class)

    # switch_layer param only needed for the composite methods
    # the parameter depicts the first layer for which the second formula type is used.
    # interesting values for caffenet are: 0, 4, 8, 10, 12 | 15, 18, 21 (convolution layers | innerproduct layers)
    switch_layer = 13


    ## ################################## ##
    # Heatmap calculation and presentation #
    ## ################################## ##

    # LRP
    backward = lrp_hm(net, transformed_input, lrp_method=lrp_type, lrp_param=lrp_param, target_class_inds=classind, switch_layer=switch_layer)

    sum_over_channels  = True
    normalize_heatmap  = False

    if lrp_type == 'deconv':
        sum_over_channels = False
        normalize_heatmap  = True

    # post-process the relevance values
    heatmaps = process_raw_heatmaps(backward, normalize=normalize_heatmap, sum_over_channels=sum_over_channels)

    for im_idx in range(num_images):
        
        if classind == -1:
            print('top class!')
            target_index = top_predictions[im_idx]
        else:
            target_index = classind

        # stretch input to input dimensions (only for visualization)
        stretched_input = transform_input(example_images[im_idx], False, False, in_hei = in_hei, in_wid = in_wid, mean=cropped_mean)
        heatmap = heatmaps[im_idx]

        # presentation
        plt.subplot(1,2,1)
        plt.title('Prediction: {}'.format(top_predictions[im_idx]))
        plt.imshow(stretched_input)
        plt.axis('off')

        # normalize heatmap for visualization
        max_abs = np.max(np.absolute(heatmap))
        norm = mpl.colors.Normalize(vmin = -max_abs, vmax = max_abs)

        plt.subplot(1,2,2)

        if lrp_type in ['epsilon', 'alphabeta', 'eps', 'ab']:
            plt.title('{}-LRP heatmap for class {}'.format(lrp_type, target_index))

        if lrp_type in ['eps_n_flat', 'eps_n_square', 'std_n_ab']:
            if lrp_type == 'eps_n_flat':
                first_method    = 'epsilon'
                second_method   = 'wflat'

            elif lrp_type == 'eps_n_square':
                first_method    = 'epsilon'
                second_method   = 'wsquare'

            elif lrp_type == 'std_n_ab':
                first_method    = 'epsilon'
                second_method   = 'alphabeta'

            plt.title('LRP heatmap for class {}\nstarting with {}\n {} from layer {} on.'.format(target_index, first_method, second_method, switch_layer))

        if sum_over_channels:
            # relevance values are averaged over the pixel channels, use a 1-channel colormap (seismic)
            plt.imshow(heatmap[...,0], cmap='seismic', norm=norm, interpolation='none')
        else:
            # 1 relevance value per color channel
            heatmap = normalize_color_hm(heatmap)
            plt.imshow(heatmap, interpolation = 'none')

        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
