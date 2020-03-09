
# coding: utf-8

# In[1]:


# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# mxnet imports
import mxnet as     mx
from   mxnet import nd
from   mxnet import autograd

from mxlrp import patch_lrp_gradient

import types
import matplotlib.pyplot as plt
import time

# mxnet imports
import mxnet as     mx
from   mxnet import nd
from   mxnet import autograd
from mxnet.gluon.nn.basic_layers import Activation

# standalone lrp imports
import numpy as np
na = np.newaxis
from mxmodules import Sequential, Linear, Rect, Tanh
import model_io, data_io


# # LRP using GLUON Hybridblocks
#
# ### Current functionality:
# - patch the hybrid_forward of dense layers with activation to enable lrp in the backward pass
# - normal run and hybridized run for fully-connected networks
#
# ### Current issues:
# - deferred initalization not supported (what about that?)
# - not working with float 64, therefore not exacltly comparable to numpy version
# - extracting and running the symbolic program doesn't work - misunderstood it?
#
# ### Other TODOS:
# - find better solution than block_grad method to change activation backward pass to identity

# ## Load Patched Functions

# In[2]:


from mxlrp import patch_lrp_gradient, translate_to_gluon


# ## Define Context

# In[3]:


ctx = mx.gpu()

if ctx == mx.cpu():
    dtype='float64'
else:
    dtype='float32'


# ## 1. Model Definition

# In[4]:


## ############## ##
# STANDALONE MODEL #
## ############## ##

long_rect_sta = model_io.read('../models/MNIST/long-rect.nn') # 99.17% prediction accuracy

# remove softmax layer for simplicity
# TODO: add softmax layer treatment and remove this part
long_rect_sta.modules = long_rect_sta.modules[:-1]


# In[5]:


## ######### ##
# GLUON MODEL #
## ######### ##
long_rect_gluon = translate_to_gluon(long_rect_sta, ctx, dtype)


# ## 2. Patch Gluon Gradient to LRP

# In[6]:


lrp_type = 'simple'
lrp_param= 0.
patch_lrp_gradient(long_rect_gluon, lrp_type, lrp_param)


# ## 3. Compare To Standalone Implementation

# In[7]:


## ######## ##
# LOAD MNIST #
## ######## ##
X = data_io.read('../data/MNIST/test_images.npy')
Y = data_io.read('../data/MNIST/test_labels.npy')

X = X.astype(np.float32)
Y = Y.astype(np.float32)

# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
X =  X / 127.5 - 1

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1


# In[ ]:


batch_size = 32
yselect    = 3
num_runs   = 10

yselect = (np.arange(Y.shape[1])[na,:] == yselect) * np.ones((batch_size, Y.shape[1]))
im = X[:batch_size]

im_mx      = nd.array(im,      ctx=ctx, dtype=dtype)
yselect_mx = nd.array(yselect, ctx=ctx, dtype=dtype)


# In[ ]:


## ######## ##
# STANDALONE #
## ######## ##
sta_times = []
for i in range(num_runs):
    strt = time.time()

    pred_sta   = long_rect_sta.forward(im)
    hm_sta     = long_rect_sta.lrp(yselect, lrp_var=lrp_type, param=lrp_param)

    stp = time.time()
    sta_times.append(stp-strt)
sta_times = np.array(sta_times)
time_sta = np.mean(sta_times)

print('...finished sta')

## ###### ##
# GLUON ND #
## ###### ##
im_mx.attach_grad()

gluon_times = []
for i in range(num_runs):
    strt = time.time()
    with autograd.record():
        pred_gluon = long_rect_gluon(im_mx)
    pred_gluon.backward(yselect_mx)

    hm_gluon = im_mx.grad

    hm_gluon   = hm_gluon.asnumpy()
    pred_gluon = pred_gluon.asnumpy()

    stp = time.time()
    gluon_times.append(stp-strt)
gluon_times = np.array(gluon_times)
time_gluon = np.mean(gluon_times)

print('...finished gluon non-hybrid')

## ############## ##
# GLUON HYBRIDIZED #
## ############## ##
long_rect_gluon.hybridize()

hg_times = []
for i in range(num_runs):
    strt = time.time()
    with autograd.record():
        pred_hg = long_rect_gluon(im_mx)
    pred_hg.backward(yselect_mx)

    hm_hg = im_mx.grad

    hm_hg   = hm_hg.asnumpy()
    pred_hg = pred_hg.asnumpy()

    stp = time.time()
    hg_times.append(stp-strt)
hg_times = np.array(hg_times)
time_hg = np.mean(hg_times)

print('...finished gluon     hybrid')


# ### BENCHMARKS:

# In[ ]:


preds_same = False
hms_same   = False

if np.allclose(pred_sta, pred_gluon) and np.allclose(pred_sta, pred_hg):
    print('Predictions identical between implementations!')
    preds_same = True

else:
    # numerically compare heatmaps, check how much difference float32 (np) / float 64 (mx) makes
    prec_pred = -1
    while np.allclose(pred_sta, pred_hg.squeeze(), atol=10**(-prec_pred)):
            prec_pred+=1

if np.allclose(hm_sta, hm_gluon) and np.allclose(hm_sta, hm_hg):
    print('Heatmaps identical between implementations!')
    hms_same = True

else:
    prec_lrp  = -1
    while np.allclose(hm_sta, hm_hg, atol=10**(-prec_lrp)):
            prec_lrp+=1


# In[ ]:


print('## ######## ##')
print('# EVALUATION #')
print('## ######## ##')

print('Comparison on Long-Rect network (4 fully-connected layers)\n')

print('1. TIME')

print('gluon running on {} averaged over {} runs'.format(ctx, num_runs))
print('')
print('Standalone: calculated predictions and heatmaps for {} images in {:4f} s'.format(batch_size, time_sta))
print('Gluon ND  : calculated predictions and heatmaps for {} images in {:4f} s'.format(batch_size, time_gluon))
print('Gluon HYB : calculated predictions and heatmaps for {} images in {:4f} s'.format(batch_size, time_hg))
print('')
print('Speedup Gluon (non-hybridized): {:2f} faster than standalone'.format(time_sta / time_gluon))
print('Speedup Gluon (    hybridized): {:2f} faster than standalone'.format(time_sta / time_hg))
print('')

print('2. RESULT')
if preds_same:
    print('Predictions identical between implementations!')
else:
    if prec_pred >= 0:
        print('Predictions are the same up to absolute tolerance 10e-{}'.format(prec_pred))
    else:
        print('ERROR: Standalone / Gluon predictions differ')

if hms_same:
    print('Heatmaps identical between implementations!')
else:
    if prec_lrp >= 0:
        print('Relevances  are the same up to absolute tolerance 10e-{}'.format(prec_lrp))
    else:
        print('ERROR: Standalone / Gluon lrp relevances differ')


# In[ ]:


num_images = 0

for i in range(num_images):

    print('Standalone pred:\n{}'.format(pred_sta[i]))
    print('Gluon      pred:\n{}'.format(pred_gluon[i]))

    fig = plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.title('Standalone')
    plt.imshow(hm_sta[i].reshape(28, 28), interpolation='none')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gluon')
    plt.imshow(hm_hg[i].reshape(28, 28), interpolation='none')
    plt.colorbar()
    plt.axis('off')

    plt.show()
