'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 08.11.2017
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause

compute execution times for different lrp computation variants, from naive to optimized.
'''

import time
import matplotlib.pyplot as plt
import numpy as np ; na = np.newaxis

import model_io
import data_io
import render


#load a neural network, as well as the MNIST test data and some labels
nn = model_io.read('../models/MNIST/long-rect.nn') # 99.17% prediction accuracy
X = data_io.read('../data/MNIST/test_images.npy')
Y = data_io.read('../data/MNIST/test_labels.npy')

# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
X =  X / 127.5 - 1

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1

#permute data order for demonstration. or not. your choice.
I = np.arange(X.shape[0])


# do some benchmarking.

#for B in [1, 16, 64]: #batch sizes for work laptop with only 8GB RAM
for B in [1, 16, 64, 256]:

    forward_times_old = []
    forward_times_new = []
    forward_times_aware = []
    lrp_times_old = []
    lrp_times_old2 = []
    lrp_times_new = []
    lrp_times_new2 = []
    lrp_times_aware = []
    lrp_times_aware2 = []


    print '#####################################'
    print 'Measuring Speed Gain for batch of {}'.format(B)
    print '#####################################'
    for i in xrange(10):
        x = X[:B,:]
        # old code
        t_start = time.time();  yold = nn.forward(x);                               forward_times_old.append(time.time() - t_start)
        t_start = time.time();  Rold = nn.lrp(yold, 'simple_slow');                 lrp_times_old.append(time.time() - t_start)
        t_start = time.time();  REold = nn.lrp(yold, 'epsilon_slow', 0.01);         lrp_times_old2.append(time.time() - t_start)
        # newer lrp code
        t_start = time.time();  ynew = nn.forward(x);                               forward_times_new.append(time.time() - t_start)
        t_start = time.time();  Rnew = nn.lrp(ynew, 'simple');                      lrp_times_new.append(time.time() - t_start)
        t_start = time.time();  REnew = nn.lrp(ynew, 'epsilon', 0.01);              lrp_times_new2.append(time.time() - t_start)
        # lrp aware code and forward pass
        t_start = time.time();  yaw = nn.forward(x, lrp_aware=True);                forward_times_aware.append(time.time() - t_start)
        t_start = time.time();  Raw = nn.lrp(yaw, 'simple');                        lrp_times_aware.append(time.time() - t_start)
        t_start = time.time();  REaw= nn.lrp(yaw, 'epsilon', 0.01);                 lrp_times_aware2.append(time.time() - t_start)

        np.testing.assert_allclose(yold, ynew, rtol=1e-8) # predictions
        np.testing.assert_allclose(yold, yaw, rtol=1e-8)

        np.testing.assert_allclose(Rold, Rnew, rtol=1e-8) # simple lrp maps
        np.testing.assert_allclose(Rold, Raw, rtol=1e-8)

        np.testing.assert_allclose(REold, REnew, rtol=1e-8) # eps lrp maps
        np.testing.assert_allclose(REold, REaw, rtol=1e-8)

        print '.'

    print '    Mean Forward pass times:'
    print '      old:  ', np.mean(forward_times_old), '({}% speedup vs old)'.format(int(100*(1 - np.mean(forward_times_old)/np.mean(forward_times_old))))
    print '      new:  ', np.mean(forward_times_new), '({}% speedup vs old)'.format(int(100*(1 - np.mean(forward_times_new)/np.mean(forward_times_old))))
    print '      aware:', np.mean(forward_times_aware), '({}% speedup vs old)'.format(int(100*(1 - np.mean(forward_times_aware)/np.mean(forward_times_old))))
    print '    Mean LRP times 1 (simple lrp):'
    print '      old:  ', np.mean(lrp_times_old), '({}% speedup vs old)'.format(int(100*(1 - np.mean(lrp_times_old)/np.mean(lrp_times_old))))
    print '      new:  ', np.mean(lrp_times_new), '({}% speedup vs old)'.format(int(100*(1 - np.mean(lrp_times_new)/np.mean(lrp_times_old))))
    print '      aware:', np.mean(lrp_times_aware), '({}% speedup vs old)'.format(int(100*(1 - np.mean(lrp_times_aware)/np.mean(lrp_times_old))))
    print '    Mean LRP times 2 (epsilon lrp):'
    print '      old:  ', np.mean(lrp_times_old2), '({}% speedup vs old)'.format(int(100*(1 - np.mean(lrp_times_old2)/np.mean(lrp_times_old2))))
    print '      new:  ', np.mean(lrp_times_new2), '({}% speedup vs old)'.format(int(100*(1 - np.mean(lrp_times_new2)/np.mean(lrp_times_old2))))
    print '      aware:', np.mean(lrp_times_aware2), '({}% speedup vs old)'.format(int(100*(1 - np.mean(lrp_times_aware2)/np.mean(lrp_times_old2))))
    print '    Mean Total times with LRP once:'
    oldtotal = np.mean(np.array(lrp_times_old) + np.array(forward_times_old))
    newtotal = np.mean(np.array(lrp_times_new) + np.array(forward_times_new))
    awaretotal = np.mean(np.array(lrp_times_aware) + np.array(forward_times_aware))
    print '      old:  ', oldtotal, '({}% speedup vs old)'.format(int(100*(1 - oldtotal/oldtotal)))
    print '      new:  ', newtotal, '({}% speedup vs old)'.format(int(100*(1 - newtotal/oldtotal)))
    print '      aware:', awaretotal, '({}% speedup vs old)'.format(int(100*(1 - awaretotal/oldtotal)))
    print '    Mean Total times with LRP twice:'
    oldtotaltwice = np.mean(np.array(lrp_times_old) + np.array(lrp_times_old2) + np.array(forward_times_old))
    newtotaltwice = np.mean(np.array(lrp_times_new) + np.array(lrp_times_new2) + np.array(forward_times_new))
    awaretotaltwice = np.mean(np.array(lrp_times_aware) + np.array(lrp_times_aware2) + np.array(forward_times_aware))
    print '      old:  ', oldtotaltwice, '({}% speedup vs old)'.format(int(100*(1 - oldtotaltwice/oldtotaltwice)))
    print '      new:  ', newtotaltwice, '({}% speedup vs old)'.format(int(100*(1 - newtotaltwice/oldtotaltwice)))
    print '      aware:', awaretotaltwice, '({}% speedup vs old)'.format(int(100*(1 - awaretotaltwice/oldtotaltwice)))
    print ''
    print ''

