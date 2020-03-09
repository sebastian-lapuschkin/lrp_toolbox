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
import importlib.util as imp
import numpy
import numpy as np
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
na = np.newaxis

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
def benchmark():
    for B in [1 ,16, 64]: #batch sizes for work laptop with only 8GB RAM
    #for B in [1, 16, 64, 256]:

        forward_times_old = []
        forward_times_new = []
        forward_times_aware = []

        lrp_times_old = []  #simple
        lrp_times_old2 = [] #epsilon
        lrp_times_old3 = [] #alpha 2, beta -1
        lrp_times_old4 = [] #alpha 1, beta 0
        lrp_times_old5 = [] #alpha 0, beta 1

        lrp_times_new = []  #s
        lrp_times_new2 = [] #e
        lrp_times_new3 = [] #a2b-1
        lrp_times_new4 = [] #a1b0
        lrp_times_new5 = [] #a0b1

        lrp_times_aware = []  #s
        lrp_times_aware2 = [] #e
        lrp_times_aware3 = [] #a2b-1
        lrp_times_aware4 = [] #a1b0
        lrp_times_aware5 = [] #a0b1


        print('#####################################')
        print('Measuring Speed Gain for batch of {} on FCNN using {}'.format(B, np.__name__))
        print('#####################################')
        for i in range(10):
            x = X[:B,:]
            # old code
            t_start = time.time();  yold = nn.forward(x);                               forward_times_old.append(time.time() - t_start)
            t_start = time.time();  Rold = nn.lrp(yold, 'simple_slow');                 lrp_times_old.append(time.time() - t_start)
            t_start = time.time();  REold = nn.lrp(yold, 'epsilon_slow', 0.01);         lrp_times_old2.append(time.time() - t_start)
            t_start = time.time();  RAold2 = nn.lrp(yold, 'alphabeta_slow', 2.);          lrp_times_old3.append(time.time() - t_start)
            t_start = time.time();  RAold1 = nn.lrp(yold, 'alphabeta_slow', 1.);          lrp_times_old4.append(time.time() - t_start)
            t_start = time.time();  RAold0 = nn.lrp(yold, 'alphabeta_slow', 0.);          lrp_times_old5.append(time.time() - t_start)
            # newer lrp code
            t_start = time.time();  ynew = nn.forward(x);                               forward_times_new.append(time.time() - t_start)
            t_start = time.time();  Rnew = nn.lrp(ynew, 'simple');                      lrp_times_new.append(time.time() - t_start)
            t_start = time.time();  REnew = nn.lrp(ynew, 'epsilon', 0.01);              lrp_times_new2.append(time.time() - t_start)
            t_start = time.time();  RAnew2 = nn.lrp(ynew, 'alphabeta', 2.);              lrp_times_new3.append(time.time() - t_start)
            t_start = time.time();  RAnew1 = nn.lrp(ynew, 'alphabeta', 1.);              lrp_times_new4.append(time.time() - t_start)
            t_start = time.time();  RAnew0 = nn.lrp(ynew, 'alphabeta', 0.);              lrp_times_new5.append(time.time() - t_start)

            # lrp aware code and forward pass
            t_start = time.time();  yaw = nn.forward(x, lrp_aware=True);                forward_times_aware.append(time.time() - t_start)
            t_start = time.time();  Raw = nn.lrp(yaw, 'simple');                        lrp_times_aware.append(time.time() - t_start)
            t_start = time.time();  REaw= nn.lrp(yaw, 'epsilon', 0.01);                 lrp_times_aware2.append(time.time() - t_start)
            t_start = time.time();  RAaw2= nn.lrp(yaw, 'alphabeta', 2.);                 lrp_times_aware3.append(time.time() - t_start)
            t_start = time.time();  RAaw1= nn.lrp(yaw, 'alphabeta', 1.);                 lrp_times_aware4.append(time.time() - t_start)
            t_start = time.time();  RAaw0= nn.lrp(yaw, 'alphabeta', 0.);                 lrp_times_aware5.append(time.time() - t_start)

            tolerance = 1e-8
            np.testing.assert_allclose(yold, ynew, rtol=tolerance) # predictions
            np.testing.assert_allclose(yold, yaw, rtol=tolerance)

            np.testing.assert_allclose(Rold, Rnew, rtol=tolerance) # simple lrp maps
            np.testing.assert_allclose(Rold, Raw, rtol=tolerance)

            np.testing.assert_allclose(REold, REnew, rtol=tolerance) # eps lrp maps
            np.testing.assert_allclose(REold, REaw, rtol=tolerance)

            np.testing.assert_allclose(RAold2, RAnew2, rtol=tolerance) # alpha2 lrp maps
            np.testing.assert_allclose(RAold2, RAaw2, rtol=tolerance)

            np.testing.assert_allclose(RAold1, RAnew1, rtol=tolerance) # alpha1 lrp maps
            np.testing.assert_allclose(RAold1, RAaw1, rtol=tolerance)

            np.testing.assert_allclose(RAold0, RAnew0, rtol=tolerance) # alpha0 lrp maps
            np.testing.assert_allclose(RAold0, RAaw0, rtol=tolerance)

            print('.',end='')

        print()
        print('    Mean Forward pass times:')
        print('      old:  ', numpy.mean(forward_times_old), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(forward_times_old)/numpy.mean(forward_times_old)))))
        print('      new:  ', numpy.mean(forward_times_new), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(forward_times_new)/numpy.mean(forward_times_old)))))
        print('      aware:', numpy.mean(forward_times_aware), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(forward_times_aware)/numpy.mean(forward_times_old)))))
        print('    Mean LRP times 1 (simple lrp):')
        print('      old:  ', numpy.mean(lrp_times_old), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_old)/numpy.mean(lrp_times_old)))))
        print('      new:  ', numpy.mean(lrp_times_new), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_new)/numpy.mean(lrp_times_old)))))
        print('      aware:', numpy.mean(lrp_times_aware), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_aware)/numpy.mean(lrp_times_old)))))
        print('    Mean LRP times 2 (epsilon lrp):')
        print('      old:  ', numpy.mean(lrp_times_old2), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_old2)/numpy.mean(lrp_times_old2)))))
        print('      new:  ', numpy.mean(lrp_times_new2), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_new2)/numpy.mean(lrp_times_old2)))))
        print('      aware:', numpy.mean(lrp_times_aware2), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_aware2)/numpy.mean(lrp_times_old2)))))
        print('    Mean LRP times 3 (alpha=2 lrp):')
        print('      old:  ', numpy.mean(lrp_times_old3), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_old3)/numpy.mean(lrp_times_old3)))))
        print('      new:  ', numpy.mean(lrp_times_new3), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_new3)/numpy.mean(lrp_times_old3)))))
        print('      aware:', numpy.mean(lrp_times_aware3), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_aware3)/numpy.mean(lrp_times_old3)))))
        print('    Mean LRP times 4 (alpha=1 lrp):')
        print('      old:  ', numpy.mean(lrp_times_old4), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_old4)/numpy.mean(lrp_times_old4)))))
        print('      new:  ', numpy.mean(lrp_times_new4), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_new4)/numpy.mean(lrp_times_old4)))))
        print('      aware:', numpy.mean(lrp_times_aware4), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_aware4)/numpy.mean(lrp_times_old4)))))
        print('    Mean LRP times 5 (alpha=0 lrp):')
        print('      old:  ', numpy.mean(lrp_times_old5), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_old5)/numpy.mean(lrp_times_old5)))))
        print('      new:  ', numpy.mean(lrp_times_new5), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_new5)/numpy.mean(lrp_times_old5)))))
        print('      aware:', numpy.mean(lrp_times_aware5), '({}% speedup vs old)'.format(int(100*(1 - numpy.mean(lrp_times_aware5)/numpy.mean(lrp_times_old5)))))
        print('    Mean Total times with LRP once:')
        oldtotal = numpy.mean(numpy.array(lrp_times_old) + numpy.array(forward_times_old))
        newtotal = numpy.mean(numpy.array(lrp_times_new) + numpy.array(forward_times_new))
        awaretotal = numpy.mean(numpy.array(lrp_times_aware) + numpy.array(forward_times_aware))
        print('      old:  ', oldtotal, '({}% speedup vs old)'.format(int(100*(1 - oldtotal/oldtotal))))
        print('      new:  ', newtotal, '({}% speedup vs old)'.format(int(100*(1 - newtotal/oldtotal))))
        print('      aware:', awaretotal, '({}% speedup vs old)'.format(int(100*(1 - awaretotal/oldtotal))))
        print('    Mean Total times with LRP twice (simple+epsilon):')
        oldtotaltwice = numpy.mean(numpy.array(lrp_times_old) + numpy.array(lrp_times_old2) + numpy.array(forward_times_old))
        newtotaltwice = numpy.mean(numpy.array(lrp_times_new) + numpy.array(lrp_times_new2) + numpy.array(forward_times_new))
        awaretotaltwice = numpy.mean(numpy.array(lrp_times_aware) + numpy.array(lrp_times_aware2) + numpy.array(forward_times_aware))
        print('      old:  ', oldtotaltwice, '({}% speedup vs old)'.format(int(100*(1 - oldtotaltwice/oldtotaltwice))))
        print('      new:  ', newtotaltwice, '({}% speedup vs old)'.format(int(100*(1 - newtotaltwice/oldtotaltwice))))
        print('      aware:', awaretotaltwice, '({}% speedup vs old)'.format(int(100*(1 - awaretotaltwice/oldtotaltwice))))
        print('    Mean Total times with LRP five times(simple,epsilon,alpha=2,alpha=1,alpha=0):')
        oldtotalfive = oldtotaltwice + numpy.mean(numpy.array(lrp_times_old3) + numpy.array(lrp_times_old4) + numpy.array(lrp_times_old5))
        newtotalfive = newtotaltwice +  numpy.mean(numpy.array(lrp_times_new3) + numpy.array(lrp_times_new4) + numpy.array(lrp_times_new5))
        awaretotalfive = awaretotaltwice +  numpy.mean(numpy.array(lrp_times_aware3) + numpy.array(lrp_times_aware4) + numpy.array(lrp_times_aware5))
        print('      old:  ', oldtotalfive, '({}% speedup vs old)'.format(int(100*(1 - oldtotalfive/oldtotalfive))))
        print('      new:  ', newtotalfive, '({}% speedup vs old)'.format(int(100*(1 - newtotalfive/oldtotalfive))))
        print('      aware:', awaretotalfive, '({}% speedup vs old)'.format(int(100*(1 - awaretotalfive/oldtotalfive))))
        print('')
        print('')

#run benchmark
benchmark()
