'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import os
import pickle
from modules import Sequential,Linear,Tanh,Rect,SoftMax,Convolution,Flatten,SumPool,MaxPool
import numpy
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"):
    import cupy
    import cupy as np
na = np.newaxis

#--------------------
#   model reading
#--------------------

def read(path, fmt = None):
    '''
    Read neural network model from given path. Supported are files written in either plain text or via python's pickle module.

    Parameters
    ----------

    path : str
        the path to the file to read

    fmt : str
        optional. explicitly state how to interpret the target file. if not given, format is inferred from path.
        options are 'pickled','pickle','' and 'nn' to specify the pickle file format and 'txt' for a plain text
        format shared with the matlab implementation of the toolbox

    Returns
    -------
        model: modules.Sequential
            the  neural network model, realized as a sequence of network modules.

    Notes
    -----
        the plain text file format is shared with the matlab implementation of the LRP Toolbox and describes
        the model by listing its computational layers line by line as

        <Layername_i> [<input_size> <output_size>]
        [<Layer_params_i>]

        since all implemented modules except for modules.Linear operate point-wise on the given data, the optional
        information indicated by brackets [ ] is not used and only the name of the layer is witten, e.g.

        Rect

        Tanh

        SoftMax

        Flatten

        The exception formed by the linear layer implementation modules.Linear and modules.Convolution incorporates in raw text form as

        Linear m n
        W.flatten()
        B.flatten()

        with m and n being integer values describing the dimensions of the weight matrix W as [m x n] ,
        W being the human readable ascii-representation of the flattened matrix in m * n white space separated double values.
        After the line describing W, the bias term B is written out as a single line of n white space separated double values.

        Convolution h w d n s0 s1
        W.flatten()
        B.flatten()

        Semantics as above, with h, w, d being the filter heigth, width and depth and n being the number of filters of that layer.
        s0 and s1 specify the stride parameter in vertical (axis 0) and horizontal (axis 1) direction the layer operates on.

        Pooling layers have a parameterized one-line-description

        [Max|Sum]Pool h w s0 s1

        with h and w designating the pooling mask size and s0 and s1 the pooling stride.
    '''

    if not os.path.exists(path):
        raise IOError('model_io.read : No such file or directory: {0}'.format(path))

    if fmt is None: #try to infer format
        fmt = os.path.splitext(path)[1].replace('.','').lower()

    model = _read_as[fmt](path)
    if imp.find_spec("cupy"):
        model.to_cupy()
    return model


def _read_pickled(path):
    print('loading pickled model from',path)
    with open(path,'rb') as f:
        p = pickle.load(f, encoding='latin1')
    return p


def _read_txt(path):
    print('loading plain text model from',path)

    def _read_txt_helper(path):
        with open(path,'r') as f:
            content = f.read().split('\n')

            modules = []
            c = 0
            line = content[c]

            while len(line) > 0:
                if line.startswith(Linear.__name__): # @UndefinedVariable import error suppression for PyDev users
                    '''
                    Format of linear layer
                    Linear <rows_of_W> <columns_of_W>
                    <flattened weight matrix W>
                    <flattened bias vector>
                    '''
                    _,m,n = line.split();   m = int(m); n = int(n)
                    layer = Linear(m,n)
                    layer.W = np.array([float(weightstring) for weightstring in content[c+1].split() if len(weightstring) > 0]).reshape((m,n))
                    layer.B = np.array([float(weightstring) for weightstring in content[c+2].split() if len(weightstring) > 0])
                    modules.append(layer)
                    c+=3 # the description of a linear layer spans three lines

                elif line.startswith(Convolution.__name__): # @UndefinedVariable import error suppression for PyDev users
                    '''
                    Format of convolution layer
                    Convolution <rows_of_W> <columns_of_W> <depth_of_W> <number_of_filters_W> <stride_axis_0> <stride_axis_1>
                    <flattened filter block W>
                    <flattened bias vector>
                    '''

                    _,h,w,d,n,s0,s1 = line.split()
                    h = int(h); w = int(w); d = int(d); n = int(n); s0 = int(s0); s1 = int(s1)
                    layer = Convolution(filtersize=(h,w,d,n), stride=(s0,s1))
                    layer.W = np.array([float(weightstring) for weightstring in content[c+1].split() if len(weightstring) > 0]).reshape((h,w,d,n))
                    layer.B = np.array([float(weightstring) for weightstring in content[c+2].split() if len(weightstring) > 0])
                    modules.append(layer)
                    c+=3 #the description of a convolution layer spans three lines

                elif line.startswith(SumPool.__name__): # @UndefinedVariable import error suppression for PyDev users
                    '''
                    Format of sum pooling layer
                    SumPool <mask_heigth> <mask_width> <stride_axis_0> <stride_axis_1>
                    '''

                    _,h,w,s0,s1 = line.split()
                    h = int(h); w = int(w); s0 = int(s0); s1 = int(s1)
                    layer = SumPool(pool=(h,w),stride=(s0,s1))
                    modules.append(layer)
                    c+=1 # one line of parameterized layer description

                elif line.startswith(MaxPool.__name__): # @UndefinedVariable import error suppression for PyDev users
                    '''
                    Format of max pooling layer
                    MaxPool <mask_heigth> <mask_width> <stride_axis_0> <stride_axis_1>
                    '''

                    _,h,w,s0,s1 = line.split()
                    h = int(h); w = int(w); s0 = int(s0); s1 = int(s1)
                    layer = MaxPool(pool=(h,w),stride=(s0,s1))
                    modules.append(layer)
                    c+=1 # one line of parameterized layer description

                elif line.startswith(Flatten.__name__): # @UndefinedVariable import error suppression for PyDev users
                    modules.append(Flatten()) ; c+=1 #one line of parameterless layer description
                elif line.startswith(Rect.__name__): # @UndefinedVariable import error suppression for PyDev users
                    modules.append(Rect()) ; c+= 1 #one line of parameterless layer description
                elif line.startswith(Tanh.__name__): # @UndefinedVariable import error suppression for PyDev users
                    modules.append(Tanh()) ; c+= 1 #one line of parameterless layer description
                elif line.startswith(SoftMax.__name__): # @UndefinedVariable import error suppression for PyDev users
                    modules.append(SoftMax()) ; c+= 1 #one line of parameterless layer description
                else:
                    raise ValueError('Layer type identifier' + [s for s in line.split() if len(s) > 0][0] +  ' not supported for reading from plain text file')

                #skip info of previous layers, read in next layer header
                line = content[c]



        return Sequential(modules)
    # END _read_txt_helper()

    try:
        return _read_txt_helper(path)

    except ValueError as e:
        #numpy.reshape may throw ValueErros if reshaping does not work out.
        #In this case: fall back to reading the old plain text format.
        print('probable reshaping/formatting error while reading plain text network file.')
        print('ValueError message: {}'.format(e))
        print('Attempting fall-back to legacy plain text format interpretation...')
        return _read_txt_old(path)
        print('fall-back successfull!')


def _read_txt_old(path):
    print('loading plain text model from', path)

    with open(path, 'r') as f:
        content = f.read().split('\n')

        modules = []
        c = 0
        line = content[c]
        while len(line) > 0:
            if line.startswith(Linear.__name__): # @UndefinedVariable import error suppression for PyDev users
                lineparts = line.split()
                m = int(lineparts[1])
                n = int(lineparts[2])
                mod = Linear(m,n)
                for i in range(m):
                    c+=1
                    mod.W[i,:] = np.array([float(val) for val in content[c].split() if len(val) > 0])

                c+=1
                mod.B = np.array([float(val) for val in content[c].split()])
                modules.append(mod)

            elif line.startswith(Rect.__name__): # @UndefinedVariable import error suppression for PyDev users
                modules.append(Rect())
            elif line.startswith(Tanh.__name__): # @UndefinedVariable import error suppression for PyDev users
                modules.append(Tanh())
            elif line.startswith(SoftMax.__name__): # @UndefinedVariable import error suppression for PyDev users
                modules.append(SoftMax())
            else:
                raise ValueError('Layer type ' + [s for s in line.split() if len(s) > 0][0] +  ' not supported by legacy plain text format.')

            c+=1
            line = content[c]

        return Sequential(modules)


_read_as = {'pickled': _read_pickled,\
            'pickle':_read_pickled,\
            'nn':_read_pickled,\
            '':_read_pickled,\
            'txt':_read_txt,\
            }




#--------------------
#   model writing
#--------------------

def write(model, path, fmt = None):
    '''
    Write neural a network model to a given path. Supported are either plain text or via python's pickle module.
    The model is cleaned of any temporary variables , e.g. hidden layer inputs or outputs, prior to writing

    Parameters
    ----------

    model : modules.Sequential
        the object representing the model.

    path : str
        the path to the file to read

    fmt : str
        optional. explicitly state how to write the file. if not given, format is inferred from path.
        options are 'pickled','pickle','' and 'nn' to specify the pickle file format and 'txt' for a plain text
        format shared with the matlab implementation of the toolbox

    Notes
    -----
        see the Notes - Section in the function documentation of model_io.read() for general info and a format
        specification of the plain text representation of neural network models
    '''

    model.clean()
    if not np == numpy: #np = cupy
        model.to_numpy() #TODO reconvert after writing?
    if fmt is None:
        fmt = os.path.splitext(path)[1].replace('.','').lower()

    _write_as[fmt](model, path)


def _write_pickled(model, path):
    print('writing model pickled to',path)
    with open(path, 'wb') as f:
        pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)


def _write_txt(model,path):
    print('writing model as plain text to',path)

    if not isinstance(model, Sequential):
        raise Exception('Argument "model" must be an instance of module.Sequential, wrapping a sequence of neural network computation layers, but is {0}'.format(type(model)))

    with open(path, 'w') as f:
        for layer in model.modules:
            if isinstance(layer,Linear):
                '''
                Format of linear layer
                Linear <rows_of_W> <columns_of_W>
                <flattened weight matrix W>
                <flattened bias vector>
                '''

                f.write('{0} {1} {2}\n'.format(layer.__class__.__name__,layer.m,layer.n))
                f.write(' '.join([repr(w) for w in layer.W.flatten()]) + '\n')
                f.write(' '.join([repr(b) for b in layer.B.flatten()]) + '\n')

            elif isinstance(layer,Convolution):
                '''
                    Format of convolution layer
                    Convolution <rows_of_W> <columns_of_W> <depth_of_W> <number_of_filters_W> <stride_axis_0> <stride_axis_1>
                    <flattened filter block W>
                    <flattened bias vector>
                '''

                f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(
                    layer.__class__.__name__,\
                    layer.fh,\
                    layer.fw,\
                    layer.fd,\
                    layer.n,\
                    layer.stride[0],\
                    layer.stride[1]
                ))
                f.write(' '.join([repr(w) for w in layer.W.flatten()]) + '\n')
                f.write(' '.join([repr(b) for b in layer.B.flatten()]) + '\n')

            elif isinstance(layer,SumPool):
                '''
                    Format of sum pooling layer
                    SumPool <mask_heigth> <mask_width> <stride_axis_0> <stride_axis_1>
                '''

                f.write('{0} {1} {2} {3} {4}\n'.format(
                    layer.__class__.__name__,\
                    layer.pool[0],\
                    layer.pool[1],\
                    layer.stride[0],\
                    layer.stride[1]))

            elif isinstance(layer,MaxPool):
                '''
                    Format of max pooling layer
                    MaxPool <mask_heigth> <mask_width> <stride_axis_0> <stride_axis_1>
                '''

                f.write('{0} {1} {2} {3} {4}\n'.format(
                    layer.__class__.__name__,\
                    layer.pool[0],\
                    layer.pool[1],\
                    layer.stride[0],\
                    layer.stride[1]))

            else:
                '''
                all other layers are free from parameters. Format is thus:
                <Layername>
                '''
                f.write(layer.__class__.__name__ + '\n')


_write_as = {'pickled': _write_pickled,\
            'pickle':_write_pickled,\
            'nn':_write_pickled,\
            '':_write_pickled,\
            'txt':_write_txt,\
            }


