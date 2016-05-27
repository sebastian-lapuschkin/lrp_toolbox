'''
@author: Sebastian Bach
@maintainer: Sebastian Bach
@contact: sebastian.bach@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.0
@copyright: Copyright (c)  2015, Sebastian Bach, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
@license : BSD-2-Clause
'''

import os
import pickle
import numpy as np
from modules import Sequential,Linear,Tanh,Rect,SoftMax

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

        The exception formed by the linear layer implementation modules.Linear incorporates in raw text form as

        Linear m n
        W
        B

        with m and n being integer values describing the dimensions of the weight matrix W as [m x n] ,
        W being the human readable ascii-representation of the matrix, where each row of W is written out as a
        white space separated line of doubles.
        After the m lines describing W, the bias term B is written out as a single line of n white space separated double values.
    '''

    if not os.path.exists(path):
        raise IOError('model_io.read : No such file or directory: {0}'.format(path))

    if fmt is None: #try to infer format
        fmt = os.path.splitext(path)[1].replace('.','').lower()

    return _read_as[fmt](path)


def _read_pickled(path):
    print 'loading pickled model from',path
    return pickle.load(open(path,'rb'))


def _read_txt(path):
    print 'loading plain text model from', path

    with open(path, 'rb') as f:
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
                for i in xrange(m):
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

            c+=1;
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
    if fmt is None:
        fmt = os.path.splitext(path)[1].replace('.','').lower()

    _write_as[fmt](model, path)


def _write_pickled(model, path):
    print 'writing model pickled to',path
    with open(path, 'wb') as f:
        pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)


def _write_txt(model,path):
    print 'writing model as plain text to',path

    if not isinstance(model, Sequential):
        ''' TODO: Error Handling '''

    with open(path, 'wb') as f:
        for m in model.modules:
            if isinstance(m,Linear):
                f.write('{0} {1} {2}\n'.format(m.__class__.__name__,m.m,m.n))
                for row in m.W:
                    f.write(' '.join([str(r) for r in row]) + '\n' )
                f.write(' '.join([str(b) for b in m.B]) +'\n')
            else:
                f.write(m.__class__.__name__ + '\n')


_write_as = {'pickled': _write_pickled,\
            'pickle':_write_pickled,\
            'nn':_write_pickled,\
            '':_write_pickled,\
            'txt':_write_txt,\
            }


