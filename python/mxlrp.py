import mxnet as mx
from mxnet import nd

import types

import numpy as np # TODO: remove numpy, update to latest mxnet version

def translate_to_gluon(nn_sta, ctx=mx.cpu(), dtype='float32'):
    '''
    Helper function that takes a network defined with the lrp toolbox standalone implementation and converts it to a gluon network of same functionality.
    '''

    nn_gluon = mx.gluon.nn.HybridSequential()

    print('INPUT NETWORK:\n')

    print('-----------------')
    for m_ind, mod in enumerate(nn_sta.modules):

        # find next module (used to integrate activation functions into gluon blocks)
        if m_ind < len(nn_sta.modules) - 1:
            next_module_name = nn_sta.modules[m_ind + 1].__class__.__name__
        else:
            next_module_name = None


        print(mod.__class__.__name__)

        if mod.__class__.__name__ == 'Linear':

            if next_module_name == 'Rect':
                activation = 'relu'
            else:
                activation = None

            weight = mod.W
            bias   = mod.B

            print('Dense layer specifications:')
            print('W : {}'.format(weight.shape))
            print('B : {}'.format(bias.shape))
            print('-----------------')

            # reproduce Linear layer in gluon
            dense = mx.gluon.nn.Dense(units=weight.shape[1], in_units=weight.shape[0], use_bias=True, activation=activation)
            dense.collect_params().initialize(ctx=ctx)

            # adjust dtype and transfer parameters
            if ctx != mx.cpu():
                print('Parameter currently not supported on gpu (takes ages)')
            else:
                print('->Begin parameter casting')
                dense.weight.cast(dtype)
                dense.bias.cast(dtype)
            dense.weight.set_data(nd.array(weight.T, dtype=dtype))
            dense.bias.set_data(  nd.array(bias,     dtype=dtype))

            nn_gluon.add(dense)

        elif mod.__class__.__name__ == 'Convolution':

            if next_module_name == 'Rect':
                activation = 'relu'
            else:
                activation = None

            filtersize = mod.fh, mod.fw, mod.fd, mod.n
            stride     = mod.stride
            weight     = mod.W
            bias       = mod.B

            print('Conv layer specifications:')
            print(filtersize, stride, weight.shape, bias.shape)

            # reproduce conv layer in gluon
            conv = mx.gluon.nn.Conv2D(filtersize[3], filtersize[0:2], stride, padding=(0, 0), dilation=(1, 1), groups=1, layout='NCHW', activation=activation, use_bias=True, weight_initializer=None, bias_initializer='zeros', in_channels=filtersize[2])
            print('... layer created')
            conv.collect_params().initialize(ctx=ctx)
            print('... parameters initialized')

            # adjust dtype and transfer parameters
            if ctx != mx.cpu():
                print('Parameter currently not supported on gpu (takes ages)')
            else:
                print('->Begin parameter casting')
                conv.weight.cast(dtype)
                conv.bias.cast(dtype)
            print('->Begin parameter update')
            conv.weight.set_data(nd.array(weight.transpose(3, 2, 0, 1), dtype=dtype))
            conv.bias.set_data(nd.array(bias, dtype=dtype))
            print('... parameters updated')

            nn_gluon.add(conv)

        elif mod.__class__.__name__ == 'SumPool':

            pool = mod.pool
            stride = mod.stride

            print('Pool layer specifications:')
            print(pool, stride)
            print('-----------------')

            pool = mx.gluon.nn.AvgPool2D(pool_size = pool, strides = stride)
            pool.is_sumpool = True
            nn_gluon.add(pool)

        elif mod.__class__.__name__ == 'MaxPool':

            pool = mod.pool
            stride = mod.stride

            print('Maxpool layer specifications:')
            print(pool, stride)
            print('-----------------')

            pool = mx.gluon.nn.MaxPool2D(pool_size = pool, strides = stride)
            nn_gluon.add(pool)

        else:
            print('-----------------')

    print('\n\n')

    print('OUTPUT NETWORK:')
    print('-----------------')
    print(nn_gluon)

    return nn_gluon

def patch_lrp_gradient(net):
    for layer in net._children:
        if layer.__class__.__name__ == 'Dense':
            layer.hybrid_forward = types.MethodType(dense_hybrid_forward_lrp, layer)
            print('...updated dense layer')
        elif layer.__class__.__name__ == 'Conv2D':
            layer.hybrid_forward = types.MethodType(convolution_hybrid_forward_lrp, layer)
            print('...updated conv layer')
        elif layer.__class__.__name__ == 'AvgPool2D':

            if hasattr(layer, 'is_sumpool'):
                layer.hybrid_forward = types.MethodType(sumpool_hybrid_forward_lrp, layer)
                print('updated AvgPool2D (that is sumpool)')
            else:
                # TODO: add regular sumpool treatment (manage the pooling flag, rest should be the same)
                print('regular sumpool not yet implemented, add that')

        elif layer.__class__.__name__ == 'MaxPool2D':
                layer.hybrid_forward = types.MethodType(maxpool_hybrid_forward_lrp, layer)
                print('updated MaxPool2D (VERY INEFFICIENT!!!)')

## ######################### ##
# LRP-PATCHED HYBRID_FORWARDS #
## ######################### ##


def dense_hybrid_forward_lrp(self, F, x, weight, bias=None, lrp_type=None, lrp_param=0.):
        act = F.Custom(x, weight, bias, lrp_type=lrp_type, lrp_param=lrp_param, op_type='dense_lrp')

        if self.act is not None:
            act = F.BlockGrad(self.act(act) - act) + act
        return act


def convolution_hybrid_forward_lrp(self, F, x, weight, bias=None, lrp_type=None, lrp_param=0.):

        kernel, stride, dilate, pad, num_filter, num_group, no_bias, layout = map(self._kwargs.get, ['kernel', 'stride', 'dilate', 'pad', 'num_filter', 'num_group', 'no_bias', 'layout'])

        act = F.Custom(x, weight, bias, kernel=kernel, stride=stride, dilate=dilate, pad=pad, num_filter=num_filter, num_group=num_group, no_bias=no_bias, layout=layout, lrp_type=lrp_type, lrp_param=lrp_param, op_type='conv_lrp')

        if self.act is not None:
            act = F.BlockGrad(self.act(act) - act) + act
        return act

def sumpool_hybrid_forward_lrp(self, F, x, lrp_type=None, lrp_param=0.):

        kernel, stride, pad = map(self._kwargs.get, ['kernel', 'stride', 'pad'])

        return F.Custom(x, kernel=kernel, stride=stride, pad=pad, lrp_type=lrp_type, lrp_param=lrp_param, op_type='sumpool_lrp')

def maxpool_hybrid_forward_lrp(self, F, x, lrp_type=None, lrp_param=0.):

        kernel, stride, pad = map(self._kwargs.get, ['kernel', 'stride', 'pad'])

        if np.sum(pad) != 0:
            print('error: padded max pooling lrp not yet implemented')
            exit()

        return F.Custom(x, kernel=kernel, stride=stride, lrp_type=lrp_type, lrp_param=lrp_param, op_type='maxpool_lrp')

## ############### ##
# CONVOLUTION LAYER #
## ############### ##
class ConvLRP(mx.operator.CustomOp):

    def __init__(self, kernel, stride, dilate, pad, num_filter, num_group, no_bias, layout, lrp_type, lrp_param):
        self.kernel     = kernel
        self.stride     = stride
        self.dilate     = dilate
        self.pad        = pad
        self.num_filter = num_filter
        self.num_group  = num_group
        self.no_bias    = no_bias
        self.layout     = layout
        self.lrp_type   = lrp_type
        self.lrp_param  = lrp_param

    def forward(self, is_train, req, in_data, out_data, aux):
        x      = in_data[0]
        weight = in_data[1]
        bias   = in_data[2]

        y = nd.Convolution(x, weight, bias, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=self.no_bias, layout=self.layout)

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x      = in_data[0]
        weight = in_data[1]
        y      = out_data[0]
        ry     = out_grad[0]

        # simple-LRP
        zs = y + 1e-16*((y >= 0)*2 - 1.)
        F = ry/zs

        rx = x * nd.Deconvolution(F, weight, bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout)
        self.assign(in_grad[0], req[0], rx)

@mx.operator.register("conv_lrp")
class ConvLRPProp(mx.operator.CustomOpProp):
    def __init__(self, kernel, stride, dilate, pad, num_filter, num_group, no_bias, layout, lrp_type, lrp_param):
        super(ConvLRPProp, self).__init__(True)

        self.kernel     = eval(kernel)
        self.stride     = eval(stride)
        self.dilate     = eval(dilate)
        self.pad        = eval(pad)
        self.num_filter = eval(num_filter)
        self.num_group  = eval(num_group)
        self.no_bias    = eval(no_bias)
        self.layout     = layout
        self.lrp_type   = lrp_type
        self.lrp_param  = eval(lrp_param)

    def list_arguments(self):
        return ['data', 'weight', 'bias']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape   = in_shapes[0]
        weight_shape = in_shapes[1]
        bias_shape   = in_shapes[2]

        h = (data_shape[2] + self.pad[0] - weight_shape[2]) // self.stride[0] + 1
        w = (data_shape[3] + self.pad[1] - weight_shape[3]) // self.stride[1] + 1

        output_shape = (data_shape[0], weight_shape[0], h, w)
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape, weight_shape, bias_shape), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return ConvLRP(self.kernel, self.stride, self.dilate, self.pad, self.num_filter, self.num_group, self.no_bias, self.layout, self.lrp_type, self.lrp_param)

## ######### ##
# DENSE LAYER #
## ######### ##
class DenseLRP(mx.operator.CustomOp):
    def __init__(self, lrp_type, lrp_param):

        self.lrp_type   = lrp_type
        self.lrp_param  = lrp_param

    def forward(self, is_train, req, in_data, out_data, aux):
        x      = in_data[0]
        weight = in_data[1]
        bias   = in_data[2]

        y = nd.dot(x, weight.T) + bias # attention: Weight seems the other way around than in toolbox

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x      = in_data[0]
        weight = in_data[1]
        y      = out_data[0]
        ry     = out_grad[0]

        # simple-LRP
        zs = y + 1e-16*( (y >= 0) * 2 - 1.) #add weakdefault stabilizer to denominator
        z  = nd.expand_dims(weight.T, axis=0) * nd.expand_dims(x, axis=2) #localized preactivations
        rx = nd.sum(z * nd.expand_dims(ry/zs, 1), axis=2)

        self.assign(in_grad[0], req[0], rx)

@mx.operator.register("dense_lrp")  # register with name "dense_lrp"
class DenseLRPProp(mx.operator.CustomOpProp):
    def __init__(self, lrp_type, lrp_param):
        super(DenseLRPProp, self).__init__(True)

        self.lrp_type   = lrp_type
        self.lrp_param  = eval(lrp_param)

    def list_arguments(self):
        return ['data', 'weight', 'bias']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape   = in_shapes[0]
        weight_shape = in_shapes[1]
        bias_shape   = in_shapes[2]
        output_shape = (data_shape[0], weight_shape[0])
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape, weight_shape, bias_shape), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return DenseLRP(self.lrp_type, self.lrp_param)

# WORK IN PROGRESS:

## ########### ##
# POOLING LAYER #
## ########### ##
# TODO: try to imitate the sumpooling op of the lrp toolbox with convolution
class SumPoolLRP(mx.operator.CustomOp):

    def __init__(self, kernel, stride, pad, lrp_type, lrp_param, normalizer='dimsqrt'):
        self.kernel    = kernel
        self.stride    = stride
        self.pad       = pad
        self.normalizer = normalizer
        self.normalizer_value = None

        self.channels = None
        self.weight   = None

        self.lrp_type = lrp_type
        self.lrp_param= lrp_param

    def forward(self, is_train, req, in_data, out_data, aux):
        x      = in_data[0]

        if self.weight is None or self.channels is None or self.weight.shape[0] != self.channels:
            # convolution weight that imitates a sumpooling operation
            self.channels = x.shape[1]
            self.weight   = nd.array(np.ones(self.kernel)[None, None, :, :] * np.eye(self.channels)[:,:,None,None], dtype=x.dtype, ctx=x.context) # nd.expand_dims(nd.expand_dims(nd.ones(self.kernel), 0), 0) * nd.expand_dims(nd.expand_dims(nd.eye(self.channels), 2), 2)

        y = nd.Pooling(x, kernel=self.kernel, stride=self.stride, pool_type = 'sum', pad=self.pad)

        if self.normalizer == 'dimsqrt':
            self.normalizer_value = 1 / np.sqrt(np.prod(self.kernel))
        else:
            print('Normalizer value invalid: standard sumpool performed')

        y = y  * self.normalizer_value

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x      = in_data[0]
        y      = out_data[0]
        ry     = out_grad[0]

        # simple-LRP
        zs = y / self.normalizer_value + 1e-16*((y >= 0)*2 - 1.)
        F = ry/zs
        rx = x * nd.Deconvolution(F, self.weight, bias=None, kernel=self.kernel, stride=self.stride, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], no_bias=True)

        self.assign(in_grad[0], req[0], rx)

@mx.operator.register("sumpool_lrp")  # register with name "sumpool_lrp"
class SumPoolLRPProp(mx.operator.CustomOpProp):
    def __init__(self, kernel, stride, lrp_type, lrp_param, pad=(0,0)):
        super(SumPoolLRPProp, self).__init__(True)

        self.normalizer= 'dimsqrt'
        self.kernel    = eval(kernel)
        self.stride    = eval(stride)
        self.pad       = eval(pad)
        self.lrp_type  = lrp_type
        self.lrp_param = eval(lrp_param)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape   = in_shapes[0]

        h = (data_shape[2] + self.pad[0] - self.kernel[0]) // self.stride[0] + 1
        w = (data_shape[3] + self.pad[1] - self.kernel[1]) // self.stride[1] + 1

        output_shape = (data_shape[0], data_shape[1], h, w)
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return SumPoolLRP(self.kernel, self.stride, self.pad, self.lrp_type, self.lrp_param, self.normalizer)


# MaxPool as in LRP python Toolbox: very inefficient

class MaxPoolLRP(mx.operator.CustomOp):

    def __init__(self, kernel, stride, lrp_type, lrp_param):
        self.kernel    = kernel
        self.stride    = stride

        self.lrp_type = lrp_type
        self.lrp_param = lrp_param

    def forward(self, is_train, req, in_data, out_data, aux):
        x      = in_data[0]
        N,D,H,W = x.shape

        hpool,   wpool   = self.kernel
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) // hstride + 1
        Wout = (W - wpool) // wstride + 1

        #initialize pooled output
        y = nd.zeros((N,D,Hout,Wout))

        for i in range(Hout):
            for j in range(Wout):
                y[:,:,i,j] = x[:,:, i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool:].max(axis=(2,3))

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x      = in_data[0]
        y      = out_data[0]
        ry     = out_grad[0]

        N,D,H,W = x.shape

        hpool,   wpool   = self.kernel
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) // hstride + 1
        Wout = (W - wpool) // wstride + 1

        rx = nd.zeros_like(x,dtype=x.dtype)

        for i in range(Hout):
            for j in range(Wout):
                Z = y[:,:,i:i+1,j:j+1] == x[:,:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool]
                Zs = Z.sum(axis=(2,3),keepdims=True) #thanks user wodtko for reporting this bug/fix
                rx[:,:,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool] += (Z / Zs) * ry[:,:,i:i+1,j:j+1]

        self.assign(in_grad[0], req[0], rx)

@mx.operator.register("maxpool_lrp")  # register with name "maxpool_lrp"
class MaxPoolLRPProp(mx.operator.CustomOpProp):
    def __init__(self, kernel, stride, lrp_type, lrp_param):
        super(MaxPoolLRPProp, self).__init__(True)

        self.normalizer= 'dimsqrt'
        self.kernel    = eval(kernel)
        self.stride    = eval(stride)
        self.pad       = [0,0]
        self.lrp_type  = lrp_type
        self.lrp_param = eval(lrp_param)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape   = in_shapes[0]

        h = (data_shape[2] + self.pad[0] - self.kernel[0]) // self.stride[0] + 1
        w = (data_shape[3] + self.pad[1] - self.kernel[1]) // self.stride[1] + 1

        output_shape = (data_shape[0], data_shape[1], h, w)
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return MaxPoolLRP(self.kernel, self.stride, self.lrp_type, self.lrp_param)
