## Programmed using mxnet version 1.1.0 ##

import mxnet as mx
from mxnet import nd, autograd

import types
import numpy as np 

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
                # print('Parameter currently not supported on gpu (takes ages)')
                print('gpu mode: still so slow???')


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

def patch_lrp_gradient(net, lrp_type='simple', lrp_param = 0., switch_layer = -1, debug_output=False, init_idx = 0, outer_call=True, maxpool_treatment='grad_n_sum'):
    # important: current implementation uses layer indices that assume that we onle use a tree structure of HybridSequential modules.
    # as soon as we add hybridconcurrents, the layer numbering might get trickier.

    if debug_output:
        print('switch_layer = {}'.format(switch_layer))

    inefficient_maxpool = False
    wta_mpool = False

    for layer_idx, layer in enumerate(net._children):

        layer_idx += init_idx

        if debug_output:
            print(layer_idx, layer.__class__.__name__)

        if layer.__class__.__name__ == 'HybridSequential':
            print('... entering HybridSequential')
            init_idx = patch_lrp_gradient(layer, lrp_type, lrp_param, switch_layer, init_idx = layer_idx + 1, outer_call=False, maxpool_treatment=maxpool_treatment, debug_output=debug_output)

        if layer.__class__.__name__ == 'Dense':

            if layer._flatten:
                flatter = lambda x: x.reshape((0,-1))
            else:
                flatter = lambda x: x

            hybrid_forward_lrp = lambda self, F, x, weight, bias: dense_hybrid_forward_lrp(self, F, flatter(x), weight, bias, lrp_type=lrp_type, lrp_param=lrp_param)
            layer.hybrid_forward = types.MethodType(hybrid_forward_lrp, layer)

            if debug_output:
                print('...updated dense layer')
                print('lrp_type: {} | param: {}'.format(lrp_type, lrp_param))

        elif layer.__class__.__name__ == 'Conv2D':

            if layer_idx <= switch_layer:
                lrp_type_sw = 'alphabeta'
                lrp_param_sw = 1.

                if debug_output:
                    print('Conv layer (layer_idx {}): switched to alpha'.format(layer_idx))
            else:
                lrp_type_sw = lrp_type
                lrp_param_sw = lrp_param

            hybrid_forward_lrp = lambda self, F, x, weight, bias: convolution_hybrid_forward_lrp(self, F, x, weight, bias, lrp_type=lrp_type_sw, lrp_param=lrp_param_sw)
            layer.hybrid_forward = types.MethodType(hybrid_forward_lrp, layer)

            if debug_output:
                print('...updated conv layer')
                print('layer_idx={}, sw_layer={}'.format(layer_idx, switch_layer))

                print('lrp_type: {} | param: {}'.format(lrp_type_sw, lrp_param_sw))

        elif layer.__class__.__name__ == 'AvgPool2D':
            if hasattr(layer, 'is_sumpool'):
                hybrid_forward_lrp = lambda self, F, x: sumpool_hybrid_forward_lrp(self, F, x, lrp_type=lrp_type, lrp_param=lrp_param)
                layer.hybrid_forward = types.MethodType(hybrid_forward_lrp, layer)

                if debug_output:
                    print('updated AvgPool2D (that is sumpool)')
            else:
                # TODO: add regular sumpool treatment (manage the pooling flag, rest should be the same)
                print('regular sumpool not yet implemented, add that')

        elif layer.__class__.__name__ == 'MaxPool2D':

            if maxpool_treatment == 'loops':
                inefficient_maxpool = True
            elif maxpool_treatment == 'grad_n_sum':
                wta_mpool = True
            else:
                print('Warning: unknown maxpool treatment -{}-, using grad_n_sum (faster)'.format(maxpool_treatment))
                maxpool_treatment='grad_n_sum'

            hybrid_forward_lrp = lambda self, F, x: maxpool_hybrid_forward_lrp(self, F, x, lrp_type=lrp_type, lrp_param=lrp_param, maxpool_method=maxpool_treatment)
            layer.hybrid_forward = types.MethodType(hybrid_forward_lrp, layer)


    if inefficient_maxpool:
        print('WARNING: using inefficient maxpool implementation (LOOPS)!!!')
    elif wta_mpool:
        pass
        # print('WARNING: maxpool implementation uses mxnet gradient, relevance is not redistributed if several inputs in the window are equal and maximal. First max activation gets all the relevance.')

    if not outer_call:
        return layer_idx + 1

## ######################### ##
# LRP-PATCHED HYBRID_FORWARDS #
## ######################### ##


def dense_hybrid_forward_lrp(self, F, x, weight, bias=None, lrp_type='simple', lrp_param=0.):
        act = F.Custom(x, weight, bias, lrp_type=lrp_type, lrp_param=lrp_param, op_type='dense_lrp')

        if self.act is not None:
            act = F.BlockGrad(self.act(act) - act) + act
        return act


def convolution_hybrid_forward_lrp(self, F, x, weight, bias=None, lrp_type='simple', lrp_param=0.):

        kernel, stride, dilate, pad, num_filter, num_group, no_bias, layout = map(self._kwargs.get, ['kernel', 'stride', 'dilate', 'pad', 'num_filter', 'num_group', 'no_bias', 'layout'])

        act = F.Custom(x, weight, bias, kernel=kernel, stride=stride, dilate=dilate, pad=pad, num_filter=num_filter, num_group=num_group, no_bias=no_bias, layout=layout, lrp_type=lrp_type, lrp_param=lrp_param, op_type='conv_lrp')

        if self.act is not None:
            act = F.BlockGrad(self.act(act) - act) + act
        return act

def sumpool_hybrid_forward_lrp(self, F, x, lrp_type='simple', lrp_param=0.):

        kernel, stride, pad = map(self._kwargs.get, ['kernel', 'stride', 'pad'])

        return F.Custom(x, kernel=kernel, stride=stride, pad=pad, lrp_type=lrp_type, lrp_param=lrp_param, op_type='sumpool_lrp')

def maxpool_hybrid_forward_lrp(self, F, x, lrp_type='simple', lrp_param=0., maxpool_method='grad_n_sum'):

        kernel, stride, pad = map(self._kwargs.get, ['kernel', 'stride', 'pad'])

        if np.sum(pad) != 0:
            print('error: padded max pooling lrp not yet implemented')
            exit()

        return F.Custom(x, kernel=kernel, stride=stride, lrp_type=lrp_type, lrp_param=lrp_param, op_type='maxpool_lrp', maxpool_method=maxpool_method)

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
        bias   = in_data[2]
        y      = out_data[0]
        ry     = out_grad[0]

        # print('conv: {}|{}'.format(self.lrp_type, self.lrp_param))

        if self.lrp_type == 'simple':
            zs = y + 1e-16*((y >= 0)*2 - 1.)
            F = ry/zs
            rx = x * nd.Deconvolution(F, weight, bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout)

        elif self.lrp_type == 'epsilon' or self.lrp_type == 'eps':
            zs = y + self.lrp_param*((y >= 0)*2 - 1.)
            F = ry/zs
            rx = x * nd.Deconvolution(F, weight, bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout)

        elif self.lrp_type == 'alphabeta' or self.lrp_type == 'alpha':

            alpha = self.lrp_param
            beta  = 1 - alpha
            default_stabilizer = 1e-16


            weight_pos_idxs = weight >= 0
            weight_p = weight * weight_pos_idxs
            weight_n = weight * (1-weight_pos_idxs)

            x_pos_idxs = x >=0
            xp = x * x_pos_idxs
            xn = x * (1-x_pos_idxs)

            bias_pos_idxs = bias >= 0
            bias_p = bias * bias_pos_idxs
            bias_n = bias * (1-bias_pos_idxs)


            #index mask of positive forward predictions
            if alpha * beta != 0: #the general case: both parameters are not 0

                # TODO: wrong???
                # pos_idxs = y >= 0
                # zsp = y * pos_idxs + default_stabilizer * (pos_idxs*2 - 1.)
                #
                # neg_idxs = 1 - pos_idxs
                # zsn = y * neg_idxs + default_stabilizer * (neg_idxs*2 - 1.)
                no_bias_here=True
                Tp_nb = nd.Convolution(xp, weight_p, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=no_bias_here, layout=self.layout) + \
                        nd.Convolution(xn, weight_n, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=no_bias_here, layout=self.layout)
                Tn_nb = nd.Convolution(xn, weight_p, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=no_bias_here, layout=self.layout) + \
                        nd.Convolution(xp, weight_n, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=no_bias_here, layout=self.layout)

                # add the respective parts of the bias
                Tp = Tp_nb + nd.expand_dims(nd.expand_dims(nd.expand_dims(bias_p,0), 2), 3)
                Tn = Tn_nb + nd.expand_dims(nd.expand_dims(nd.expand_dims(bias_n,0), 2), 3)

                Fp = ry / Tp
                Fn = ry / Tn

                rp = xp * nd.Deconvolution(Fp, weight_p , bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout) + \
                     xn * nd.Deconvolution(Fp, weight_n , bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout)

                rn = xn * nd.Deconvolution(Fn, weight_p , bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout) + \
                     xp * nd.Deconvolution(Fn, weight_n , bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout)

                rx = alpha * rp + beta * rn

            elif alpha: #only alpha is not 0 -> alpha = 1, beta = 0
                no_bias_here=True
                Tp_nb = nd.Convolution(xp, weight_p, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=no_bias_here, layout=self.layout) + \
                        nd.Convolution(xn, weight_n, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=no_bias_here, layout=self.layout)

                Tp = Tp_nb + nd.expand_dims(nd.expand_dims(nd.expand_dims(bias_p,0), 2), 3)

                Fp = ry / Tp
                rp = xp * nd.Deconvolution(Fp, weight_p , bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout) + \
                     xn * nd.Deconvolution(Fp, weight_n , bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout)

                rx = alpha * rp

            elif beta: # only beta is not 0 -> alpha = 0, beta = 1

                no_bias_here=True
                Tn_nb = nd.Convolution(xn, weight_p, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=no_bias_here, layout=self.layout) + \
                        nd.Convolution(xp, weight_n, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, num_filter=self.num_filter, num_group=self.num_group, no_bias=no_bias_here, layout=self.layout)

                Tn = Tn_nb + nd.expand_dims(nd.expand_dims(nd.expand_dims(bias_n,0), 2), 3)

                Fn = ry / Tn
                rn = xn * nd.Deconvolution(Fn, weight_p , bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout) + \
                     xp * nd.Deconvolution(Fn, weight_n , bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout)

                rx = beta * rn
        else:
            print('Error in Conv layer: unknown LRP type {}'.format(self.lrp_type))

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

        h = (data_shape[2] + 2 * self.pad[0] - weight_shape[2]) // self.stride[0] + 1
        w = (data_shape[3] + 2 * self.pad[1] - weight_shape[3]) // self.stride[1] + 1

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

        # TODO: enable flatten support by using F.FullyConnected
        # act = F.FullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
        #                        flatten=self._flatten, name='fwd')

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x      = in_data[0]
        weight = in_data[1]
        bias   = in_data[2]
        y      = out_data[0]
        ry     = out_grad[0]

        # print('Dense: {}|{}'.format(self.lrp_type, self.lrp_param))

        if self.lrp_type == 'simple':
            zs = y + 1e-16*( (y >= 0) * 2 - 1.) #add weakdefault stabilizer to denominator
            F = ry / zs
            rx = x * nd.dot(F, weight)

        elif self.lrp_type == 'epsilon' or self.lrp_type == 'eps':
            zs = y + self.lrp_param*( (y >= 0) * 2 - 1.) #add epsilon stabilizer
            F = ry / zs
            rx = x * nd.dot(F, weight)

        elif self.lrp_type == 'alphabeta' or self.lrp_type == 'alpha':

            alpha = self.lrp_param
            beta  = 1 - alpha
            default_stabilizer = 1e-16

            z = nd.expand_dims(weight.T, axis=0) * nd.expand_dims(x, axis=2) #localized preactivations

            #index mask of positive forward predictions
            zplus = z > 0
            if alpha * beta != 0: #the general case: both parameters are not 0
                zp = z * zplus
                zsp = nd.sum(zp, axis=1) + nd.expand_dims(bias * (bias > 0), axis=0) + default_stabilizer

                zn = z - zp
                zsn = y - zsp - default_stabilizer

                rxp = alpha * nd.sum(zp * nd.expand_dims(ry/zsp, axis=1), axis=2)
                rxn = beta * nd.sum(zn * nd.expand_dims(ry/zsn, axis=1), axis=2)
                rx = rxp + rxn

            elif alpha: #only alpha is not 0 -> alpha = 1, beta = 0
                zp = z * zplus
                zsp = nd.sum(zp, axis=1) + nd.expand_dims(bias * (bias > 0), axis=0) + default_stabilizer
                rx = nd.sum(zp * nd.expand_dims(ry/zsp, axis=1), axis=2)

            elif beta: # only beta is not 0 -> alpha = 0, beta = 1
                zn  = z * (1-zplus)
                zsn = nd.sum(zn, axis=1) + nd.expand_dims(bias * (bias < 0), axis=0) - default_stabilizer
                rx = nd.sum(zn * nd.expand_dims(ry/zsn, axis=1), axis=2)

        else:
            print('Error in Dense layer: unknown LRP type {}'.format(self.lrp_type))

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

        if self.lrp_type == 'simple':
            # simple-LRP
            zs = y / self.normalizer_value + 1e-16*((y >= 0)*2 - 1.)
            F = ry/zs
            rx = x * nd.Deconvolution(F, self.weight, bias=None, kernel=self.kernel, stride=self.stride, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], no_bias=True)

        elif self.lrp_type == 'epsilon' or self.lrp_type == 'eps':
            # simple-LRP
            zs = y / self.normalizer_value + self.lrp_param*((y >= 0)*2 - 1.)
            F = ry/zs
            rx = x * nd.Deconvolution(F, self.weight, bias=None, kernel=self.kernel, stride=self.stride, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], no_bias=True)

        else:
            print('Error in SumPool layer: unknown LRP type {}'.format(self.lrp_type))

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

    def __init__(self, kernel, stride, lrp_type, lrp_param, method='grad_n_sum'):
        self.kernel    = kernel
        self.stride    = stride
        self.pad       = [0,0]

        self.lrp_type = lrp_type
        self.lrp_param = lrp_param

        self.method = method

    def forward(self, is_train, req, in_data, out_data, aux):
        x      = in_data[0]
        y = nd.Pooling(x, kernel=self.kernel, stride=self.stride, pool_type = 'max', pad=self.pad)

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

        if self.method == 'loops':
            # get input flags

            if self.lrp_type == 'simple' or self.lrp_type == 'epsilon' or self.lrp_type == 'eps' or self.lrp_type == 'alphabeta': # since maxpool only has equal dominant activations
                rx = nd.zeros_like(x,dtype=x.dtype)
                for i in range(Hout):
                    for j in range(Wout):
                        Z = y[:,:,i:i+1,j:j+1] == x[:,:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool]
                        Zs = Z.sum(axis=(2,3),keepdims=True) #thanks user wodtko for reporting this bug/fix
                        rx[:,:,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool] += (Z / Zs) * ry[:,:,i:i+1,j:j+1]

            else:
                print('Error in MaxPool layer: unsupported LRP type {}'.format(self.lrp_type))

        elif self.method == 'grad_n_sum':
            # approach: detect which inputs are maximal, set the rest to zero and then treat it as if it was a sumpool layer
            # in the case of one max value, their behaviour is the same, if there are multiple, redistribute to all of them.


            if self.lrp_type == 'simple' or self.lrp_type == 'epsilon' or self.lrp_type == 'eps': # since maxpool only has equal dominant activations

                # get flags whether the input contributed to the output via the gradient
                x.attach_grad()
                with autograd.record():
                    layer_output = nd.Pooling(x, kernel=self.kernel, stride=self.stride, pool_type = 'max', pad=self.pad)
                input_gradient = autograd.grad(layer_output, x, head_grads=None, retain_graph=None, create_graph=False, train_mode=False)[0]

                # get sums in output as conv of input flags
                out_sums = nd.Pooling(input_gradient, kernel=self.kernel, stride=self.stride, pool_type = 'sum', pad=self.pad)
                # divide relevance by output sums

                # print(out_sums)

                init_rel = ry / out_sums

                # print(ry)

                # get lrp as backprop to input
                with autograd.record():
                    layer_output = nd.Pooling(x, kernel=self.kernel, stride=self.stride, pool_type = 'max', pad=self.pad)
                rx = autograd.grad(layer_output, x, head_grads=init_rel, retain_graph=None, create_graph=False, train_mode=False)[0]

                # print(rx)

            else:
                print('Error in MaxPool layer: unsupported LRP type {}'.format(self.lrp_type))

        else:
            print('Error in MaxPool layer: backward method -{}- unknown!'.format(self.method))

        self.assign(in_grad[0], req[0], rx)

@mx.operator.register("maxpool_lrp")  # register with name "maxpool_lrp"
class MaxPoolLRPProp(mx.operator.CustomOpProp):
    def __init__(self, kernel, stride, lrp_type, lrp_param, maxpool_method='grad_n_sum'):
        super(MaxPoolLRPProp, self).__init__(True)
        self.kernel    = eval(kernel)
        self.stride    = eval(stride)
        self.pad       = [0,0]
        self.lrp_type  = lrp_type
        self.lrp_param = eval(lrp_param)
        self.maxpool_method = maxpool_method

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
        return MaxPoolLRP(self.kernel, self.stride, self.lrp_type, self.lrp_param, method=self.maxpool_method)
