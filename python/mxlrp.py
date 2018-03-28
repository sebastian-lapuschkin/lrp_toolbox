import mxnet as mx
from mxnet import nd

## ######################### ##
# LRP-PATCHED HYBRID_FORWARDS #
## ######################### ##


def dense_hybrid_forward_lrp(self, F, x, weight, bias=None):
        act = F.Custom(x, weight, bias, op_type='dense_lrp')

        if self.act is not None:
            act = F.BlockGrad(self.act(act) - act) + act
        return act


def convolution_hybrid_forward_lrp(self, F, x, weight, bias=None):

        print(self._kwargs)

        kernel, stride, dilate, pad, num_filter, num_group, no_bias, layout = map(self._kwargs.get, ['kernel', 'stride', 'dilate', 'pad', 'num_filter', 'num_group', 'no_bias', 'layout'])

        act = F.Custom(x, weight, bias, kernel=kernel, stride=stride, dilate=dilate, pad=pad, num_filter=num_filter, num_group=num_group, no_bias=no_bias, layout=layout, op_type='conv_lrp')

        if self.act is not None:
            act = F.BlockGrad(self.act(act) - act) + act
        return act

## ############### ##
# CONVOLUTION LAYER #
## ############### ##
class ConvLRP(mx.operator.CustomOp):

    def __init__(self, kernel, stride, dilate, pad, num_filter, num_group, no_bias, layout):
        self.kernel     = kernel
        self.stride     = stride
        self.dilate     = dilate
        self.pad        = pad
        self.num_filter = num_filter
        self.num_group  = num_group
        self.no_bias    = no_bias
        self.layout     = layout

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

        print(F.shape)

        rx = x * nd.Deconvolution(F, weight, bias=None, kernel=self.kernel, stride=self.stride, dilate=self.dilate, pad=self.pad, target_shape= (x.shape[2], x.shape[3]), num_filter=x.shape[1], num_group=self.num_group, no_bias=True, layout=self.layout)
        self.assign(in_grad[0], req[0], rx)

@mx.operator.register("conv_lrp")
class ConvLRPProp(mx.operator.CustomOpProp):
    def __init__(self, kernel, stride, dilate, pad, num_filter, num_group, no_bias, layout):
        super(ConvLRPProp, self).__init__(True)

        self.kernel     = eval(kernel)
        self.stride     = eval(stride)
        self.dilate     = eval(dilate)
        self.pad        = eval(pad)
        self.num_filter = eval(num_filter)
        self.num_group  = eval(num_group)
        self.no_bias    = eval(no_bias)
        self.layout     = layout

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
        w = (data_shape[3] + self.pad[1] - weight_shape[3] //  self.stride[1] + 1)

        output_shape = (data_shape[0], weight_shape[0], h, w)
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape, weight_shape, bias_shape), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return ConvLRP(self.kernel, self.stride, self.dilate, self.pad, self.num_filter, self.num_group, self.no_bias, self.layout)

## ######### ##
# DENSE LAYER #
## ######### ##
class DenseLRP(mx.operator.CustomOp):

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
    def __init__(self):
        super(DenseLRPProp, self).__init__(True)

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
        return DenseLRP()

class DenseLRPBlock(mx.gluon.Block):
    def __init__(self, in_channels, channels, **kwargs):
        super(DenseLRPBlock, self).__init__(**kwargs)
        self._bias  = self.params.get('bias', shape=(channels), init=weight_initializer)
        self.weight = self.params.get('weight', shape=(channels, in_channels))

    def forward(self, x):
        ctx = x.context
        return mx.nd.Custom(x, self.weight.data(ctx), self.bias.data(ctx), bias=self._bias, op_type='dense_lrp')
