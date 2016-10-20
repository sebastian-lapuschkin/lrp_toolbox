
from modules import SumPooling, MaxPooling, Convolution, Flatten
import numpy as np

# -----------------------------------------
print '#----------------------------------'
print '# SOME DATA + TENSORFLOT PLAYGROUND'
print '#----------------------------------'

def tensordottest():
    a = np.array([  [1,1,1,1],\
                    [1,1,1,1],\
                    [0,0,0,0],\
                    [0,0,0,0]
                ])

    b = np.array([  [1,1,0,0],\
                    [1,1,0,0],\
                    [1,1,0,0],\
                    [1,1,0,0]
                ])

    c = np.array([  [1,1,1,1],\
                    [1,1,1,0],\
                    [1,1,0,0],\
                    [1,0,0,0]])

    x = np.array([a,b,c])
    f = np.array(b)

    print x.shape
    print f.shape

    y = np.tensordot(x,f,axes = ([1,2],[0,1]))
    print 'expecting [4, 8, 7] :\n', y,'\n'

    x = np.array([a[...,None],b[...,None],c[...,None]]) #  layer 0 concatenation to create a mini batch of items of depth 1
    print x.shape # 3 4 4 1 : 3 inputs of shape 4 x 4 x 1

    f = np.concatenate((b[...,None,None],a[...,None,None]),axis=3)
    print f.shape # 4 4 1 2 : filters shaped 4 x 4 x 1, 2 of those.

    y = np.tensordot(x,f,axes = ([1,2,3],[0,1,2])) #convolutions for each filter on each input. should produce 2 outputs for each of the 3 inputs

    print y.shape # 3 2
    print 'expecting [[4 8],[8 4],[7 7]] :\n', y, '\n'
    print 'we actually need a 4-axis output (3,1,1,2) again. how to achieve that?'

    z = np.zeros([3,1,1,2]) #try just filling in the values.
    b = np.random.normal(0,0.001,(2))
    z[:,0,0,:] = y + b
    print z #this seems to work out. nice.
    print 'b=', b

#tensordottest()

# -----------------------------
print '# ---------------------'
print '# Flattening Layer Test'
print '# ---------------------'

def flattentest(x):
    print ''
    print 'x.shape:', x.shape
    F = Flatten()
    y = F.forward(x)
    print 'y.shape:', y.shape
    xr = F.backward(y)
    print 'xr.shape:', xr.shape, 'xr == x', np.all(x == xr)

#flattentest(....)

# ------------------------------
print '# ----------------------'
print '# Max Pooling Layer Test'
print '# ----------------------'

def maxpooltest(x,e,pool,stride):
    print ''
    print 'x.shape', x.shape
    print 'pool', pool
    print 'stride', stride
    print 'e.shape', e.shape

    M = MaxPooling(pool=pool, stride=stride)
    y =  M.forward(x)

    print 'y.shape', y.shape
    print y
    print e
    print 'y == e :', np.all(e == y)
    assert(np.all(e == y))

def maxpoolRtest(x,Rin,Rex,pool,stride):
    print ''
    print 'x.shape', x.shape
    print 'pool', pool
    print 'stride', stride
    print 'Rin.shape', Rin.shape

    M = MaxPooling(pool=pool, stride=stride)
    y = M.forward(x)
    R = M.lrp(Rin)

    print 'x.shape', x.shape
    print 'R.shape', R.shape
    print 'Rex.shape', Rex.shape
    print 'x', x
    print 'R',R
    print 'Rex', Rex
    print 'R == Rex :', np.all(R == Rex)
    assert(np.all(R == Rex))

#construct filter and inputs.
a = np.array([[  [1,1,1,1],\
                 [1,1,2,1],\
                 [0,0,0,0],\
                 [0,0,0,0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
a.astype(np.float)

b = np.array([[  [1,1,0,0],\
                 [1,1,0,0],\
                 [1,1,0,0],\
                 [2,1,0,0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
b.astype(np.float)

c = np.array([[  [1,1,1,1],\
                 [1,1,2,0],\
                 [1,1,0,0],\
                 [1,0,0,0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
c.astype(np.float)

#construct multiple 1-layer input data points
x = np.concatenate((a,b,c),axis = 0)
expected = [
    [[1,2],[0,0]],\
    [[1,0],[2,0]],\
    [[1,2],[1,0]]
 ]
expected = np.array(expected)[...,None]

maxpooltest(x,expected,pool = (2,2),stride = (2,2))

x2 = np.concatenate((a,b,c),axis = 3) # 1 x 4 x 4 x 3
e2 = np.zeros([1,2,2,3])
e2[0,:,:,0] = np.array([[2,2],[2,2]])
e2[0,:,:,1] = np.array([[1,1],[2,1]])
e2[0,:,:,2] = np.array([[2,2],[2,2]])

maxpooltest(x2,e2,pool=(3,3),stride=(1,1))

x3 = x2
e3 = np.zeros([1,1,1,3])
e3[0,:,:,0] = np.array([[[2]]])
e3[0,:,:,1] = np.array([[[2]]])
e3[0,:,:,2] = np.array([[[2]]])

maxpooltest(x3,e3,pool=(4,4),stride=(1,1))


#testing relevance allocation
Rin = np.ones_like(e3)
Rex = (x2 == 2) * 1.0
maxpoolRtest(x3,Rin,Rex,pool=(4,4),stride=(4,4))

xa = a
Rina = np.array([[1.,1.],[1.,1.]])
Rina = np.reshape(Rina, [1,2,2,1])
Rexa = np.array([[[0.25,0.25,0,0],\
                 [0.25,0.25,1,0],\
                 [.25,.25,.25,.25],\
                 [.25,.25,.25,.25]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D

maxpoolRtest(xa,Rina,Rexa,pool=(2,2),stride=(2,2))

xa = a
Rina = np.array([[1.,1.],[0.,0.]])
Rina = np.reshape(Rina, [1,2,2,1])
Rexa = np.array([[[0.25,0.25,0,0],\
                 [0.25,0.25,1,0],\
                 [.0,.0,.0,.0],\
                 [.0,.0,.0,.0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D

maxpoolRtest(xa,Rina,Rexa,pool=(2,2),stride=(2,2))

#overlapping 3x3 relevance pooling test with three layers

stride = (1,1)
pool = (3,3)

xa = a
Rina = np.array([[1.,1.],[1.,1.]]).reshape([1,2,2,1])
Rexa = np.array([[[0,0,0,0],\
                 [0,0,4,0],\
                 [.0,.0,.0,.0],\
                 [.0,.0,.0,.0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D

maxpoolRtest(xa,Rina,Rexa,pool=pool,stride=stride)

xb = b
Rinb = np.array([[1.,1.],[1.,1.]]).reshape([1,2,2,1])
s = 1./6 ; d = 1./3
Rexb = np.array([[[s,s+d,0,0],\
                 [s,s+d+d,0,0],\
                 [s,s+d+d,.0,.0],\
                 [1,d,.0,.0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
maxpoolRtest(xb,Rinb,Rexb,pool=pool,stride=stride)

x = np.concatenate((xa,xb),axis=0)
Rin = np.concatenate((Rina, Rinb),axis=0)
Rex = np.concatenate((Rexa, Rexb),axis = 0)
maxpoolRtest(x,Rin,Rex,pool=pool,stride=stride)
# ----------------------
# Sum Pooling Layer Test
# ----------------------



# ----------------------
# Convolution Layer Test
# ----------------------


print 'done'