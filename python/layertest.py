
from modules import SumPool, MaxPool, Convolution, Flatten
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
    print 'MAX POOL FORWARD TEST:'
    print 'x.shape', x.shape
    print 'pool', pool
    print 'stride', stride
    print 'e.shape', e.shape

    M = MaxPool(pool=pool, stride=stride)
    y =  M.forward(x)

    print 'y.shape', y.shape
    print y
    print e
    print 'y == e :', np.all(e == y)
    assert(np.all(e == y))

def maxpoolRtest(x,Rin,Rex,pool,stride):

    print ''
    print 'MAX POOL RELEVANCE TEST'
    print 'x.shape', x.shape
    print 'pool', pool
    print 'stride', stride
    print 'Rin.shape', Rin.shape

    M = MaxPool(pool=pool, stride=stride)
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



if False:

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


print '# ----------------------'
print '# Sum Pooling Layer Test'
print '# ----------------------'

if False:
    def sumpooltest(x,e,pool,stride):
        print ''
        print 'SUM POOL FORWARD TEST'
        print 'x.shape', x.shape
        print 'pool', pool
        print 'stride', stride
        print 'e.shape', e.shape

        S = SumPool(pool=pool, stride=stride)
        y =  S.forward(x)

        print 'y.shape', y.shape
        print 'y', y
        print 'e', e
        print 'y == e :', np.all(e == y)
        assert(np.all(e == y))

    def sumpoolRtest(x,Rin,Rex,pool,stride):
        print ''
        print 'SUM POOL RELEVANCE TEST'
        print 'x.shape', x.shape
        print 'pool', pool
        print 'stride', stride
        print 'Rin.shape', Rin.shape

        S = SumPool(pool=pool, stride=stride)
        y = S.forward(x)
        R = S.lrp(Rin)

        print 'x.shape', x.shape
        print 'R.shape', R.shape
        print 'Rex.shape', Rex.shape
        print 'x', x
        print 'R',R
        print 'Rex', Rex
        print 'R ~ Rex :', np.all(np.abs(R - Rex) <= 1e-10), '(tolerance = 1e-10)'
        if not np.all(np.abs(R - Rex) <= 1e-10):
            print 'delta (R - Rex):', R - Rex
        assert(np.all(np.abs(R - Rex) <= 1e-10))


    #construct filter and inputs.
    a = np.array([[  [1,1,1,1],\
                    [1,1,2,1],\
                    [0,0,0,0],\
                    [0,0,0,0]
                ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
    a = a.astype(np.float)

    b = np.array([[  [1,1,0,0],\
                    [1,1,0,0],\
                    [1,1,0,0],\
                    [2,1,0,0]
                ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
    b  = b.astype(np.float)

    c = np.array([[  [1,1,1,1],\
                    [1,1,2,0],\
                    [1,1,0,0],\
                    [1,0,0,0]
                ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
    c = c.astype(np.float)

    #construct multiple 1-layer input data points
    x = np.concatenate((a,b,c),axis = 0)
    expected = [
        [[4,5],[0,0]],\
        [[4,0],[5,0]],\
        [[4,4],[3,0]]
    ]
    expected = np.array(expected)[...,None]

    sumpooltest(x,expected,pool = (2,2),stride = (2,2))


    x2 = np.concatenate((a,-b,c),axis = 3) # 1 x 4 x 4 x 3
    e2 = np.zeros([1,2,2,3])
    e2[0,:,:,0] = np.array([[7,7],[4,4]])
    e2[0,:,:,1] = np.array([[-6,-3],[-7,-3]])
    e2[0,:,:,2] = np.array([[9,7],[7,4]])

    sumpooltest(x2,e2,pool=(3,3),stride=(1,1))


    x3 = x2
    e3 = np.zeros([1,1,1,3])
    e3[0,:,:,0] = np.array([[[9]]])
    e3[0,:,:,1] = np.array([[[-9]]])
    e3[0,:,:,2] = np.array([[[11]]])

    sumpooltest(x3,e3,pool=(4,4),stride=(1,1))


    #testing relevance allocation
    Rin = np.ones(e3.shape)
    Rex = x3.copy().astype(np.float)
    Rex[...,0] *= 1./Rex[...,0].sum()
    Rex[...,1] *= 1./Rex[...,1].sum()
    Rex[...,2] *= 1./Rex[...,2].sum()
    sumpoolRtest(x3,Rin,Rex,pool=(4,4),stride=(4,4))


    xa = a #note that this case - adding relevance to non-firing inputs-would never happen. (e.g. the relevance would not be there in the first place.)
    Rina = np.array([[1.,1.],[1.,1.]])
    Rina = np.reshape(Rina, [1,2,2,1])
    Rexa = np.array([[[0.25,0.25,.2,.2],\
                    [0.25,0.25,.4,.2],\
                    [.0,.0,.0,.0],\
                    [.0,.0,.0,.0]
                ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D

    sumpoolRtest(xa,Rina,Rexa,pool=(2,2),stride=(2,2))

    xa = a
    Rina = np.array([[1.,1.],[0.,0.]])
    Rina = np.reshape(Rina, [1,2,2,1])
    Rexa = np.array([[[0.25,0.25,.2,.2],\
                    [0.25,0.25,.4,.2],\
                    [.0,.0,.0,.0],\
                    [.0,.0,.0,.0]
                ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D

    sumpoolRtest(xa,Rina,Rexa,pool=(2,2),stride=(2,2))

    #overlapping 3x3 relevance pooling test with three layers

    stride = (1,1)
    pool = (3,3)

    xa = a
    Rina = np.array([[1.,1.],[1.,1.]]).reshape([1,2,2,1])
    v = 1./4 ; s = 1./7
    Rexa = np.array([[[s,2*s,2*s,s],\
                    [s+v,2*s+2*v,4*s+4*v,s+v],\
                    [.0,.0,.0,.0],\
                    [.0,.0,.0,.0]
                ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D

    sumpoolRtest(xa,Rina,Rexa,pool=pool,stride=stride)


    xb = b
    Rinb = np.array([[1.,1.],[1.,1.]]).reshape([1,2,2,1])
    s = 1./6 ; d = 1./3 ; z = 1./7
    Rexb = np.array([[[s,s+d,0,0],\
                    [s+z,s+2*d+z,0,0],\
                    [s+z,s+2*d+z,.0,.0],\
                    [2*z,z+d,.0,.0]
                ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
    sumpoolRtest(xb,Rinb,Rexb,pool=pool,stride=stride)


    x = np.concatenate((xa,xb),axis=0)
    Rin = np.concatenate((Rina, Rinb),axis=0)
    Rex = np.concatenate((Rexa, Rexb),axis = 0)
    sumpoolRtest(x,Rin,Rex,pool=pool,stride=stride)



print '# ----------------------'
print '# Convolution Layer Test'
print '# ----------------------'

def convolutiontest(x,e,filter,stride):
    print ''
    print 'CONVOLUTION FORWARD TEST'
    print 'x.shape', x.shape
    print 'x', x
    print 'filter', filter
    print 'stride', stride
    print 'e.shape', e.shape

    C = Convolution(filtersize = filter.shape, stride=stride)
    C.W = filter
    y =  C.forward(x)

    print 'y.shape', y.shape
    print 'y', y
    print 'e', e
    print 'y == e :', np.all(np.abs(e - y) <= 1e-10), '(tolerance = 1e-20)'
    if not np.all(np.abs(e - y) <= 1e-10):
        print 'delta (e - y):', e - y
    assert(np.all(np.abs(e - y) <= 1e-10))

def convolutionRtest(x,Rin,Rex,filter,stride):
    print ''
    print 'CONVOLUTION RELEVANCE TEST'
    print 'x.shape', x.shape
    print 'filter', filter
    print 'stride', stride
    print 'Rin.shape', Rin.shape

    C = Convolution(filtersize=filter.shape, stride=stride)
    C.W = filter
    y = C.forward(x)
    R = C.lrp(Rin)

    print 'x.shape', x.shape
    print 'R.shape', R.shape
    print 'Rex.shape', Rex.shape
    print 'x', x
    print 'R',R
    print 'Rex', Rex
    print 'R ~ Rex :', np.all(np.abs(R - Rex) <= 1e-10), '(tolerance = 1e-10)'
    if not np.all(np.abs(R - Rex) <= 1e-10):
        print 'delta (R - Rex):', R - Rex
    assert(np.all(np.abs(R - Rex) <= 1e-10))


#construct filter and inputs.
a = np.array([[  [1,1,1,1],\
                 [1,1,2,1],\
                 [0,0,0,0],\
                 [0,0,0,0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
a = a.astype(np.float)

b = np.array([[  [1,1,0,0],\
                 [1,1,0,0],\
                 [1,1,0,0],\
                 [2,1,0,0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
b  = b.astype(np.float)

c = np.array([[  [1,1,1,1],\
                 [1,1,2,0],\
                 [1,1,0,0],\
                 [1,0,0,0]
            ]])[...,None] # 1 x 4 x 4 x 1 = N x H x W x D
c = c.astype(np.float)

fa = a[0,...,None]
fb = b[0,...,None]*0.5
fc = -1.*c[0,...,None]

yaa = np.tensordot(a,fa,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(a,yaa,fa,stride=(1,1))

yab = np.tensordot(a,fb,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(a,yab,fb,stride=(1,1))

yac = np.tensordot(a,fc,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(a,yac,fc,stride=(1,1))

yba = np.tensordot(b,fa,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(b,yba,fa,stride=(1,1))

ybb = np.tensordot(b,fb,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(b,ybb,fb,stride=(1,1))

ybc = np.tensordot(b,fc,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(b,ybc,fc,stride=(1,1))

yca = np.tensordot(c,fa,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(c,yca,fa,stride=(1,1))

ycb = np.tensordot(c,fb,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(c,ycb,fb,stride=(1,1))

ycc = np.tensordot(c,fc,axes = ([1,2,3],[0,1,2]))[None,...,None]
convolutiontest(c,ycc,fc,stride=(1,1))



#combine multiple filters and samples

x = np.concatenate((a,b,c),axis = 0)
f = np.concatenate((fa,fb,fc),axis = 3)

ea = np.concatenate((yaa,yab,yac),axis = 3)
eb = np.concatenate((yba,ybb,ybc),axis = 3)
ec = np.concatenate((yca,ycb,ycc),axis = 3)
e = np.concatenate((ea,eb,ec),axis = 0)
convolutiontest(x,e,f,stride=(1,1))


# test stride 2,2 on 2,2 filter on 2 inputs, then two filtes on one input, then lrp

f = np.array([[0.1,0.2],[0.2,0.1]])
f = np.reshape(f,[2,2,1,1])

stride = (2,2)
fsize = (2,2)

x = np.concatenate((b,c),axis = 0)
eb = [[0.6,0],[0.8,0]]
eb = np.reshape(np.array(eb),[1,2,2,1])
ec = [[.6,.7],[.5,0]]
ec = np.reshape(np.array(ec),[1,2,2,1])
e  = np.concatenate((eb,ec),axis=0)

Rin = np.ones_like(e)
s = 1./6 ; d = 2*s ; aa = 1./8 ; v = 2*aa
Rexb = [[s, d, 0, 0],\
        [d, s, 0, 0],\
        [aa, v, 0, 0],\
        [2*v,aa,0, 0]]
z = 1./7
g = 0.2
Rexc  = [[s,d,z,2*z],\
         [d,s,4*z,0],\
         [g,2*g,0,0],\
         [2*g,0,0,0]]

Rexb = np.reshape(np.array(Rexb),b.shape)
Rexc = np.reshape(np.array(Rexc),c.shape)
Rex = np.concatenate((Rexb,Rexc),axis = 0)

convolutiontest(x,e,f,stride=(2,2))
convolutionRtest(x,Rin,Rex,f,stride=(2,2))


f2 = np.array([[1,2],[-3,-4]])
f2 = np.reshape(f2,[2,2,1,1])
f3 = np.concatenate((f,f2),axis = 3)
x = a
ef=[[.6,.8],[0,0]]
ef = np.reshape(np.array(ef),[1,2,2,1])
ef2=[[-4,-7],[0,0]]
ef2 = np.reshape(np.array(ef2),[1,2,2,1])
e = np.concatenate((ef,ef2),axis=3)

Rinf= np.ones_like(ef)
Rexf = [[s,d,aa,v],\
        [d,s,2*v,aa],\
        [0,0,0,0],\
        [0,0,0,0]]

Rexf = np.array(Rexf)
Rexf = np.reshape(Rexf,[1,4,4,1])

Rinf2 = np.ones_like(ef2)
v = 0.25 ; z = 1./7
Rexf2 = [[-v, -2*v, -z, -2*z],\
         [3*v, 1  , 6*z, 4*z],\
         [0,0,0,0],\
         [0,0,0,0]]

Rexf2 = np.array(Rexf2)
Rexf2 = np.reshape(Rexf2,[1,4,4,1])

convolutionRtest(x,Rinf,Rexf,f,stride=(2,2))
convolutionRtest(x,Rinf2,Rexf2,f2,stride=(2,2))

Rin = np.concatenate((Rinf, Rinf2),axis=3)
Rex = Rexf + Rexf2 #after computing relevace contribution for each filter, aggregate at the input again by summation.

convolutiontest(x,e,f3,stride=(2,2))
convolutionRtest(x,Rin,Rex,f3,stride=(2,2))


x = a
f = np.ones((3,3,1,1)) ; f[1,1,...] = 2
stride = (1,1)
Rin = np.ones((1,2,2,1))
Rex11 = [[aa,aa,aa,0],\
         [aa,v,v,0],\
         [0,0,0,0],\
         [0,0,0,0]]
n = 1./9
Rex12 = [[0,n,n,n],\
         [0,n,4*n,n],\
         [0,0,0,0],\
         [0,0,0,0]]

Rex21 = [[0,0,0,0],\
         [v,v,2*v,0],\
         [0,0,0,0],\
         [0,0,0,0]]

Rex22 = [[0,0,0,0],\
         [0,v,2*v,v],\
         [0,0,0,0],\
         [0,0,0,0]]

Rex = np.array(Rex11) + np.array(Rex21) + np.array(Rex12) + np.array(Rex22)
Rex = np.reshape(Rex,x.shape)
convolutionRtest(x,Rin,Rex,f,stride)