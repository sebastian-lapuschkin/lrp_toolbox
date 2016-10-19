#from modules import SumPooling, MaxPooling
import numpy as np


# TODO max pooling layer testing

#forward test

#relevance allocation / backward test (same)





# TODO #sum pooling layer testing

#relevance allocation / backward test (same)



# tensordot playground

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

