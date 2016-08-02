'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.0
@copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
@license : BSD-2-Clause
'''

import numpy as np ; na = np.newaxis



# -------------------------------
# Modules for the neural network
# -------------------------------
class Module:
	'''
	Superclass for all computation layer implementations
	'''
	
	def __init__(self): pass
	def update(self, lrate): pass
	def clean(self): pass
	def lrp(self,R,lrp_var=None,param=0): return R
	def backward(self,DY): return DY
	def train(self, X, Y, batchsize, iters, lrate, status, shuffle_data): pass
	def forward(self,X): return X


# -------------------------------
# Linear layer
# -------------------------------
class Linear(Module):
	'''
	Linear Layer
	'''

	def __init__(self,m,n):
		'''
		Initiates an instance of a linear computation layer.
		
		Parameters
		----------
		m : int
			input dimensionality
		n : int
			output dimensionality
			
		Returns
		-------
		the newly created object instance
		'''
		
		self.m = m
		self.n = n
		self.B = np.zeros([self.n])
		self.W = np.random.normal(0,1.0*m**(-.5),[self.m,self.n])


	def forward(self,X):
		self.X = X
		self.Y = np.dot(X,self.W)+self.B
		return self.Y

	
	def lrp(self,R, lrp_var=None,param=0):
		'''
		performs LRP by calling subroutines, depending on lrp_var and param
		
		Parameters
		----------
		
		R : numpy.ndarray
			relevance input for LRP.
			should be of the same shape as the previously produced output by Linear.forward
		
		lrp_var : str
			either 'none' or 'simple' or None for standard Lrp ,
			'epsilon' for an added epsilon slack in the denominator
			'alphabeta' for weighting positive and negative contributions separately. param specifies alpha with alpha + beat = 1
		
		param : double
			the respective parameter for the lrp method of choice
		
		Returns
		-------
		R : the backward-propagated relevance scores.
			shaped identically to the previously processed inputs in Linear.forward		
		'''

		if lrp_var is None or lrp_var.lower() == 'none' or lrp_var.lower() == 'simple':
			return self._simple_lrp(R)
		elif lrp_var.lower() == 'epsilon':
			return self._epsilon_lrp(R,param)
		elif lrp_var.lower() == 'alphabeta':
			return self._alphabeta_lrp(R,param)
		else:
			print 'Unknown lrp variant', lrp_var
		

	def _simple_lrp(self,R):
		'''
		LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
		'''
		Z = self.W[na,:,:]*self.X[:,:,na] #localized preactivations
		Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] #preactivations
		return ((Z / Zs) * R[:,na,:]).sum(axis=2)

	def _epsilon_lrp(self,R,epsilon):
		'''
		LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
		'''
		Z = self.W[na,:,:]*self.X[:,:,na] # localized preactivations
		Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] # preactivations
		
		# add slack to denominator. we require sign(0) = 1. since np.sign(0) = 0 would defeat the purpose of the numeric stabilizer we do not use it.
		Zs += epsilon * ((Zs >= 0)*2-1)
		return ((Z / Zs) * R[:,na,:]).sum(axis=2)

	def _alphabeta_lrp(self,R,alpha):
		'''
		LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
		'''
		beta = 1 - alpha
		Z = self.W[na,:,:]*self.X[:,:,na] # localized preactivations
		
		Zp = Z * (Z > 0);
		Zsp = Zp.sum(axis=1)[:,na,:] + (self.B * (self.B > 0))[na,na,:]
		
		Zn = Z * (Z < 0)
		Zsn = Zn.sum(axis=1)[:,na,:] + (self.B * (self.B < 0))[na,na,:]
		
		return alpha * ((Zp / Zsp) * R[:,na,:]).sum(axis=2) + beta * ((Zn / Zsn) * R[:,na,:]).sum(axis=2)
		
		
	def backward(self,DY):
		self.dW = np.dot(self.X.T,DY)
		self.dB = DY.sum(axis=0)
		return np.dot(DY,self.W.T)*self.m**.5/self.n**.5
	
	
	def update(self, lrate):
		self.W -= lrate*self.dW/self.m**.5
		self.B -= lrate*self.dB/self.m**.25	


	def clean(self):
		self.X = None
		self.Y = None
		self.dW = None
		self.dB = None


# -------------------------------
# Tanh layer
# -------------------------------
class Tanh(Module):
	'''
	Tanh Layer
	'''

	def forward(self,X):
		self.Y = np.tanh(X)
		return self.Y
	

	def backward(self,DY):
		return DY*(1.0-self.Y**2)


	def clean(self):
		self.Y = None

# -------------------------------
# Rectification layer
# -------------------------------
class Rect(Module):
	'''
	Rectification Layer
	'''

	def forward(self,X):
		self.Y = np.maximum(0,X)
		return self.Y
	
	
	def backward(self,DY):
		return DY*(self.Y!=0)
	

	def clean(self):
		self.Y = None


# -------------------------------
# Softmax layer
# -------------------------------
class SoftMax(Module):
	'''
	Softmax Layer
	'''
	
	def forward(self,X):
		self.X = X
		self.Y = np.exp(X) / np.exp(X).sum(axis=1)[:,na]
		return self.Y


	def lrp(self,R,lrp_var,param):
		return R*self.X 


	def clean(self):
		self.X = None
		self.Y = None




# -------------------------------
# Sequential layer
# -------------------------------   
class Sequential(Module):
	'''
	Top level access point and incorporation of the neural network implementation.
	Sequential manages a sequence of computational neural network modules and passes
	along in- and outputs.
	'''

	def __init__(self,modules):
		'''
		Constructor
				
		Parameters
		----------
		modules : list, tuple, etc. enumerable.
			an enumerable collection of instances of class Module
		'''
		self.modules = modules

	def forward(self,X):
		'''
		Realizes the forward pass of an input through the net
				
		Parameters
		----------
		X : numpy.ndarray
			a network input.
		
		Returns
		-------
		X : numpy.ndarray
			the output of the network's final layer
		'''
		
		for m in self.modules:
			X = m.forward(X)
		return X



	def lrp(self,R,lrp_var=None,param=0):
		'''
		Performs LRP using the network and temporary data produced by a forward call				
				
		Parameters
		----------
		R : numpy.ndarray
			final layer relevance values. usually the network's prediction of some data points
			for which the output relevance is to be computed
			dimensionality should be equal to the previously computed predictions
			
		lrpvar : str
			either 'none' or 'simple' or None for standard Lrp ,
			'epsilon' for an added epsilon slack in the denominator
			'alphabeta' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1
		
		param : double
			the respective parameter for the lrp method of choice
		
		Returns
		-------
		
		R : numpy.ndarray
			the first layer relevances as produced by the neural net wrt to the previously forward
			passed input data. dimensionality is equal to the previously into forward entered input data
		
		Note
		----
		
		Requires the net to be populated with temporary variables, i.e. forward needed to be called with the input
		for which the explanation is to be computed. calling clean in between forward and lrp invalidates the
		temporary data
		'''
	
		for m in self.modules[::-1]:
			R = m.lrp(R,lrp_var,param)
		return R


	

	def train(self, X, Y,  Xval = [], Yval = [],  batchsize = 25, iters = 10000, lrate = 0.005, status = 250, shuffle_data = True):
		''' 		
			X the training data
			Y the training labels
			
			Xval some validation data
			Yval the validation data labels
			
			batchsize the batch size to use for training
			iters max number of training iterations . TODO: introduce convergence criterion
			lrate the learning rate
			status number of iterations of silent training until status print and evaluation on validation data.
			shuffle_data permute data order prior to training
		'''
		
		if Xval == [] or Yval ==[]:
			Xval = X
			Yval = Y
		
		N,D = X.shape
		if shuffle_data:
			r = np.random.permutation(N)
			X = X[r,:]
			Y = Y[r,:]
			
		for i in xrange(iters):
			samples = np.mod(np.arange(i,i+batchsize),N)
			Ypred = self.forward(X[samples,:])
			self.backward(Ypred - Y[samples,:])
			self.update(lrate)
			
			if i % status == 0:
				Ypred = self.forward(Xval)
				acc = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Yval, axis=1))
				print 'Accuracy after {0} iterations: {1}%'.format(i,acc*100) 
		
		
		
		
		
		
	def backward(self,DY):
		for m in self.modules[::-1]:
			DY = m.backward(DY)
		return DY
	
	def update(self,lrate):
		for m in self.modules: m.update(lrate)
	
	def clean(self):
		'''
		Removes temporary variables from all network layers.
		'''
		for m in self.modules: m.clean()


