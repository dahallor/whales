from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
import random
import math

EPSILON = 10**-7

class Layer(ABC):
	def __init__(self):
		self.__prevIn = []
		self.__prevOut = []

	def setPrevIn(self, dataIn):
		self.__prevIn = dataIn

	def setPrevOut(self, out):
		self.__prevOut = out

	def getPrevIn(self):
		return self.__prevIn

	def getPrevOut(self):
		return self.__prevOut

	def backward(self, gradIn, eta = 0.0001):
		return self.gradient()*gradIn

	@abstractmethod
	def forward(self, dataIn):
		pass

	@abstractmethod
	def gradient(self):
		pass

class InputLayer(Layer):
	def __init__(self, dataIn):
		super().__init__()
		self.meanX = []
		self.stdX = []

		# generate row vector for mean
		self.meanX  = np.mean(dataIn, axis=0)

		for i in range(len(dataIn[0])):
			self.stdX.append(np.std(dataIn[:, i]))

	def forward(self, X):
		super().setPrevIn(X)

		zScoredMatrix = stats.zscore(X, axis=1)

		super().setPrevOut(zScoredMatrix)

		return zScoredMatrix

	def gradient(self):
		pass

class FullyConnectedLayer(Layer):
	def __init__(self, featuresIn, featuresOut):
		super().__init__()
		self.featuresIn =  featuresIn
		self.featuresOut = featuresOut
		self.weights = np.random.rand(featuresIn, featuresOut) * 10.0**-4.0
		self.biases = np.random.rand(1, featuresOut) * 10.0**-4.0

	def getWeights(self):
		return self.weights

	def setWeights(self, weights):
		self.weights = weights

	def getBias(self):
		return self.biases

	def setBias(self, biases):
		self.biases = biases

	def forward(self, dataIn):
		super().setPrevIn(dataIn)
		result = (dataIn @ self.weights) + self.biases
		super().setPrevOut(result)
		return result

	def backward(self, gradIn, eta = 0.0001):
		return self.updateWeights(gradIn, eta)

	# mult learning_rate by result
	# def get_weight_adam(self, gradient, t):
	# 	rho1 = .9
	# 	rho2 = .999
	# 	delta = .00000001

	# 	self.ws = rho1 * self.ws + (1-rho1) * gradient
	# 	self.wr = rho2 * self.wr + (1-rho2) * (gradient * gradient)
	# 	combined = (self.ws/(1- rho1 ** t))/(np.sqrt(self.wr/(1-math.pow(rho2, t))) + delta)
	# 	return combined

	# # mult learning_rate by result
	# def get_bias_adam(self, gradient, t):
	# 	rho1 = .9
	# 	rho2 = .999
	# 	delta = .00000001

	# 	self.bs = rho1 * self.bs + (1-rho1) * gradient
	# 	self.br = rho2 * self.br + (1-rho2) * (gradient * gradient)
	# 	combined = (self.bs/(1- rho1 ** t))/(np.sqrt(self.br/(1-math.pow(rho2, t))) + delta)
	# 	return combined

	def updateWeights(self, gradIn, eta):
		gradient_weights = self.getPrevIn().T @ gradIn
		gradient_bias = np.ones((1,self.getPrevIn().shape[0])) @ gradIn

		# weight_adam = self.get_weight_adam(gradient_weights, epoch)
		# bias_adam = self.get_bias_adam(gradient_bias, epoch)
		# weight_eta = (eta*weight_adam)
		# bias_eta = (eta*bias_adam)

		self.weights -= eta/ self.getPrevIn().shape[0] * gradient_weights
		self.biases -= eta/self.getPrevIn().shape[0] * gradient_bias

		return self.getPrevOut() @ self.gradient()

	def gradient(self):
		return self.weights.T

class ObjectiveLayer(ABC):
	@abstractmethod
	def eval(self, y, yhat):
		pass

	@abstractmethod
	def gradient(self, y, yhat):
		pass
		
class LeastSquares(ObjectiveLayer):
	def eval(self, y, yhat):
		return (y - yhat) * (y - yhat)

	def gradient(self, y, yhat):
		return 2*(y-yhat) #TODO check

class LogLoss(ObjectiveLayer):
	def eval(self, y, yhat):
		return -((y*np.log(yhat + EPSILON)) + ((1-y) * np.log(1-yhat + EPSILON)))

	def gradient(self, y, yhat):
		return -1* (y - yhat)/(yhat*(1-yhat) + EPSILON)

class CrossEntropy(ObjectiveLayer):
	def eval(self, y, yhat):
		yhat = np.clip(yhat, 1e-15, 1 - 1e-15)
		return - y * np.log(yhat) - (1 - y) * np.log(1 - yhat)

	def gradient(self, y, yhat):
		return -(y/(yhat + EPSILON))