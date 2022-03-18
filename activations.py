from layers import Layer
import numpy as np
import math

EPSILON = 10**-7

class LinearLayer(Layer):
	def __init__(self):
		super().__init__()

	def forward(self, dataIn):
		super().setPrevIn(dataIn)
		super().setPrevOut(dataIn)
		return dataIn

	def gradient(self):
		return np.ones(self.getPrevIn().shape)

class ReLuLayer(Layer):
	def __init__(self):
		super().__init__()
		self.relu = lambda x: max((0, x))
		self.drelu = lambda x: int(x >= 0)

	def forward(self, dataIn):
		super().setPrevIn(dataIn)
		result = np.vectorize(self.relu)(dataIn)
		super().setPrevOut(result)
		return result

	def gradient(self):
		output = self.getPrevIn()
		gradient = np.vectorize(self.drelu)(output)
		return gradient
		
class SigmoidLayer(Layer):
	def __init__(self):
		super().__init__()
		self.sigmoid = lambda x: 1/(1+np.exp(-x))
		self.dsigmoid = lambda x: self.sigmoid(x)*(1-self.sigmoid(x)+ EPSILON)

	def forward(self, dataIn):
		super().setPrevIn(dataIn)
		sigmoid = np.vectorize(self.sigmoid)(dataIn)
		super().setPrevOut(sigmoid)
		return sigmoid

	def gradient(self):
		output = self.getPrevOut()
		return np.vectorize(self.dsigmoid)(output)

class SoftmaxLayer(Layer):
	def __init__(self):
		super().__init__()

	def forward(self, dataIn):
		super().setPrevIn(dataIn)
		# Subtract max for numeric stability
		normalizedMax = np.exp(dataIn - np.max(dataIn))
		softMax = normalizedMax / normalizedMax.sum()
		super().setPrevOut(softMax)
		return softMax

	def gradient(self):
		output = self.getPrevOut()
		grad = self.forward(output)
		return grad * (1 - grad)

class TanhLayer(Layer):
	def __init__(self):
		super().__init__()
		self.tanh = lambda x: (math.e**x - math.e **-x)/(math.e**x + math.e **-x)
		self.dtanh = lambda x: (1-x**2) + EPSILON

	def forward(self, dataIn):
		super().setPrevIn(dataIn)
		result = np.vectorize(self.tanh)(dataIn)
		super().setPrevOut(result)
		return result

	def gradient(self):
		output = self.getPrevOut()
		return np.vectorize(self.dtanh)(output)