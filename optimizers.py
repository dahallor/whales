import numpy as np

class Adam():
	def __init__(self, learning_rate, p1=0.9, p2=0.999):
		self.p1 = p1
		self.p2 = p2
		self.learning_rate = learning_rate

		self.s = 0
		self.r = 0
		self.d = 10**-8

	def calc(self, gradient, epoch):
		self.s = self.p1*self.s + ((1-self.p1) *gradient)
		self.r = self.p2*self.r + (1-self.p2)*(gradient*gradient)

		return self.learning_rate * ((self.s/(1-self.p1**epoch))/(np.sqrt(self.r/(1-self.p2**epoch)) + self.d))