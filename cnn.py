import numpy as np
import pdb
from layers import Layer

'''
Convolutional layer. Forward and back-propegates image data.

Note that backprop may bit a bit busted RN. It's a work in progress.
'''

class ConvolutionalLayer(Layer):
	def __init__(self, kernel_size=3, kernel=None, learning_rate=0.001):
		if kernel is None:
			kernel = np.random.rand((kernel_size, kernel_size)) * 10**-4
		else:
			self.kernel = kernel
		self.biases = np.array([0,0,0])
		self.learning_rate = learning_rate

		self.prevIn = None

	def forward(self, image, num_channels=1 ,stride=1, zero_pad=True):
		self.prevIn = image
		m = kernel.shape[0]

		if zero_pad:
			zeros_v = np.zeros((image.shape[0] + 2, 1))
			zeros_h = np.zeros((1, image.shape[1]))
			image = np.vstack((zeros_h, image, zeros_h)) # add rows
			image = np.hstack((zeros_v, image, zeros_v)) # add cols

		output_shape = (int((image.shape[0] - m) / stride) + 1, int((image.shape[1] - m) / stride) + 1, num_channels)

		output = np.zeros(output_shape)

		for filter_num in range(num_channels):

			# Check for proper kernel shape
			if self.kernel.shape[0] % 2 == 0 or self.kernel.shape[1] % 2 == 0:
				raise Exception("Kernel is not odd-shaped.")

			feature_map = np.zeros((int((image.shape[0] - m) / stride) + 1, int((image.shape[1] - m) / stride) + 1))

			for i in range(0, image.shape[0] - m+1, stride):
				for j in range(0, image.shape[1]- m+1, stride):
					feature_map[i,j] = (np.multiply(image[i:i+m, j:j+m], kernel)).sum()+self.biases[filter_num]

			output[:, :, filter_num] = feature_map
			
		return output

	def backward(self, grad_in):
		m = kernel.shape[0]

		output = np.zeros((m, m))

		for i in range(0, m):
			for j in range(0, m):
				current_grad = self.prevIn[i:self.prevIn.shape[0] - m+i, j:self.prevIn.shape[1]- m+j]

				# This uses numpy correllation, which is sometimes borked. 
				output[i, j] = np.correlate(np.ndarray.flatten(grad_in), np.ndarray.flatten(current_grad)).sum()

		self.kernel -= output*self.learning_rate
		return output

if __name__ == '__main__':
	test_array = np.array([
		[1,2,3,4],
		[2,2,3,2],
		[1,3,3,3],
		[4,4,4,4]
		])

	kernel = np.array([
		[1,2,3],
		[2,2,3],
		[1,3,3]
		]).astype(float)

	cnn = ConvolutionalLayer(kernel=kernel)

	i = cnn.convolve(test_array, zero_pad=False)

	j = cnn.backprop(test_array)