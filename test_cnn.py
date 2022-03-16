import unittest
import numpy as np
from cnn import *

class TestCNNLayer(unittest.TestCase):
	def setUp(self):
		self.test_array = np.array([
		[1,2,3,4],
		[2,2,3,2],
		[1,3,3,3],
		[4,4,4,4]
		])

		self.kernel = np.array([
		[1,2,3],
		[2,2,3],
		[1,3,3]
		])

	def test_convolution(self):
		cnn = ConvolutionalLayer(kernel=self.kernel)

		expected = [[50, 57],[60, 63]]
		self.assertEqual(cnn.convolve(self.test_array), expected)

if __name__ == '__main__':
    unittest.main()