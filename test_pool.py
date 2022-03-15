import unittest
import numpy as np
from pool import *

class TestPoolingLayer(unittest.TestCase):

    def __init__(self):
        self.test_array = np.array([
        [1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1],
        [1, 3, 5, 7, 9, 11],
        [2, 4, 6, 8, 10, 12],
        [1, 1, 1, 1, 1, 1],
        [9, 9, 9, 9, 9, 9]
    ])

    PL = PoolingLayer()



    def test_MaxPool(self):
        expected = [[6.0, 4.0, 6.0],
        [4.0, 8.0, 12.0],
        [9.0, 9.0, 9.0]]
        self.assertEqual(PL.max_pool(self.test_array), expected)
    '''
    def test_AvgPool(self, test_array):
        expected = [[3.5, 3.5, 3.5],
        [2.5, 6.5, 10.5],
        [5.0, 5.0, 5.0 ]]
        self.assertEqual(avg_pool(test_array), expected)
    '''

