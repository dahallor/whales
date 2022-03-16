import numpy as np
import pdb

#TODO: be sure to import whichever files this will draw from

'''
Takes a 2D numpy array, and a type of pool and returns either a max pool or an avg pool depending on what's called

Call either the max_pool or avg_pool functions, don't touch the regular pooling function

I made the pool size 2x2 to avoid index out of bounds issues so just be sure when image resizing data 
the new images pixel sizes are an even number
'''

class PoolingLayer():
    def __init__(self):
        self.coordinates = []
        self.pool_type = ""
        self.size_pooled = []
        self.size_original = []
        self.pool = np.array([])
        self.pool_size = 2
        self.stride = 2

    def pooling(self, image, pool_type):
        self.size_original.append(image.shape[0])
        self.size_original.append(image.shape[1])
        for i in np.arange(image.shape[0], step = self.stride):
            for j in np.arange(image.shape[0], step = self.stride):
                temp = image[i:i+self.pool_size, j:j+self.pool_size]
                if pool_type == "max":
                    line1 = max(temp[0][0], temp[0][1])
                    line2 = max(temp[1][0], temp[1][1])
                    max_val = max(line1, line2)
                    if max_val == temp[0][0]:
                        coords = [i, j]
                        self.coordinates.append(coords)
                    elif max_val == temp[0][1]:
                        coords = [i, j+1]
                        self.coordinates.append(coords)
                    elif max_val == temp[1][0]:
                        coords = [i+1, j]
                        self.coordinates.append(coords)
                    else:
                        coords = [i+1, j+1]
                        self.coordinates.append(coords)
                    self.pool = np.append(self.pool, max_val)
                if pool_type == "avg":
                    line1 = np.sum(temp[0], dtype = 'float')
                    line2 = np.sum(temp[1], dtype = 'float')
                    total = line1 + line2
                    avg_val = total / (self.pool_size * self.pool_size)
                    self.pool = np.append(self.pool, avg_val)
        
        self.pool = self.pool.reshape(image.shape[0]//2, image.shape[1]//2)
        self.size_pooled.append(image.shape[0])
        self.size_pooled.append(image.shape[1])
        return self.pool



    def max_pool(self, image):
        pool_type = "max"
        self.pool_type = pool_type
        return self.pooling(image, pool_type)


    def avg_pool(self, image):
        pool_type = "avg"
        self.pool_type = pool_type
        return self.pooling(image, pool_type)

    def backprop_pooling(self, gradientIn):
        k = 0
        gradientOut = np.zeros((self.size_original[0], self.size_original[1]))
        if self.pool_type == "max":
            for i in range(gradientIn.shape[0]):
                for j in range(gradientIn.shape[1]):
                    value = gradientIn[i][j]
                    position1 = self.coordinates[k][0]
                    position2 = self.coordinates[k][1]
                    gradientOut[position1][position2] = value
                    k += 1
        if self.pool_type == "avg":
            line = gradientIn.shape[0] * gradientIn.shape[1]
            gradientIn = gradientIn.reshape(line)
            for i in np.arange(gradientOut.shape[0], step = self.stride):
                for j in np.arange(gradientOut.shape[0], step = self.stride):
                    value = gradientIn[k]/(self.pool_size * self.pool_size)
                    gradientOut[i:i+self.pool_size, j:j+self.pool_size] = value
                    k += 1
        return gradientOut

