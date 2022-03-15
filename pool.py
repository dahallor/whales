import numpy as np

#TODO: be sure to import whichever files this will draw from

'''
Takes a 2D numpy array, and a type of pool and returns either a max pool or an avg pool depending on what's called

Call either the max_pool or avg_pool functions, don't touch the regular pooling function

I made the pool size 2x2 to avoid index out of bounds issues so just be sure when image resizing data 
the new images pixel sizes are an even number
'''

def pooling(image, pool_type):
    pool = np.array([])
    pool_size = 2
    stride = 2
    for i in np.arange(image.shape[0], step = stride):
        for j in np.arange(image.shape[0], step = stride):
            temp = image[i:i+pool_size, j:j+pool_size]
            if pool_type == "max":
                line1 = max(temp[0][0], temp[0][1])
                line2 = max(temp[1][0], temp[1][1])
                max_val = max(line1, line2)
                pool = np.append(pool, max_val)
            if pool_type == "avg":
                line1 = np.mean(temp[0][0], temp[0][1])
                line2 = np.mean(temp[1][0], temp[1][1])
                avg_val = np.mean(line1, line2)
                pool = np.append(pool, avg_val)
    
    pool = pool.reshape(image.shape[0]//2, image.shape[1]//2)
    return pool

test_array = np.array([
    [1, 2, 3, 4, 5, 6],
    [6, 5, 4, 3, 2, 1],
    [1, 3, 5, 7, 9, 11],
    [2, 4, 6, 8, 10, 12],
    [1, 1, 1, 1, 1, 1],
    [9, 9, 9, 9, 9, 9]
])

def max_pool(image):
    pool_type = "max"
    return pooling(image, pool_type)


def avg_pool(image):
    pool_type = "avg"
    return pooling(image, pool_type)


