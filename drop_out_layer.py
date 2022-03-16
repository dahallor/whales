import numpy as np

class DropOutLayer:
    def __init__(self, p, input_size):
        u1 = np.random.binomial(1,p, size =input_size)/p
        
        
    def forward(x):
        return x*u1
        
    def backforward(grad):
        
        return grad*u1
    