import numpy as np

class DropOutLayer:
    def __init__(self, p, input_size):
        self.u1 = np.random.binomial(1,p, size =input_size)/p
        
    def forward(self, x):
        return x*self.u1
        
    def backforward(self, grad):
        
        return grad*self.u1
    