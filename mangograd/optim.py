import numpy as np

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)
    
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):

        super().__init__(parameters, lr)
        self.momentum = momentum
        # Keep track of velocities for momentum
        self.velocities = [np.zeros_like(p.grad) for p in parameters]
    
    def step(self):
        for p, v in zip(self.parameters, self.velocities):
            # Update velocity
            v[:] = self.momentum * v - self.lr * p.grad
            # Update weights
            p.data += v
        