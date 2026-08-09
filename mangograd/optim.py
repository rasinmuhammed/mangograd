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

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0 # timestep counter

        # First momentum ( mean of gradients ); initialized to zero
        self.m = [np.zeros_like(p.data) for p in parameters]

        # Second momentum ( square of gradients ); initialized to zero
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            # Update first moment estimate ( momentum direction )
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad

            # Update second moment estimate ( rmsprop like )
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2

            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)