import numpy as np
from mangograd.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []

    def save_state_dict(self, path):
        arrays = {f'p_{i}': p.data for i, p in enumerate(self.parameters())}
        np.savez(path, **arrays)

    def load_state_dict(self, path):
        arrays = np.load(path)
        for i, p in enumerate(self.parameters()):
            p.data = arrays[f'p_{i}']

class Linear(Module):
    def __init__(self, in_features, out_features):
        # Weight Matrix: Shape (in, out)
        # We scale by sqrt(in_features) for better initialization (Kaiming init)
        self.weight = Tensor(np.random.randn(in_features, out_features) / np.sqrt(in_features))
        # Bias Vector: Shape (1, out)
        self.bias = Tensor(np.zeros((1, out_features)))

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]

    def __repr__(self):
        return f"Linear(in_features={self.weight.data.shape[0]}, out_features={self.weight.data.shape[1]})"

class MLP(Module):
    def __init__(self, in_features, hidden_features, out_features):
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def __call__(self, x):
        x = self.layer1(x).relu()
        return self.layer2(x)

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()

class Dropout(Module):
    def __init__(self, p=0.5):
        # p = probability of dropping a neuron
        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training:
            return x
        # Create a mask of 1s and 0s. 1 = keep, 0 = drop (mask is same shape as x)
        # We keep each neuron with probability (1 - p)
        mask = Tensor((np.random.rand(*x.data.shape) > self.p).astype(np.float64))
        # Scale up survivors so the expected sum stays the same ( inverted dropout )
        return x * mask * (1.0 / (1.0 - self.p))

class BatchNorm(Module):
    def __init__(self, num_features):
        # Learnable parameters
        self.gamma = Tensor(np.ones((1, num_features))) # Scale
        self.beta = Tensor(np.zeros((1, num_features))) # Shift

        # Buffers for running statistics (only updated during inference )
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.momentum = 0.1
        self.training = True

    def __call__(self, x):
        if self.training:
            # Calculate mean and variance of the current batch
            mean = np.mean(x.data, axis=0, keepdims=True)
            var = np.var(x.data, axis=0, keepdims=True)

            # Update running statistics ( Exponentially moving average )
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - Tensor(mean)) * Tensor(1.0 / np.sqrt(var + 1e-5))

        # Rescale with learnable gamma and beta
        return self.gamma * x_norm + self.beta

    def parameters(self):
        return [self.gamma, self.beta]