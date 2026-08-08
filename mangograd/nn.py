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