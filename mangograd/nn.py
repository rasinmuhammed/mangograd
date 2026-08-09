import numpy as np
from mangograd.tensor import Tensor

class Module:
    training = True

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def _children(self):
        """Sub-modules held as attributes, including inside lists and tuples."""
        for value in vars(self).values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield item

    def parameters(self):
        """Walk the module tree.

        Subclasses that only override this return their own tensors and lose
        anything held by a child, so a BatchNorm added to a subclass would
        never be trained. Collecting from children by default makes that
        impossible to get wrong.
        """
        found = []
        for child in self._children():
            found.extend(child.parameters())
        return found

    def train(self, mode=True):
        """Set training mode on this module and every child.

        Dropout and BatchNorm behave differently at inference, and without
        this there is no way to switch a whole model at once.
        """
        self.training = mode
        for child in self._children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def save_state_dict(self, path):
        # np.savez appends .npz unless the path already has it. Normalising
        # here keeps save and load symmetric; without it the round trip
        # raises FileNotFoundError.
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        arrays = {f'p_{i}': p.data for i, p in enumerate(self.parameters())}
        np.savez(path, **arrays)
        return path

    def load_state_dict(self, path):
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        arrays = np.load(path)
        params = self.parameters()
        if len(arrays.files) != len(params):
            raise ValueError(
                f"checkpoint has {len(arrays.files)} tensors, "
                f"model has {len(params)}"
            )
        for i, p in enumerate(params):
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

class Dropout(Module):
    def __init__(self, p=0.5):
        # p = probability of dropping a neuron
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout p must be in [0, 1), got {p}")
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
        self.eps = 1e-5
        self.training = True

    def __call__(self, x):
        if self.training:
            # Statistics must stay inside the graph. Computing them on x.data
            # detaches them, so gradients flow around the normalisation
            # instead of through it, which is the entire subtlety of the
            # BatchNorm derivation and produces visibly wrong gradients.
            mean = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)

            # Running estimates are inference-time buffers, not parameters,
            # so they are plain arrays updated outside the graph.
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var.data
            )
        else:
            mean = Tensor(self.running_mean)
            var = Tensor(self.running_var)

        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_norm + self.beta

    def parameters(self):
        return [self.gamma, self.beta]