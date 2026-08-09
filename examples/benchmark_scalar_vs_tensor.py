"""Scalar autograd versus vectorised autograd, on an identical network.

Both engines live in this repo, so this is a fair comparison rather than a
claim: `Value` is the scalar engine from Karpathy's micrograd, and `Tensor`
is the NumPy-backed one. Same architecture, same batch, same epochs.

    python examples/benchmark_scalar_vs_tensor.py
"""

import random
import time

import numpy as np

from mangograd.engine import Value
from mangograd.loss import MSELoss
from mangograd.nn import MLP
from mangograd.optim import SGD
from mangograd.tensor import Tensor

BATCH, N_IN, N_HIDDEN, EPOCHS = 64, 20, 32, 20


class ScalarNeuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act

    def parameters(self):
        return self.w + [self.b]


class ScalarLayer:
    def __init__(self, n_in, n_out):
        self.neurons = [ScalarNeuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class ScalarMLP:
    def __init__(self, n_in, n_hidden, n_out):
        self.l1 = ScalarLayer(n_in, n_hidden)
        self.l2 = ScalarLayer(n_hidden, n_out)

    def __call__(self, x):
        return self.l2([v.relu() for v in self.l1(x)])

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


def main():
    random.seed(0)
    np.random.seed(0)

    X = np.random.randn(BATCH, N_IN)
    Y = np.random.randn(BATCH, 1)

    # Scalar: one Value object per number, looped in Python.
    scalar_net = ScalarMLP(N_IN, N_HIDDEN, 1)
    scalar_X = [[Value(v) for v in row] for row in X]

    started = time.time()
    for _ in range(EPOCHS):
        squared = []
        for xi, yi in zip(scalar_X, Y):
            diff = scalar_net(xi)[0] - Value(float(yi[0]))
            squared.append(diff * diff)
        loss = squared[0]
        for term in squared[1:]:
            loss = loss + term
        loss = loss * Value(1.0 / BATCH)

        for p in scalar_net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in scalar_net.parameters():
            p.data -= 0.01 * p.grad
    scalar_seconds = time.time() - started

    # Vectorised: the whole batch as one array, NumPy does the loop in C.
    tensor_net = MLP(N_IN, N_HIDDEN, 1)
    optimizer = SGD(tensor_net.parameters(), lr=0.01)
    criterion = MSELoss()
    tx, ty = Tensor(X), Tensor(Y)

    started = time.time()
    for _ in range(EPOCHS):
        loss = criterion(tensor_net(tx), ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tensor_seconds = time.time() - started

    print(f"network {N_IN}-{N_HIDDEN}-1, batch {BATCH}, {EPOCHS} epochs, "
          f"{len(scalar_net.parameters())} parameters")
    print(f"  scalar Value   {scalar_seconds:8.3f}s")
    print(f"  Tensor         {tensor_seconds:8.3f}s")
    print(f"  speedup        {scalar_seconds / tensor_seconds:8.0f}x")


if __name__ == "__main__":
    main()
