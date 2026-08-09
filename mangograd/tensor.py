"""The vectorised engine.

Same idea as engine.py, one array at a time instead of one number at a time.
Each operation computes its result and closes over a `_backward` that knows
its own local derivative. `backward()` walks the graph in reverse and calls
them in order, so the chain rule falls out of the ordering.

Read `engine.py` first if you have not. The only genuinely new concept here
is `unbroadcast`, immediately below.
"""

import numpy as np


def unbroadcast(grad, target_shape):
    """Fold a broadcast gradient back down to the shape it came from.

    Broadcasting in the forward pass is a fan-out: one bias row of shape
    (1, out) is reused by every row in a (B, out) batch. The reverse of a
    fan-out is a sum, so the gradient has to be summed back along exactly
    the axes that were expanded on the way forward.

    Without this, a bias of shape (1, 4) would receive a gradient of shape
    (32, 4) and `self.grad += ...` would fail, or worse, broadcast again.

    >>> unbroadcast(np.ones((4, 3)), (3,)).shape
    (3,)
    >>> unbroadcast(np.ones((4, 3)), (1, 3))
    array([[4., 4., 4.]])
    """
    ndims_added = grad.ndim - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        # Always float. An integer array silently breaks the optimizer, because
        # p.data += v cannot cast a float update back into an int buffer.
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, grad_shape={self.grad.shape})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        # c = a * b   ->   dL/da = b * dL/dc,   dL/db = a * dL/dc
        # The product rule, which is why each side picks up the other's data.
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        # C = A @ B   ->   dL/dA = dL/dC @ B.T,   dL/dB = A.T @ dL/dC
        #
        # The transposes are not a stylistic choice, they are forced by shape.
        # With A (n, k) and B (k, m), dL/dC is (n, m). dL/dA must come out
        # (n, k), and (n, m) @ (m, k) is the only way to get there. Same
        # argument fixes the other side. If you ever forget the formula, you
        # can rederive it just by making the dimensions line up.
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), 'matmul')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        # c = a / b   ->   dL/da = dL/dc / b,   dL/db = -a / b**2 * dL/dc
        # The second one is the quotient rule, and the minus sign is why
        # dividing by a learned quantity pushes it the opposite way.
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def sum(self):
        """Reduce the whole tensor to one number.

        Usually the last op in a graph, because backward() needs a scalar to
        start from: "the derivative of what, with respect to everything?"

        >>> t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> t.sum().backward()
        >>> t.grad
        array([[1., 1.],
               [1., 1.]])
        """
        # d(sum)/dx = 1 for every element, so the incoming gradient is copied
        # unchanged to all of them.
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        # d(e^x)/dx = e^x, which is already sitting in out.data. The one
        # operation whose derivative is its own output.
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        # d(ln x)/dx = 1/x. The epsilon keeps log(0) from becoming -inf and
        # 1/0 from becoming inf, which happens for real when a softmax
        # saturates and a probability underflows to exactly zero.
        out = Tensor(np.log(self.data + 1e-8), (self,), 'log')

        def _backward():
            self.grad += (1.0 / (self.data + 1e-8)) * out.grad

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,), 'max')

        def _backward():
            # Ties need care, and PyTorch is not internally consistent about
            # them: a global max() splits the gradient evenly across tied
            # maxima, while max(dim=) routes all of it to the first. Both are
            # valid subgradients. We follow PyTorch in each case so gradients
            # are comparable when checking against it.
            if axis is None:
                mask = (self.data == out.data).astype(self.data.dtype)
                mask /= mask.sum()
                grad = out.grad
            else:
                idx = np.argmax(self.data, axis=axis)
                mask = np.zeros_like(self.data)
                np.put_along_axis(
                    mask, np.expand_dims(idx, axis=axis), 1.0, axis=axis
                )
                grad = out.grad if keepdims else np.expand_dims(out.grad, axis=axis)
            self.grad += mask * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            count = self.data.size if axis is None else self.data.shape[axis]
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            # Gradient spreads equally over everything that was averaged.
            self.grad += np.ones_like(self.data) * (grad / count)

        out._backward = _backward
        return out

    def var(self, axis=None, keepdims=False):
        """Biased (population) variance, which is what BatchNorm normalises by.

        Note there is no hand-written `_backward` here. It is built out of
        `mean`, `-` and `*`, so the chain rule already knows how to get
        through it. That is the payoff of a real autograd engine: you write
        the forward maths once and differentiation comes for free.

        >>> Tensor([[1.0, 2.0], [3.0, 4.0]]).var(axis=0).data
        array([1., 1.])
        """
        mu = self.mean(axis=axis, keepdims=True)
        diff = self - mu
        return (diff * diff).mean(axis=axis, keepdims=keepdims)

    def sqrt(self):
        # d(sqrt(x))/dx = 1 / (2 * sqrt(x))
        out = Tensor(np.sqrt(self.data), (self,), 'sqrt')

        def _backward():
            self.grad += (0.5 / np.sqrt(self.data)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        # max(0, x). The derivative is 1 where the input was positive and 0
        # elsewhere, so ReLU acts as a gate: gradient passes through the
        # neurons that fired and stops dead at the ones that did not. That is
        # also how a neuron dies, it stops firing and never gets a gradient
        # to recover with.
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self, explain=False):
        """Run the chain rule backwards through the whole graph.

        Two steps. First a topological sort, so every node is visited only
        after everything that depends on it, otherwise a node would fire
        before its gradient had finished accumulating. Then walk that order
        in reverse, calling each node's local `_backward`.

        Pass `explain=True` to print each node's gradient as it flows, with
        exploding, vanishing and NaN flagged. Useful when a network refuses
        to train and you need to see where the signal dies.

        >>> a = Tensor([2.0]); b = Tensor([3.0])
        >>> (a * b).backward()
        >>> a.grad, b.grad
        (array([3.]), array([2.]))
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed: dL/dL = 1. Everything downstream is this multiplied by local
        # derivatives, which is the chain rule doing its job.
        self.grad = np.ones_like(self.data)


        for i, node in enumerate(reversed(topo)):
            node._backward()

            if explain:
                name = node.label if node.label else f"node_{i}"
                op = node._op if node._op else "leaf"
                grad_abs = np.abs(node.grad)
                
                status = "✓"
                if np.any(np.isnan(node.grad)):
                    status = "💀 NaN DETECTED"
                elif grad_abs.max() > 1000:
                    status = "🔥 EXPLODING"
                elif grad_abs.max() < 1e-7 and node._op:
                    status = "🧊 VANISHING"
                
                print(f"  [{op:>8}] {name:<20} | shape: {str(node.data.shape):<12} | "
                      f"grad min: {node.grad.min():>10.6f} | "
                      f"grad max: {node.grad.max():>10.6f} | "
                      f"grad mean: {node.grad.mean():>10.6f} | {status}")

