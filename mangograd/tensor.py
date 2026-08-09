import numpy as np

def unbroadcast(grad, target_shape):
    """
    If 'grad' has a larger shape than 'target_shape' due to broadcasting,
    sum over the extra/expanded dimensions so the shapes match.
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
        self.data = np.array(data)
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
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)
       
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), 'matmul')

        def _backward():
            self.grad += out.grad @ other.data.T 
            other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out

    def __truediv__(self, other):
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
        # sums the entire tensor into a single scalar value
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            # The gradient routes 1.0 to every element in the original tensor
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        # Adding a tiny epsilon to prevent taking the log of absolute zero
        out = Tensor(np.log(self.data + 1e-8), (self,), 'log')

        def _backward():
            self.grad += (1.0 / (self.data + 1e-8)) * out.grad

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,), 'max')
        def _backward():
            # Gradient routes out to the maximum elements
            # We create a boolean mask of where the max elements are
            mask = (self.data == out.data)
            self.grad += unbroadcast( mask * out.grad, self.data.shape)
        
        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(np.mean(self.data), (self,), 'mean')
        def _backward():
            # Gradient distributed equally, scaled down by number of elements
            self.grad += np.ones_like(self.data) * (out.grad / self.data.size)
        
        out._backward = _backward
        return out

    def relu(self):
        # Forward pass: max(0, x)
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            # Backward pass: Gradient flows only if the data was > 0
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out

    def backward(self, explain=False):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # The initial gradient of the loss is a matrix of 1.0s
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

