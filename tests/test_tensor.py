"""Gradient correctness for the tensor engine, checked against PyTorch.

torch is a test-only dependency. mangograd itself needs nothing but numpy.
"""

import numpy as np
import pytest
import torch

from mangograd.tensor import Tensor, unbroadcast


def test_data_is_always_float():
    """An integer array silently breaks the optimizer later, because
    p.data += update cannot cast a float back into an int buffer."""
    assert Tensor([1, 2, 3]).data.dtype == np.float64


def test_matmul_add_relu_matches_torch():
    a, b, c = np.random.randn(4, 3), np.random.randn(3, 5), np.random.randn(1, 5)
    ta, tb, tc = Tensor(a), Tensor(b), Tensor(c)
    ((ta @ tb + tc).relu()).sum().backward()

    pa = torch.tensor(a, requires_grad=True)
    pb = torch.tensor(b, requires_grad=True)
    pc = torch.tensor(c, requires_grad=True)
    (pa @ pb + pc).relu().sum().backward()

    assert np.allclose(ta.grad, pa.grad.numpy())
    assert np.allclose(tb.grad, pb.grad.numpy())
    assert np.allclose(tc.grad, pc.grad.numpy())   # broadcast bias


def test_div_exp_log_mean_match_torch():
    x = np.abs(np.random.randn(3, 4)) + 0.5
    tx = Tensor(x)
    (tx.log() * tx.exp() / Tensor(2.0)).mean().backward()

    px = torch.tensor(x, requires_grad=True)
    ((px.log() * px.exp()) / 2.0).mean().backward()
    assert np.allclose(tx.grad, px.grad.numpy())


def test_gradient_accumulates_when_tensor_used_twice():
    y = np.random.randn(3, 3)
    ty = Tensor(y)
    (ty * ty).sum().backward()

    py = torch.tensor(y, requires_grad=True)
    (py * py).sum().backward()
    assert np.allclose(ty.grad, py.grad.numpy())


def test_max_over_axis_breaks_ties_like_torch():
    """A plain equality mask routes gradient to every tied maximum, which
    double counts: [1, 5, 5] would get [0, 1, 1] instead of [0, 1, 0]."""
    d = np.array([[1.0, 5.0, 5.0]])
    t = Tensor(d)
    t.max(axis=1, keepdims=True).sum().backward()

    p = torch.tensor(d, requires_grad=True)
    p.max(dim=1, keepdim=True)[0].sum().backward()
    assert np.allclose(t.grad, p.grad.numpy())


def test_global_max_splits_ties_like_torch():
    """PyTorch is not internally consistent about ties: a global max splits
    the gradient evenly, while max(dim=) routes it all to the first."""
    z = np.array([[1.0, 9.0], [3.0, 9.0]])
    t = Tensor(z)
    t.max().backward()

    p = torch.tensor(z, requires_grad=True)
    p.max().backward()
    assert np.allclose(t.grad, p.grad.numpy())


@pytest.mark.parametrize("keepdims", [True, False])
def test_mean_over_axis_matches_torch(keepdims):
    m = np.random.randn(4, 5)
    t = Tensor(m)
    t.mean(axis=0, keepdims=keepdims).sum().backward()

    p = torch.tensor(m, requires_grad=True)
    p.mean(dim=0, keepdim=keepdims).sum().backward()
    assert np.allclose(t.grad, p.grad.numpy())


def test_var_matches_torch_population_variance():
    v = np.random.randn(6, 3)
    t = Tensor(v)
    t.var(axis=0, keepdims=True).sum().backward()

    p = torch.tensor(v, requires_grad=True)
    p.var(dim=0, keepdim=True, unbiased=False).sum().backward()
    assert np.allclose(t.grad, p.grad.numpy(), atol=1e-6)


def test_sqrt_matches_torch():
    x = np.abs(np.random.randn(3, 3)) + 0.1
    t = Tensor(x)
    t.sqrt().sum().backward()

    p = torch.tensor(x, requires_grad=True)
    p.sqrt().sum().backward()
    assert np.allclose(t.grad, p.grad.numpy())


def test_unbroadcast_collapses_added_dimensions():
    grad = np.ones((4, 3))
    assert unbroadcast(grad, (3,)).shape == (3,)
    assert unbroadcast(grad, (1, 3)).shape == (1, 3)
