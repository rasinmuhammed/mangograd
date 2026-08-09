"""Layers, optimizers, losses, and the module tree."""

import os
import tempfile

import numpy as np
import pytest
import torch

from mangograd.loss import CrossEntropyLoss, MSELoss
from mangograd.nn import BatchNorm, Dropout, Linear, MLP, Module
from mangograd.optim import SGD, Adam
from mangograd.tensor import Tensor


def test_linear_shapes():
    layer = Linear(3, 2)
    out = layer(Tensor(np.random.randn(5, 3)))
    assert out.data.shape == (5, 2)
    assert len(layer.parameters()) == 2


def test_mlp_parameters_found_through_the_tree():
    assert len(MLP(3, 4, 2).parameters()) == 4


def test_subclass_child_modules_are_discovered():
    """A layer held by a subclass must still be trained. Before parameters()
    walked the tree, a BatchNorm added this way was silently frozen."""

    class Net(MLP):
        def __init__(self):
            super().__init__(3, 4, 2)
            self.bn = BatchNorm(4)

    assert len(Net().parameters()) == 6


def test_train_and_eval_propagate_to_children():
    class Net(Module):
        def __init__(self):
            self.drop = Dropout(0.5)
            self.bn = BatchNorm(4)

    net = Net()
    net.eval()
    assert net.training is False and net.drop.training is False and net.bn.training is False
    net.train()
    assert net.training is True and net.drop.training is True


def test_dropout_is_identity_in_eval():
    x = Tensor(np.ones((4, 4)))
    drop = Dropout(0.5)
    drop.eval()
    assert np.allclose(drop(x).data, x.data)


def test_dropout_rejects_impossible_probability():
    with pytest.raises(ValueError):
        Dropout(1.0)


def test_batchnorm_input_gradient_matches_torch():
    """The batch statistics must stay inside the graph. Computing them on
    x.data detaches them, so gradients flow around the normalisation rather
    than through it, which is the whole subtlety of the BatchNorm derivation.
    """
    x = np.random.randn(8, 4)
    bn = BatchNorm(4)
    tx = Tensor(x)
    bn(tx).sum().backward()

    px = torch.tensor(x, requires_grad=True)
    ref = torch.nn.BatchNorm1d(4, eps=1e-5).double()
    ref.train()
    ref(px).sum().backward()

    assert np.allclose(tx.grad, px.grad.numpy(), atol=1e-6)


def test_batchnorm_learns_gamma_and_beta():
    bn = BatchNorm(4)
    x = Tensor(np.random.randn(8, 4))
    (bn(x) * Tensor(np.random.randn(8, 4))).sum().backward()
    assert np.abs(bn.gamma.grad).sum() > 0
    assert np.abs(bn.beta.grad).sum() > 0


def test_mse_loss_matches_torch():
    p_, t_ = np.random.randn(5, 3), np.random.randn(5, 3)
    tp = Tensor(p_)
    MSELoss()(tp, Tensor(t_)).backward()

    pp = torch.tensor(p_, requires_grad=True)
    torch.nn.functional.mse_loss(pp, torch.tensor(t_)).backward()
    assert np.allclose(tp.grad, pp.grad.numpy())


def test_cross_entropy_matches_torch():
    logits = np.random.randn(6, 4)
    targets = np.array([0, 1, 2, 3, 1, 0])

    tl = Tensor(logits)
    loss = CrossEntropyLoss()(tl, targets)
    loss.backward()

    pl = torch.tensor(logits, requires_grad=True)
    ref = torch.nn.functional.cross_entropy(pl, torch.tensor(targets))
    ref.backward()

    assert np.allclose(loss.data, ref.item())
    assert np.allclose(tl.grad, pl.grad.numpy())


def test_integer_input_survives_an_optimizer_step():
    layer = Linear(3, 2)
    layer(Tensor([[1, 2, 3]])).sum().backward()
    SGD(layer.parameters(), lr=0.1).step()


@pytest.mark.parametrize("make_opt", [
    lambda p: SGD(p, lr=0.1),
    lambda p: SGD(p, lr=0.1, momentum=0.9),
    lambda p: Adam(p, lr=0.05),
])
def test_optimizers_reduce_loss(make_opt):
    np.random.seed(0)
    x = Tensor(np.random.randn(20, 3))
    y = Tensor(np.random.randn(20, 1))
    model = MLP(3, 8, 1)
    opt = make_opt(model.parameters())
    lossf = MSELoss()

    first = lossf(model(x), y).data
    for _ in range(50):
        loss = lossf(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.data < first


def test_state_dict_round_trip():
    """np.savez appends .npz, so save and load have to agree on the suffix
    or the round trip raises FileNotFoundError."""
    model = MLP(3, 4, 2)
    before = model.parameters()[0].data.copy()
    path = os.path.join(tempfile.mkdtemp(), "weights")

    model.save_state_dict(path)
    model.parameters()[0].data[:] = 0
    model.load_state_dict(path)

    assert np.allclose(model.parameters()[0].data, before)


def test_load_state_dict_rejects_mismatched_checkpoint():
    """A 4-tensor MLP checkpoint loaded into a 2-tensor Linear should fail
    loudly rather than half-restore the weights."""
    path = os.path.join(tempfile.mkdtemp(), "weights")
    MLP(3, 4, 2).save_state_dict(path)

    with pytest.raises(ValueError, match="tensors"):
        Linear(3, 2).load_state_dict(path)


def test_network_learns_a_non_linear_problem():
    """Correct gradients do not guarantee a network trains. This is the
    end-to-end check, with BatchNorm and Dropout in the stack."""
    np.random.seed(0)
    n = 300
    t = np.random.rand(n) * np.pi
    X = np.vstack([np.c_[np.cos(t), np.sin(t)],
                   np.c_[1 - np.cos(t), 1 - np.sin(t) - 0.5]])
    y = np.array([0] * n + [1] * n)
    X += np.random.randn(*X.shape) * 0.1

    class Net(Module):
        def __init__(self):
            self.l1 = Linear(2, 32)
            self.bn = BatchNorm(32)
            self.drop = Dropout(0.1)
            self.l2 = Linear(32, 2)

        def __call__(self, x):
            return self.l2(self.drop(self.bn(self.l1(x).relu())))

    net = Net()
    opt = Adam(net.parameters(), lr=0.02)
    lossf = CrossEntropyLoss()

    net.train()
    for _ in range(150):
        loss = lossf(net(Tensor(X)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    net.eval()
    accuracy = (np.argmax(net(Tensor(X)).data, axis=1) == y).mean()
    assert accuracy > 0.95
