# 🥭 Mangograd

<div align="center">
  <img src="assets/cybermango.png" alt="Cybermango Hacker" width="400"/>
</div>

**The Origin Story:** It's late. I'm watching Andrej Karpathy's legendary [Micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0). I'm eating a massive, perfectly ripe mango. The juices are flowing. The gradients are flowing. Suddenly, a thought hits me: *What if I built my own version of this, but actually made it fast enough to be usable?*

Welcome to **Mangograd**. A NumPy-backed tensor autograd engine that bridges the gap between educational scalar engines and industrial-grade frameworks.

> Built on top of the ideas in [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).
> Micrograd teaches backpropagation on scalars in about 150 lines you can read in one sitting.
> Mangograd picks up where that leaves off.

## The Gap: Why does this exist?

If you want to understand how neural networks work, you have two extremes:
1. **Micrograd:** Brilliant for learning. But because it operates entirely on scalars (single numbers) inside Python `for` loops, a 2-layer network takes hours to train. It's a toy.
2. **PyTorch / TinyGrad:** The industry standards. They are insanely fast because they compile graphs down to C++, CUDA, or Metal. But if you want to actually read the source code to see how the math of backpropagation works on a `Linear` layer, you end up digging through hardware compilation kernels instead of raw math.

Mangograd sits directly in the middle. It uses multidimensional **Tensors** backed by `numpy`, so you get the massive speedups of C-level vectorization and matrix broadcasting. But the source code is 100% pure, readable Python. No JIT compilers, no hardware abstraction layers. Just math.

## When to use Mangograd

**Use it if:**
* You are studying Deep Learning and want to read exactly how the matrix calculus of backpropagation works under the hood.
* You want to prototype a weird, custom neural network layer mathematically, and PyTorch's backend is too dense to hack on.
* You are deploying to a highly restricted environment (like a cheap edge device) where you can only `pip install numpy` and cannot afford the 2GB PyTorch wheel.

**Do NOT use it if:**
* You are training a large language model.
* You need GPU support. (Go use PyTorch).

## Installation

```bash
git clone https://github.com/rasinmuhammed/mangograd.git
cd mangograd

# Install in editable mode
pip install -e .
```

## Quick Start (It's exactly like PyTorch)

Mangograd implements the PyTorch API you already know.

```python
import numpy as np
from mangograd.tensor import Tensor
from mangograd.nn import MLP
from mangograd.optim import SGD
from mangograd.loss import CrossEntropyLoss

# 1. Create Data
X = Tensor(np.random.randn(32, 10)) # Batch of 32, 10 features
y = np.random.randint(0, 3, size=(32,)) # 3 target classes

# 2. Build Model
model = MLP(in_features=10, hidden_features=16, out_features=3)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = CrossEntropyLoss()

# 3. Train
for epoch in range(100):
    logits = model(X)
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch} | Loss: {loss.data:.4f}")

# 4. Save
model.save_state_dict("model.npz")
```

## Reading the source

The whole library is about 700 lines. If you want to understand how autograd
actually works, read it in this order:

**1. `mangograd/engine.py`** (~100 lines) is micrograd: one number at a time.
Start with `__add__` and `__mul__`. Notice that each operation stores a
`_backward` closure holding its own local derivative, and that `backward()`
just calls them in reverse topological order. That is the entire idea. Every
deep learning framework is this, plus performance engineering.

**2. `mangograd/tensor.py`** is the same thing over NumPy arrays. The
operations are identical in spirit, so the only genuinely new concept is
`unbroadcast`, which is the first function in the file and the one worth
slowing down for. `__matmul__` is the other one to sit with: the transposes
in its backward pass are forced by shape, not chosen.

**3. `mangograd/nn.py`** is Tensors in a trench coat. No layer has a
`backward` method because none of them needs one, the engine already handles
it. `BatchNorm` is the interesting one: its batch statistics are computed
with `Tensor` ops rather than raw NumPy, specifically so gradients flow
*through* the normalisation instead of around it.

**4. `mangograd/optim.py` and `mangograd/loss.py`** last. SGD is four lines.
Adam is worth reading closely if you have only ever used it as a string.

Every backward pass has the derivative written above it in maths notation,
so you can check the code against the calculus without leaving the file.

## Benchmark: scalars versus tensors

Micrograd represents every number as its own `Value` object and loops in
Python, which is what makes it so readable and also what makes it unusable for
real training. Mangograd keeps the same readable backward passes but runs them
over NumPy arrays.

Both engines are in this repo, so you can run the comparison yourself on an
identical network:

```
network 20-32-1, batch 64, 20 epochs, 705 parameters
  scalar Value     19.335s
  Tensor            0.007s
  speedup            2736x
```

```bash
python examples/benchmark_scalar_vs_tensor.py
```

This is not a criticism of micrograd. It is deliberately scalar so the code
stays small enough to teach from. The point is that once you understand it,
you need something vectorised, and that is the gap this fills.

## Correctness

Every gradient is verified against PyTorch in the test suite, not asserted by
eye. The checks cover matmul, broadcasting, division, exp, log, mean and
variance over axes, max (including how ties are handled), ReLU, MSE,
cross entropy, and gradient accumulation when a tensor is used more than once.

BatchNorm is checked against `torch.nn.BatchNorm1d` for the input gradient
specifically, because computing batch statistics outside the graph is an easy
mistake that produces plausible but wrong gradients.

There is also an end-to-end test that trains a network with BatchNorm and
Dropout to over 95% accuracy on a non-linearly separable problem, since
correct gradients alone do not prove a model can learn.

Docstring examples are executed as part of the suite, so documentation that
drifts out of date fails CI rather than quietly misleading whoever reads it.

```bash
pip install -e ".[dev]"
pytest -q
```

## Contributing

Issues and pull requests are welcome. Run `pytest -q` before opening one.

Eat a mango while you work. 🥭

## License

MIT. See [LICENSE](LICENSE).

