# 🥭 Mangograd

<div align="center">
  <img src="assets/cybermango.png" alt="Cybermango Hacker" width="400"/>
</div>

**The Origin Story:** It's late. I'm watching Andrej Karpathy's legendary [Micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0). I'm eating a massive, perfectly ripe mango. The juices are flowing. The gradients are flowing. Suddenly, a thought hits me: *What if I built my own version of this, but actually made it fast enough to be usable?*

Welcome to **Mangograd**. A NumPy-backed tensor autograd engine that bridges the gap between educational scalar engines and industrial-grade frameworks.

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

## Contributing
Eat a mango. Write some code. Submit a PR. 🥭
