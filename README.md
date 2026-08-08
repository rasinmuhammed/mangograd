# 🥭 Mangograd

<div align="center">
  <img src="assets/cybermango.png" alt="Cybermango Hacker" width="400"/>
</div>

**The Origin Story:** It's late. I'm watching Andrej Karpathy's legendary [Micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0). I'm eating an absolutely massive, perfectly ripe mango. The juices are flowing. The gradients are flowing. Suddenly, a thought hits me: *What if I built my own version of this? What if I added my own variations? And what if I named it after this exact fruit?*

Welcome to **Mangograd** as I build it. 

A transparent, pure Python autograd engine and neural network library. Born out of a mango-fueled coding session, with ambitions to bridge the gap between simple educational autograd engines and industrial-grade frameworks like PyTorch.

## Why Mangograd?

In a world where PyTorch exists (but is a massive, complex C++ behemoth) and Micrograd exists (but is a purely educational scalar toy), Mangograd aims to sit in the sweet, juicy middle. 

Trying to building a framework that is:
1. **Readable:** 100% pure Python. If you want to know how the backward pass of a neural network layer works, you can read it directly without getting lost in hardware abstraction layers.
2. **Transparent:** No magic. Just math.
3. **Familiar:** It looks and feels exactly like the PyTorch API you already know.

## Installation

```bash
# Clone the repository (you might want to rename your outer folder to mangograd first!)
git clone https://github.com/yourusername/mangograd.git
cd mangograd

# Install in editable mode
pip install -e .
```

## Quick Start (It's exactly like PyTorch!)

```python
from mangograd.engine import Value
from mangograd.nn import MLP

# Create a multi-layer perceptron (3 inputs, two hidden layers of 4, 1 output)
model = MLP(3, [4, 4, 1])

# Some inputs
x = [Value(2.0), Value(3.0), Value(-1.0)]

# Forward pass
out = model(x)

# Backward pass
out.backward()

# Look at those juicy gradients
for p in model.parameters():
    print(p.grad)
```

## Contributing
Eat a mango. Write some code. Submit a PR. 🥭
