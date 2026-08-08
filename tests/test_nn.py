import torch
from mangograd.engine import Value
from mangograd.nn import Neuron, Layer, MLP

def test_neuron():
    n = Neuron(3)
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    out = n(x)
    assert isinstance(out, Value)
    
def test_mlp():
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    out = n(x)
    assert isinstance(out, Value)
    assert len(n.parameters()) == 41 # 3*4+4 (16) + 4*4+4 (20) + 4*1+1 (5) = 41
