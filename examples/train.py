import numpy as np
from mangograd.tensor import Tensor
from mangograd.nn import MLP
import time

# Create a small dataset (The XOR problem)
# X shape: (4, 2)
X = Tensor(np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]))

# y shape: (4, 1)
y = Tensor(np.array([
    [-1.0],
    [ 1.0],
    [ 1.0],
    [-1.0]
]))

# Initialize our new Vectorized MLP (2 inputs -> 16 hidden -> 1 output)
model = MLP(2, 16, 1)
print(f"Model initialized with {len(model.parameters())} parameter tensors.")

epochs = 100
learning_rate = 0.05

print("\nStarting training loop...")
start_time = time.time()

for epoch in range(epochs):
    # Forward pass (computes the entire batch of 4 inputs at once!)
    out = model(X)
    
    # Calculate Mean Squared Error Loss
    diff = out - y
    loss = (diff * diff).sum()
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Update weights (Vectorized SGD)
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.data:.4f}")

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.4f} seconds!")
print("\nFinal predictions (should be close to -1, 1, 1, -1):")
print(model(X).data)
