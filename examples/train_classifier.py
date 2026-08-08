import numpy as np
from mangograd.tensor import Tensor
from mangograd.nn import MLP
from mangograd.optim import SGD
from mangograd.loss import CrossEntropyLoss
import time

# Create a small classification dataset (4 samples, 2 features, 3 possible classes)
np.random.seed(42)
X_data = np.random.randn(4, 2)
y_data = np.array([0, 1, 2, 0]) # The correct class indices

X = Tensor(X_data)
# We don't wrap targets in a Tensor for CrossEntropy, it expects a raw NumPy array of indices
y = y_data

# Initialize our Vectorized MLP (2 inputs -> 16 hidden -> 3 output classes)
model = MLP(2, 16, 3)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = CrossEntropyLoss()

print(f"Model initialized with {len(model.parameters())} parameter tensors.")
print("Starting classification training loop...\n")

start_time = time.time()
for epoch in range(100):
    # Forward pass
    logits = model(X)
    
    # Calculate CrossEntropy Loss
    loss = criterion(logits, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights (SGD with Momentum!)
    optimizer.step()
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.data:.4f}")

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.4f} seconds!")

# Save the model
model.save_state_dict("classifier.npz")
print("Model saved to classifier.npz!")

# Check final predictions
final_logits = model(X).data
predicted_classes = np.argmax(final_logits, axis=1)

print("\nFinal predictions:")
print(f"Predicted Classes: {predicted_classes}")
print(f"True Classes:      {y}")
