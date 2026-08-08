import numpy as np
from mangograd.tensor import Tensor

class MSELoss:
    def __call__(self, predictions, targets):
        # (predictions - targets)^2.mean()
        diff = (predictions - targets)
        return (diff * diff).mean()

class CrossEntropyLoss:
    def __call__(self, logits, targets):
        """
        logits: Raw predictions from the network (Tensor of shape [batch_size, num_classes])
        targets: True labels (NumPy array of shape [batch_size]) containing class indices.
        """
        batch_size = logits.data.shape[0]
        # Subtract max for numerical stability (prevents e^1000 from overflowing)
        max_logits = logits.max(axis=1, keepdims=True)
        # Exponentiate
        exp_logits = (logits - max_logits).exp()

        # Normalize (divide by the sum to get probabilities adding to 1.0)
        sum_exp = Tensor(np.sum(exp_logits.data, axis=1, keepdims=True))
        probs = exp_logits / sum_exp

        # Grab the correct probability using the target indices
        correct_probs = probs.data[np.arange(batch_size), targets]

        # Negative Log Likelihood
        loss = Tensor(-np.log(correct_probs + 1e-8)).mean()

        def _backward():
            # The calculus derivative of Softmax + CrossEntropy is: (Predicted Probability - 1) for the correct class.
            dlogits = probs.data.copy()
            dlogits[np.arange(batch_size), targets] -= 1.0
            dlogits /= batch_size # because we took .mean() at the end
            
            logits.grad += dlogits

        loss._backward = _backward
        loss._prev = {logits}

        return loss

    
        
            
            

        
        