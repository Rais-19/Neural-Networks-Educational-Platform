import numpy as np
from backend.utils.loss import get_loss, get_loss_gradient

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, model, X, y):
        """Full training step: forward + backward + update"""
        # Forward pass
        y_pred = model.forward(X)
        
        # Compute loss
        loss_fn = get_loss('binary_crossentropy')
        loss = loss_fn(y, y_pred)
        
        # Get loss gradient w.r.t output
        loss_grad = get_loss_gradient('binary_crossentropy')
        dA = loss_grad(y, y_pred)
        
        # Backpropagation
        dA_prev = dA
        for layer in reversed(model.layers):
            # We need to modify backward to work with current structure
            pass
        
        return loss, y_pred