import numpy as np

def backward(layer, dA, learning_rate):
    """
    Compute gradients for one layer and update weights & bias.
    """
    # Retrieve cached values from forward pass
    Z = layer.z
    A_prev = layer.a_prev if hasattr(layer, 'a_prev') else None  
    
    # Get activation derivative
    activation_deriv = layer.activation_deriv if hasattr(layer, 'activation_deriv') else None
    
    if activation_deriv is None:
        # For output layer or linear activation
        dZ = dA
    else:
        # Chain rule: dZ = dA * g'(Z)
        dZ = dA * activation_deriv(Z)
    
    # Compute gradients
    m = dZ.shape[0]  # number of samples
    dW = np.dot(dZ.T, A_prev) / m
    db = np.sum(dZ, axis=0, keepdims=True).T / m
    
    # Update parameters
    layer.weights -= learning_rate * dW
    layer.bias -= learning_rate * db
    
    # Compute gradient for previous layer (for chaining)
    dA_prev = np.dot(dZ, layer.weights)
    
    return dA_prev