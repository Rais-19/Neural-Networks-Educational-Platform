import numpy as np


def forward(model, X, training=False, dropout_rate=0.0):
    """
    Run a full forward pass through every layer of the model.

    Parameters
    ----------
    model        : MLP instance — must have a .layers list of Layer objects
    X            : Input data, shape (n_samples, n_features)
    training     : bool — enables dropout when True (disabled at inference)
    dropout_rate : float — fraction of neurons to drop (hidden layers only)

    Returns
    -------
    A : np.ndarray — final output of the network, shape (n_samples, n_outputs)

    How it works
    ------------
    Each layer computes two things:
        Z = A_prev @ W + b          (linear combination)
        A = activation(Z)           (non-linear transformation)

    Z and A_prev are cached inside each Layer object so that
    backward.py can access them during backpropagation.
    """
    A = X
    for i, layer in enumerate(model.layers):
        is_hidden = i < len(model.layers) - 1
        dr = dropout_rate if is_hidden else 0.0
        A  = layer.forward(A, training=training, dropout_rate=dr)
    return A


def forward_layer(layer, A_prev, training=False, dropout_rate=0.0):
    """
    Forward pass through a single Layer.
    Useful for debugging or stepping through the network layer by layer.

    Parameters
    ----------
    layer        : Layer instance
    A_prev       : Input activations from previous layer (or raw X for layer 0)
    training     : bool
    dropout_rate : float

    Returns
    -------
    A : np.ndarray — activations after applying weights + activation function
    """
    return layer.forward(A_prev, training=training, dropout_rate=dropout_rate)


def get_all_activations(model, X):
    """
    Run forward pass and return ALL intermediate activations, not just the output.
    Useful for visualizing what each layer learns.

    Parameters
    ----------
    model : MLP instance
    X     : Input data, shape (n_samples, n_features)

    Returns
    -------
    activations : list of np.ndarray
        activations[0] = X (input)
        activations[1] = output of layer 0
        activations[2] = output of layer 1
        ...
        activations[-1] = final output
    """
    activations = [X]
    A = X
    for layer in model.layers:
        A = layer.forward(A, training=False, dropout_rate=0.0)
        activations.append(A)
    return activations