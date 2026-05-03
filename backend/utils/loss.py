import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    n = y_true.shape[0]
    return (2 / n) * (y_pred - y_true)

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_gradient(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    n = y_true.shape[0]
    return (y_pred - y_true) / (n * y_pred * (1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_cross_entropy_gradient(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    n = y_true.shape[0]
    return (y_pred - y_true) / n

LOSSES = {
    'mse': (mse, mse_gradient),
    'binary_cross_entropy': (binary_cross_entropy, binary_cross_entropy_gradient),
    'categorical_cross_entropy': (categorical_cross_entropy, categorical_cross_entropy_gradient),
}

def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(LOSSES.keys())}")
    return LOSSES[name]