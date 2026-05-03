import numpy as np
from backend.utils.activation import get_activation


class HistoricalPerceptron:
    """
    Rosenblatt's original perceptron (1958).
    Uses step activation and the perceptron update rule — NO gradient descent.
    Converges only for linearly separable data.
    """
    def __init__(self, input_size, learning_rate=0.1):
        self.lr = learning_rate
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.input_size = input_size

    def _step(self, z):
        return (z >= 0).astype(int)

    def _forward(self, X):
        return self._step(X @ self.weights + self.bias)

    def train(self, X, y, epochs=100, verbose=True):
        history = {'loss': [], 'accuracy': []}
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self._step(np.dot(xi, self.weights) + self.bias)
                delta = self.lr * (yi - pred)
                self.weights += delta * xi
                self.bias += delta
                errors += int(delta != 0)
            loss = errors / len(y)
            acc = 1 - loss
            history['loss'].append(loss)
            history['accuracy'].append(acc)
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}: errors={errors}, acc={acc:.4f}")
        return history

    def predict(self, X):
        return self._forward(X)

    def predict_proba(self, X):
        # For decision boundary plotting: use raw z score mapped to [0,1]
        z = X @ self.weights + self.bias
        return 1 / (1 + np.exp(-z))

    def get_params(self):
        return {'weights': self.weights.tolist(), 'bias': float(self.bias)}


class ModernPerceptron:
    """
    Single-layer perceptron with configurable activation + gradient descent.
    Equivalent to logistic regression when activation='sigmoid'.
    """
    def __init__(self, input_size, activation='sigmoid', learning_rate=0.05,
                 l2_lambda=0.0):
        self.lr = learning_rate
        self.l2 = l2_lambda
        self.input_size = input_size
        self.act_fn, self.act_deriv = get_activation(activation)
        self.activation_name = activation

        # Xavier init for better convergence
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size) * scale
        self.bias = 0.0

    def _forward(self, X):
        z = X @ self.weights + self.bias
        return self.act_fn(z), z

    def train(self, X, y, epochs=500, verbose=True):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        n = len(y)

        for epoch in range(epochs):
            # Forward
            a, z = self._forward(X)

            # Loss (BCE for sigmoid)
            a_clip = np.clip(a, 1e-9, 1 - 1e-9)
            loss = -np.mean(y * np.log(a_clip) + (1 - y) * np.log(1 - a_clip))
            loss += self.l2 * np.sum(self.weights ** 2)

            # Backward (dL/dz directly for sigmoid+BCE = a - y)
            if self.activation_name == 'sigmoid':
                dz = a - y
            else:
                dz = (a - y) * self.act_deriv(z)

            dw = (X.T @ dz) / n + 2 * self.l2 * self.weights
            db = np.mean(dz)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            preds = self.predict(X)
            acc = np.mean(preds == y)
            history['loss'].append(loss)
            history['accuracy'].append(acc)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

        return history

    def predict_proba(self, X):
        a, _ = self._forward(X)
        return a

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def get_params(self):
        return {'weights': self.weights.tolist(), 'bias': float(self.bias),
                'activation': self.activation_name, 'lr': self.lr}