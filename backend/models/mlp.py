import numpy as np
from backend.utils.activation import get_activation


class Layer:
    def __init__(self, n_in, n_out, activation='relu'):
        self.activation_name = activation
        self.act_fn, self.act_deriv = get_activation(activation)

        # He init for ReLU, Xavier for others
        if activation == 'relu':
            scale = np.sqrt(2.0 / n_in)
        else:
            scale = np.sqrt(1.0 / n_in)

        self.W = np.random.randn(n_in, n_out) * scale
        self.b = np.zeros((1, n_out))

        # Cache for backprop
        self.Z = None
        self.A = None
        self.A_prev = None

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, A_prev, training=False, dropout_rate=0.0):
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b

        if self.activation_name == 'softmax':
            self.A = self.act_fn(self.Z)
        else:
            self.A = self.act_fn(self.Z)

        # Dropout (only on hidden layers, only during training)
        self.dropout_mask = None
        if training and dropout_rate > 0 and self.activation_name not in ('softmax', 'sigmoid', 'linear'):
            self.dropout_mask = (np.random.rand(*self.A.shape) > dropout_rate) / (1 - dropout_rate)
            self.A = self.A * self.dropout_mask

        return self.A

    def backward(self, dA, l2_lambda=0.0):
        n = self.A_prev.shape[0]

        if self.dropout_mask is not None:
            dA = dA * self.dropout_mask

        if self.activation_name == 'softmax':
            dZ = dA  # gradient already computed at loss layer for softmax+CCE
        else:
            dZ = dA * self.act_deriv(self.Z)

        self.dW = (self.A_prev.T @ dZ) / n + (l2_lambda / n) * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True) / n
        dA_prev = dZ @ self.W.T
        return dA_prev

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class MLP:
    """
    Multi-Layer Perceptron supporting:
    - Binary classification (sigmoid output + BCE loss)
    - Multi-class classification (softmax output + CCE loss)
    - Regression (linear output + MSE loss)
    - L2 regularization
    - Dropout
    - Mini-batch gradient descent
    """
    def __init__(self, input_size, hidden_layers, output_size,
                 hidden_activation='relu', output_activation='sigmoid',
                 problem_type='binary'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.problem_type = problem_type  # 'binary', 'multiclass', 'regression'

        # Build layers
        self.layers = []
        sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(sizes) - 1):
            act = hidden_activation if i < len(sizes) - 2 else output_activation
            self.layers.append(Layer(sizes[i], sizes[i + 1], activation=act))

    # Forward 

    def forward(self, X, training=False, dropout_rate=0.0):
        A = X
        for i, layer in enumerate(self.layers):
            is_hidden = i < len(self.layers) - 1
            dr = dropout_rate if is_hidden else 0.0
            A = layer.forward(A, training=training, dropout_rate=dr)
        return A

    #  Loss

    def _compute_loss(self, y_true, y_pred, l2_lambda=0.0):
        n = y_true.shape[0]

        if self.problem_type == 'binary':
            y_pred_c = np.clip(y_pred, 1e-9, 1 - 1e-9)
            loss = -np.mean(y_true * np.log(y_pred_c) + (1 - y_true) * np.log(1 - y_pred_c))
        elif self.problem_type == 'multiclass':
            y_pred_c = np.clip(y_pred, 1e-9, 1 - 1e-9)
            loss = -np.mean(np.sum(y_true * np.log(y_pred_c), axis=1))
        else:  # regression
            loss = np.mean((y_true - y_pred) ** 2)

        # L2 penalty
        if l2_lambda > 0:
            l2_sum = sum(np.sum(layer.W ** 2) for layer in self.layers)
            loss += (l2_lambda / (2 * n)) * l2_sum

        return loss

    def _loss_gradient(self, y_true, y_pred):
        if self.problem_type == 'binary':
            y_pred_c = np.clip(y_pred, 1e-9, 1 - 1e-9)
            return (y_pred_c - y_true)  # collapsed gradient for sigmoid+BCE
        elif self.problem_type == 'multiclass':
            return (y_pred - y_true)    # collapsed gradient for softmax+CCE
        else:
            return 2 * (y_pred - y_true)

    # Backward 

    def backward(self, y_true, y_pred, l2_lambda=0.0):
        dA = self._loss_gradient(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, l2_lambda=l2_lambda)

    # Train 

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=500, learning_rate=0.05, l2_lambda=0.0,
              dropout_rate=0.0, batch_size=None,
              early_stopping_patience=0, verbose=True):

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        n = X_train.shape[0]
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):

            #  Mini-batch or full batch
            if batch_size is not None and batch_size < n:
                indices = np.random.permutation(n)
                X_shuf, y_shuf = X_train[indices], y_train[indices]
                for start in range(0, n, batch_size):
                    Xb = X_shuf[start:start + batch_size]
                    yb = y_shuf[start:start + batch_size]
                    yb_pred = self.forward(Xb, training=True, dropout_rate=dropout_rate)
                    self.backward(yb, yb_pred, l2_lambda=l2_lambda)
                    for layer in self.layers:
                        layer.update(learning_rate)
            else:
                y_pred = self.forward(X_train, training=True, dropout_rate=dropout_rate)
                self.backward(y_train, y_pred, l2_lambda=l2_lambda)
                for layer in self.layers:
                    layer.update(learning_rate)

            # Record metrics
            y_pred_train = self.forward(X_train, training=False)
            train_loss = self._compute_loss(y_train, y_pred_train, l2_lambda)
            train_acc = self._compute_accuracy(y_train, y_pred_train)
            history['loss'].append(float(train_loss))
            history['accuracy'].append(float(train_acc))

            if X_val is not None:
                y_pred_val = self.forward(X_val, training=False)
                val_loss = self._compute_loss(y_val, y_pred_val, l2_lambda)
                val_acc = self._compute_accuracy(y_val, y_pred_val)
                history['val_loss'].append(float(val_loss))
                history['val_accuracy'].append(float(val_acc))

                # Early stopping 
                if early_stopping_patience > 0:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
                    else:
                        patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        if best_weights:
                            for layer, (W, b) in zip(self.layers, best_weights):
                                layer.W, layer.b = W, b
                        break

            if verbose and epoch % 100 == 0:
                msg = f"Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.4f}"
                if X_val is not None:
                    msg += f" | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                print(msg)

        return history

    # Predict 

    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.problem_type == 'binary':
            return (proba.flatten() >= 0.5).astype(int)
        elif self.problem_type == 'multiclass':
            return np.argmax(proba, axis=1)
        else:
            return proba.flatten()

    def _compute_accuracy(self, y_true, y_pred):
        if self.problem_type == 'regression':
            return float(1 - np.mean(np.abs(y_true.flatten() - y_pred.flatten())) /
                         (np.std(y_true) + 1e-9))
        if self.problem_type == 'binary':
            true_labels = y_true.flatten().astype(int)
            pred_labels = (y_pred.flatten() >= 0.5).astype(int)
        else:
            true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true.astype(int)
            pred_labels = np.argmax(y_pred, axis=1)
        return float(np.mean(true_labels == pred_labels))

    def get_architecture_summary(self):
        sizes = [self.input_size]
        acts = []
        for layer in self.layers:
            sizes.append(layer.W.shape[1])
            acts.append(layer.activation_name)
        return sizes, acts