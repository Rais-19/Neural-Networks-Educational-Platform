import numpy as np
from backend.utils.loss import get_loss, get_loss_gradient
from backend.utils.metrics import accuracy

class Trainer:
    def __init__(self, model, learning_rate=0.05, epochs=2000, 
                 l2_lambda=0.0, patience=200, early_stopping=False):
        
        self.model = model
        self.lr = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda          # Regularization strength
        self.patience = patience
        self.early_stopping = early_stopping
        
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.best_loss = float('inf')
        self.best_weights = None
        self.patience_counter = 0

    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        y_train = y_train.reshape(-1, 1)
        if y_val is not None:
            y_val = y_val.reshape(-1, 1)

        for epoch in range(self.epochs):
            # Forward Pass 
            y_pred = self.model.forward(X_train)

            # Compute Loss with L2 Regularization 
            loss_fn = get_loss('binary_crossentropy')
            loss = loss_fn(y_train, y_pred)

            # Add L2 Regularization penalty
            if self.l2_lambda > 0:
                l2_penalty = 0
                for layer in self.model.layers:
                    l2_penalty += np.sum(layer.weights ** 2)
                loss += (self.l2_lambda / (2 * X_train.shape[0])) * l2_penalty

            # === Backpropagation ===
            dA = get_loss_gradient('binary_crossentropy')(y_train, y_pred)

            for i in range(len(self.model.layers) - 1, -1, -1):
                layer = self.model.layers[i]
                dZ = dA * layer.activation_deriv(layer.z)

                m = X_train.shape[0]
                dW = np.dot(dZ.T, layer.a_prev) / m
                db = np.sum(dZ, axis=0, keepdims=True).T / m

                # Apply L2 regularization to gradients
                if self.l2_lambda > 0:
                    dW += (self.l2_lambda / X_train.shape[0]) * layer.weights

                # Update weights and bias
                layer.weights -= self.lr * dW
                layer.bias -= self.lr * db

                if i > 0:
                    dA = np.dot(dZ, layer.weights)

            # Calculate metrics
            train_acc = accuracy(y_train, y_pred)
            self.history['loss'].append(loss)
            self.history['accuracy'].append(train_acc)

            # Validation
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.forward(X_val)
                val_loss = get_loss('binary_crossentropy')(y_val, y_val_pred)
                val_acc = accuracy(y_val, y_val_pred)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

                # Early Stopping
                if self.early_stopping:
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.patience_counter = 0
                        # Save best weights 
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

            if verbose and epoch % 200 == 0:
                msg = f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                print(msg)

        print("Training finished!")
        return self.history