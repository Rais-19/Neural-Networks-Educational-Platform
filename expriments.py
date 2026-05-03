import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from backend.models.perceptron import HistoricalPerceptron, ModernPerceptron
from backend.models.mlp import MLP
from backend.training.trainer import Trainer
from backend.utils.visualization import Visualizer

viz = Visualizer()

print(" Starting Mandatory Experiments...\n")

# Generate dataset
X, y = make_circles(n_samples=400, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#  EXPERIMENT 1: Perceptron vs MLP 
print("="*70)
print("EXPERIMENT 1: Perceptron vs MLP")
print("="*70)

# Modern Perceptron
print("Training Modern Perceptron...")
perc_model = ModernPerceptron(input_size=2, activation='sigmoid', learning_rate=0.1)
perc_history = perc_model.train(X_train, y_train, epochs=500, verbose=False)

print(f"Modern Perceptron Test Accuracy: {np.mean(perc_model.predict(X_test) == y_test):.4f}")

# MLP
print("Training MLP...")
mlp_model = MLP(input_size=2, hidden_layers=[8, 6], output_size=1, 
                hidden_activation='relu', output_activation='sigmoid')
trainer = Trainer(mlp_model, learning_rate=0.08, epochs=800, l2_lambda=0.001)
mlp_history = trainer.train(X_train, y_train, X_test, y_test, verbose=False)

print(f"MLP Test Accuracy: {np.mean(mlp_model.predict(X_test) == y_test):.4f}")

viz.plot_decision_boundary(perc_model, X, y, title="Decision Boundary - Modern Perceptron")
viz.plot_decision_boundary(mlp_model, X, y, title="Decision Boundary - MLP")

#  EXPERIMENT 2: Effect of Depth 
print("\n" + "="*70)
print("EXPERIMENT 2: Effect of Number of Hidden Layers (Depth)")
print("="*70)

depths = [0, 1, 2, 3]  # 0 = no hidden layer (like perceptron)
results = []

for depth in depths:
    if depth == 0:
        model = ModernPerceptron(input_size=2, activation='sigmoid', learning_rate=0.1)
        model.train(X_train, y_train, epochs=600, verbose=False)
        acc = np.mean(model.predict(X_test) == y_test)
        name = "Single Layer (Perceptron-like)"
    else:
        hidden = [10] * depth
        model = MLP(input_size=2, hidden_layers=hidden, output_size=1,
                    hidden_activation='relu', output_activation='sigmoid')
        trainer = Trainer(model, learning_rate=0.08, epochs=800, l2_lambda=0.001)
        trainer.train(X_train, y_train, verbose=False)
        acc = np.mean(model.predict(X_test) == y_test)
        name = f"{depth} Hidden Layer{'s' if depth > 1 else ''}"
    
    results.append((name, acc))
    print(f"{name:25s} → Test Accuracy: {acc:.4f}")

#EXPERIMENT 3: Overfitting vs Regularization 
print("\n" + "="*70)
print("EXPERIMENT 3: Overfitting vs Regularization")
print("="*70)

print("Training WITHOUT Regularization...")
model_no_reg = MLP(input_size=2, hidden_layers=[20, 15, 10], output_size=1,
                   hidden_activation='relu', output_activation='sigmoid')
trainer_no = Trainer(model_no_reg, learning_rate=0.1, epochs=1000, l2_lambda=0.0)
history_no = trainer_no.train(X_train, y_train, X_test, y_test, verbose=False)

print("Training WITH L2 Regularization...")
model_reg = MLP(input_size=2, hidden_layers=[20, 15, 10], output_size=1,
                hidden_activation='relu', output_activation='sigmoid')
trainer_reg = Trainer(model_reg, learning_rate=0.1, epochs=1000, l2_lambda=0.05)
history_reg = trainer_reg.train(X_train, y_train, X_test, y_test, verbose=False)

print(f"\nWithout Regularization → Test Acc: {np.mean(model_no_reg.predict(X_test) == y_test):.4f}")
print(f"With L2 Regularization → Test Acc: {np.mean(model_reg.predict(X_test) == y_test):.4f}")

viz.plot_training_history(history_no, title="Training Without Regularization")
viz.plot_training_history(history_reg, title="Training With L2 Regularization")

print("\n All Mandatory Experiments Completed!")