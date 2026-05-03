# 🧠 NeuroLab — Neural Networks Educational Platform

> Build it. Break it. Understand it.

An interactive platform to learn how neural networks work — built entirely from scratch using Python and NumPy. No TensorFlow, no PyTorch, no black boxes.

🚀 **Live Demo → [neurolab.streamlit.app]https://neural-networks-educational-platform-bxj4k3a53qk7et6difbqqp.streamlit.app/**

---

## Project Structure

```
neural-platform/
│
├── backend/
│   ├── models/
│   │   ├── perceptron.py        # Historical + Modern Perceptron
│   │   └── mlp.py               # Full MLP (forward, backprop, dropout, L2)
│   │
│   ├── utils/
│   │   ├── activation.py        # Sigmoid, ReLU, Tanh, Softmax + derivatives
│   │   ├── loss.py              # MSE, Binary CE, Categorical CE + gradients
│   │   ├── metrics.py           # Accuracy, F1, Confusion Matrix, R²
│   │   ├── datasets.py          # Dataset generators + CSV loader
│   │   └── visualization.py     # All matplotlib plots
│   │
│   └── __init__.py
│
├── frontend/
│   └── app.py                   # Streamlit app (5 tabs)
│
├── EDAs/                        # All saved experiment visualizations
│   ├── 0_EDA/
│   ├── 1_Activations/
│   ├── 2_Exp1_Perceptron_vs_MLP/
│   ├── 3_Exp2_Layer_Depth/
│   ├── 4_Exp3_Regularization/
│   ├── 5_Multiclass_Regression/
│   ├── 6_Weight_Init_LR/
│   └── 7_Optimizer_Variants/
│
├── generate_edas.py           
├── requirements.txt
└── README.md
```

---

## What's Inside

| Tab | Description |
|---|---|
| 🎯 Train & Evaluate | Configure any model, train it, see metrics + visualizations |
| ⚔️ Exp 1: Perceptron vs MLP | Why linear models fail on non-linear data |
| 📊 Exp 2: Layer Depth | How depth affects accuracy and boundaries |
| 🔬 Exp 3: Regularization | Overfitting vs L2, Dropout, Early Stopping |
| 📚 Theory | Plain-language reference for every concept |

---

## Stack

Python · NumPy · Streamlit · Matplotlib · scikit-learn (dataset generation only)