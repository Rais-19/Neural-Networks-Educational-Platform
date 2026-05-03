import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd

from backend.models.perceptron import HistoricalPerceptron, ModernPerceptron
from backend.models.mlp import MLP
from backend.utils.datasets import (get_binary_dataset, get_multiclass_dataset,
                                     get_regression_dataset, prepare_data, load_csv_dataset)
from backend.utils.metrics import (compute_classification_metrics, compute_regression_metrics)
from backend.utils import visualization as viz

#  Page config 
st.set_page_config(
    page_title="NeuroLab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.5px;
}
.stButton>button {
    background: linear-gradient(135deg, #6c63ff, #4ecdc4);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }
.metric-box {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #6c63ff;
}
.metric-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.section-header {
    border-left: 3px solid #6c63ff;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #c0c0ff;
}
.info-box {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #aaa;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab-list"] {
    background: #1a1d27;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #888;
}
.stTabs [aria-selected="true"] {
    background: #6c63ff !important;
    border-radius: 6px;
    color: white !important;
}
div[data-testid="stSidebar"] {
    background: #12141f;
    border-right: 1px solid #2a2d3a;
}
</style>
""", unsafe_allow_html=True)

#  Header
st.markdown("""
<div style='padding: 1rem 0 0.5rem 0;'>
  <h1 style='margin:0; font-size:2rem;'>🧠 NeuroLab</h1>
  <p style='color:#888; margin:0; font-size:0.95rem;'>
    Interactive Neural Networks Educational Platform — Built from scratch
  </p>
</div>
<hr style='border-color:#2a2d3a; margin: 0.75rem 0 1rem 0;'/>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("##  Configuration")
    st.markdown("---")

    # Problem Type
    st.markdown('<div class="section-header">Problem Type</div>', unsafe_allow_html=True)
    problem_type = st.selectbox("", ["Binary Classification",
                                     "Multi-class Classification",
                                     "Regression"], key="prob_type", label_visibility="collapsed")

    #  Dataset
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    data_source = st.radio("Source", ["Built-in", "Upload CSV"], horizontal=True)

    if data_source == "Built-in":
        if problem_type == "Binary Classification":
            dataset_name = st.selectbox("Dataset", ["moons", "circles", "xor", "linear"])
            noise = st.slider("Noise", 0.0, 0.5, 0.15, 0.05)
            n_samples = st.slider("Samples", 100, 1000, 400, 50)
        elif problem_type == "Multi-class Classification":
            dataset_name = st.selectbox("Dataset", ["blobs", "iris"])
            n_classes = st.slider("Classes", 2, 5, 3) if dataset_name == "blobs" else 3
            n_samples = st.slider("Samples", 100, 1000, 400, 50) if dataset_name == "blobs" else 150
        else:
            dataset_name = st.selectbox("Dataset", ["sine", "quadratic", "linear"])
            noise = st.slider("Noise", 0.0, 0.5, 0.15, 0.05)
            n_samples = st.slider("Samples", 100, 1000, 300, 50)
        test_size = st.slider("Test Split", 0.1, 0.4, 0.25, 0.05)
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        target_col = st.text_input("Target Column Name", "target")
        test_size = st.slider("Test Split", 0.1, 0.4, 0.25, 0.05)

    #  Model
    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)

    model_options = ["MLP (Multi-Layer Perceptron)", "Modern Perceptron"]
    if problem_type == "Binary Classification":
        model_options.append("Historical Perceptron")
    model_type = st.selectbox("Model", model_options)

    if model_type == "MLP (Multi-Layer Perceptron)":
        arch_str = st.text_input("Hidden Layers (neurons per layer)", "16,12,8",
                                  help="e.g. '32,16' means 2 hidden layers")
        hidden_activation = st.selectbox("Hidden Activation", ["relu", "tanh", "sigmoid"])
    elif model_type == "Modern Perceptron":
        hidden_activation = st.selectbox("Activation", ["sigmoid", "relu", "tanh"])
    else:
        hidden_activation = None

    # Training
    st.markdown('<div class="section-header">Training</div>', unsafe_allow_html=True)
    learning_rate = st.select_slider("Learning Rate",
                                     options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
                                     value=0.05)
    epochs = st.slider("Epochs", 100, 5000, 1000, 100)

    if model_type == "MLP (Multi-Layer Perceptron)":
        use_minibatch = st.checkbox("Mini-batch GD", value=False)
        batch_size = st.slider("Batch Size", 8, 256, 32, 8) if use_minibatch else None
    else:
        batch_size = None

    # Regularization
    st.markdown('<div class="section-header">Regularization</div>', unsafe_allow_html=True)
    l2_lambda = st.slider("L2 λ", 0.0, 0.1, 0.0, 0.005,
                          help="0 = no regularization")
    if model_type == "MLP (Multi-Layer Perceptron)":
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.0, 0.05)
        early_stopping = st.checkbox("Early Stopping", value=False)
        es_patience = st.slider("Patience", 5, 100, 20, 5) if early_stopping else 0
    else:
        dropout_rate = 0.0
        early_stopping = False
        es_patience = 0

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", type="primary")

# MAIN TABS
tab_train, tab_exp1, tab_exp2, tab_exp3, tab_theory = st.tabs([
    "🎯 Train & Evaluate",
    "⚔️ Exp 1: Perceptron vs MLP",
    "📊 Exp 2: Layer Depth",
    "🔬 Exp 3: Regularization",
    "📚 Theory"
])

# HELPERS
def load_data():
    """Load and prepare data based on sidebar config. Returns dicts."""
    pt_map = {
        "Binary Classification": "binary",
        "Multi-class Classification": "multiclass",
        "Regression": "regression"
    }
    pt = pt_map[problem_type]

    if data_source == "Upload CSV":
        if uploaded_file is None:
            st.error("Please upload a CSV file.")
            st.stop()
        result = load_csv_dataset(uploaded_file, target_col, pt, test_size)
    else:
        if pt == "binary":
            X, y = get_binary_dataset(dataset_name, n_samples, noise)
            n_cls = 2
        elif pt == "multiclass":
            X, y, n_cls = get_multiclass_dataset(dataset_name,
                                                   n_classes if dataset_name == "blobs" else 3,
                                                   n_samples if dataset_name == "blobs" else 150)
        else:
            X, y = get_regression_dataset(dataset_name, n_samples, noise)
            n_cls = None

        result = prepare_data(X, y, test_size=test_size, problem_type=pt,
                              n_classes=n_cls if pt == "multiclass" else None)

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, scaler = result
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_train_raw": y_train_raw, "y_test_raw": y_test_raw,
        "pt": pt, "scaler": scaler,
        "X_all": np.vstack([X_train, X_test]),
        "y_all": np.concatenate([y_train_raw, y_test_raw]),
    }


def build_model(pt, input_size, output_size=None, hidden_list=None,
                h_act=None, lr=None, l2=0.0):
    if model_type == "Historical Perceptron":
        return HistoricalPerceptron(input_size=input_size, learning_rate=lr or learning_rate)
    elif model_type == "Modern Perceptron":
        return ModernPerceptron(input_size=input_size,
                                activation=h_act or hidden_activation,
                                learning_rate=lr or learning_rate,
                                l2_lambda=l2)
    else:
        out_act = {'binary': 'sigmoid', 'multiclass': 'softmax', 'regression': 'linear'}[pt]
        return MLP(input_size=input_size,
                   hidden_layers=hidden_list or [16, 12, 8],
                   output_size=output_size or 1,
                   hidden_activation=h_act or hidden_activation,
                   output_activation=out_act,
                   problem_type=pt)


def train_model(model, data):
    pt = data["pt"]
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    if model_type == "Historical Perceptron":
        y_flat = y_tr.flatten().astype(int)
        history = model.train(X_tr, y_flat, epochs=epochs, verbose=False)
    elif model_type == "Modern Perceptron":
        y_flat = y_tr.flatten()
        history = model.train(X_tr, y_flat, epochs=epochs, verbose=False)
    else:
        history = model.train(
            X_tr, y_tr, X_val=X_te, y_val=y_te,
            epochs=epochs, learning_rate=learning_rate,
            l2_lambda=l2_lambda, dropout_rate=dropout_rate,
            batch_size=batch_size,
            early_stopping_patience=es_patience if early_stopping else 0,
            verbose=False
        )
    return history


def get_preds_and_metrics(model, data):
    pt = data["pt"]
    X_te, y_te_raw = data["X_test"], data["y_test_raw"]

    if pt == "regression":
        y_pred = model.predict(X_te)
        return compute_regression_metrics(y_te_raw, y_pred), y_pred
    else:
        y_pred = model.predict(X_te)
        return compute_classification_metrics(y_te_raw, y_pred), y_pred


def show_metrics(metrics, pt):
    if pt == "regression":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{metrics["mse"]:.4f}</div>'
                        f'<div class="metric-label">MSE</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{metrics["rmse"]:.4f}</div>'
                        f'<div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{metrics["r2"]:.4f}</div>'
                        f'<div class="metric-label">R² Score</div></div>', unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        for col, key, label in [(c1, "accuracy", "Accuracy"), (c2, "precision", "Precision"),
                                 (c3, "recall", "Recall"), (c4, "f1", "F1 Score")]:
            with col:
                st.markdown(f'<div class="metric-box">'
                            f'<div class="metric-value">{metrics[key]:.4f}</div>'
                            f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


# TAB 1 — TRAIN & EVALUATE
with tab_train:
    if train_btn:
        data = load_data()
        pt = data["pt"]
        input_size = data["X_train"].shape[1]
        n_cls = len(np.unique(data["y_all"]))
        output_size = n_cls if pt == "multiclass" else 1

        if model_type == "MLP (Multi-Layer Perceptron)":
            try:
                hidden_list = [int(x.strip()) for x in arch_str.split(",")]
            except:
                hidden_list = [16, 12, 8]
        else:
            hidden_list = None

        with st.spinner("Training... ⚡"):
            model = build_model(pt, input_size, output_size, hidden_list)
            history = train_model(model, data)
            metrics, y_pred = get_preds_and_metrics(model, data)

        # Architecture summary
        st.markdown('<div class="section-header">Model Architecture</div>', unsafe_allow_html=True)
        if model_type == "MLP (Multi-Layer Perceptron)":
            sizes, acts = model.get_architecture_summary()
            arch_parts = [f"Input({sizes[0]})"]
            for i, (s, a) in enumerate(zip(sizes[1:], acts)):
                arch_parts.append(f"→ {'Hidden' if i < len(acts)-1 else 'Output'}({s}, {a})")
            st.code(" ".join(arch_parts), language=None)
        else:
            st.code(f"{model_type} | Input({input_size}) → Output(1) | LR={learning_rate}", language=None)

        # Metrics 
        st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
        show_metrics(metrics, pt)
        st.caption(f"Epochs completed: {len(history['loss'])}")

        # Training curves 
        st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
        viz.plot_training_history(history, title=f"Training — {model_type}")

        # Visualizations
        if pt != "regression":
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="section-header">Decision Boundary</div>', unsafe_allow_html=True)
                if input_size == 2:
                    viz.plot_decision_boundary(model, data["X_all"], data["y_all"],
                                              title="Decision Boundary", problem_type=pt)
                else:
                    st.info("Decision boundary visualization requires exactly 2 input features.")
            with col_b:
                st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
                viz.plot_confusion_matrix(metrics["confusion_matrix"], metrics["classes"])
        else:
            st.markdown('<div class="section-header">Regression Results</div>', unsafe_allow_html=True)
            y_pred_all = model.predict(data["X_all"])
            viz.plot_regression_results(data["y_all"], y_pred_all,
                                        title=f"Regression — {dataset_name}")

        st.success("Training complete!")

    else:
        st.markdown("""
        <div class="info-box">
        <b>Welcome to NeuroLab</b> — configure your experiment in the sidebar and hit <b>Train Model</b>.<br><br>
        <b>What you can explore:</b><br>
        • Binary & multi-class classification + regression<br>
        • Historical Perceptron → Modern Perceptron → MLP<br>
        • Activation functions: ReLU, Sigmoid, Tanh, Softmax<br>
        • L2 regularization, Dropout, Early Stopping<br>
        • Mini-batch gradient descent<br>
        • Decision boundaries, confusion matrices, loss curves
        </div>
        """, unsafe_allow_html=True)

# TAB 2 — EXPERIMENT 1: Perceptron vs MLP
with tab_exp1:
    st.markdown("### ⚔️ Experiment 1: Perceptron vs MLP")
    st.markdown("""
    <div class="info-box">
    Compare the <b>Historical Perceptron</b>, <b>Modern Perceptron</b>, and a <b>3-layer MLP</b>
    on non-linearly separable data. The XOR and Moons datasets are deliberately chosen because
    a perceptron <i>cannot</i> solve them — this is the fundamental motivation for deep learning.
    </div>
    """, unsafe_allow_html=True)

    col_cfg, col_run = st.columns([2, 1])
    with col_cfg:
        e1_dataset = st.selectbox("Dataset", ["moons", "circles", "xor"], key="e1_ds")
        e1_noise = st.slider("Noise", 0.0, 0.4, 0.15, 0.05, key="e1_noise")
        e1_epochs = st.slider("Epochs", 100, 3000, 1000, 100, key="e1_ep")
        e1_lr = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05, 0.1], value=0.05, key="e1_lr")
    with col_run:
        run_e1 = st.button("▶ Run Experiment 1", key="btn_e1")

    if run_e1:
        with st.spinner("Running all models..."):
            X, y = get_binary_dataset(e1_dataset, 400, e1_noise)
            result = prepare_data(X, y, test_size=0.25, problem_type='binary')
            X_tr, X_te, y_tr, y_te, y_tr_r, y_te_r, _ = result
            X_all = np.vstack([X_tr, X_te])
            y_all = np.concatenate([y_tr_r, y_te_r])

            results_dict = {}
            histories = {}
            models_trained = {}

            # Historical Perceptron
            hp = HistoricalPerceptron(input_size=2, learning_rate=e1_lr)
            hist_hp = hp.train(X_tr, y_tr.flatten().astype(int), epochs=e1_epochs, verbose=False)
            acc_hp = np.mean(hp.predict(X_te) == y_te_r)
            results_dict["Historical\nPerceptron"] = {"accuracy": acc_hp}
            histories["Historical Perceptron"] = hist_hp
            models_trained["Historical Perceptron"] = hp

            # Modern Perceptron
            mp = ModernPerceptron(input_size=2, activation='sigmoid', learning_rate=e1_lr)
            hist_mp = mp.train(X_tr, y_tr.flatten(), epochs=e1_epochs, verbose=False)
            acc_mp = np.mean(mp.predict(X_te) == y_te_r)
            results_dict["Modern\nPerceptron"] = {"accuracy": acc_mp}
            histories["Modern Perceptron"] = hist_mp
            models_trained["Modern Perceptron"] = mp

            # MLP
            mlp = MLP(2, [16, 12, 8], 1, 'relu', 'sigmoid', 'binary')
            hist_mlp = mlp.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                                  epochs=e1_epochs, learning_rate=e1_lr, verbose=False)
            acc_mlp = np.mean(mlp.predict(X_te) == y_te_r)
            results_dict["MLP\n(3 hidden)"] = {"accuracy": acc_mlp}
            histories["MLP (3 hidden)"] = hist_mlp
            models_trained["MLP (3 hidden)"] = mlp

        # Results bar chart
        st.markdown('<div class="section-header">Accuracy Comparison</div>', unsafe_allow_html=True)
        viz.plot_experiment_comparison(results_dict, metric='accuracy',
                                       title='Perceptron vs MLP — Test Accuracy')

        # Loss curves
        st.markdown('<div class="section-header">Learning Curves</div>', unsafe_allow_html=True)
        viz.plot_loss_comparison(histories, title='Loss & Accuracy Over Training')

        # Decision boundaries side by side
        st.markdown('<div class="section-header">Decision Boundaries</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        for col, name, m in zip([col1, col2, col3],
                                 ["Historical\nPerceptron", "Modern\nPerceptron", "MLP\n(3 hidden)"],
                                 [hp, mp, mlp]):
            with col:
                st.markdown(f"**{name}** — Acc: `{results_dict[name]['accuracy']:.4f}`")
                viz.plot_decision_boundary(m, X_all, y_all,
                                           title=name, problem_type='binary')

        # Key insight
        st.markdown("""
        <div class="info-box">
        <b> Key Insight:</b> The Historical and Modern Perceptron are linear classifiers —
        they can only draw a straight line through the data. On XOR/moons/circles, no straight
        line separates the classes, so accuracy hovers near 50%. The MLP learns non-linear
        decision boundaries by composing multiple linear transformations with non-linear activations.
        </div>
        """, unsafe_allow_html=True)

# TAB 3 — EXPERIMENT 2: Layer Depth
with tab_exp2:
    st.markdown("###  Experiment 2: Effect of Layer Depth")
    st.markdown("""
    <div class="info-box">
    Train MLPs with 1 to 4 hidden layers (same total neurons) on the same dataset.
    Observe how depth affects accuracy, convergence speed, and decision boundary complexity.
    </div>
    """, unsafe_allow_html=True)

    col_cfg2, col_run2 = st.columns([2, 1])
    with col_cfg2:
        e2_dataset = st.selectbox("Dataset", ["moons", "circles", "xor"], key="e2_ds")
        e2_activation = st.selectbox("Activation", ["relu", "tanh", "sigmoid"], key="e2_act")
        e2_epochs = st.slider("Epochs", 100, 3000, 1000, 100, key="e2_ep")
        e2_lr = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05, 0.1], value=0.05, key="e2_lr")
    with col_run2:
        run_e2 = st.button(" Run Experiment 2", key="btn_e2")

    if run_e2:
        with st.spinner("Training 4 architectures..."):
            X, y = get_binary_dataset(e2_dataset, 400, 0.15)
            result = prepare_data(X, y, test_size=0.25, problem_type='binary')
            X_tr, X_te, y_tr, y_te, y_tr_r, y_te_r, _ = result
            X_all = np.vstack([X_tr, X_te])
            y_all = np.concatenate([y_tr_r, y_te_r])

            architectures = {
                "1 Hidden\n[32]": [32],
                "2 Hidden\n[16,16]": [16, 16],
                "3 Hidden\n[12,12,8]": [12, 12, 8],
                "4 Hidden\n[8,8,8,8]": [8, 8, 8, 8],
            }

            results_dict = {}
            histories = {}
            models_trained = {}

            for name, arch in architectures.items():
                m = MLP(2, arch, 1, e2_activation, 'sigmoid', 'binary')
                hist = m.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                               epochs=e2_epochs, learning_rate=e2_lr, verbose=False)
                acc = np.mean(m.predict(X_te) == y_te_r)
                results_dict[name] = {"accuracy": acc}
                histories[name.replace("\n", " ")] = hist
                models_trained[name] = m

        st.markdown('<div class="section-header">Accuracy by Depth</div>', unsafe_allow_html=True)
        viz.plot_experiment_comparison(results_dict, 'accuracy', 'Accuracy vs. Number of Hidden Layers')

        st.markdown('<div class="section-header">Learning Curves by Depth</div>', unsafe_allow_html=True)
        viz.plot_loss_comparison({k.replace("\n", " "): v for k, v in histories.items()},
                                  title='Training Dynamics — Layer Depth')

        st.markdown('<div class="section-header">Decision Boundaries by Depth</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (name, m) in zip(cols, models_trained.items()):
            with col:
                viz.plot_decision_boundary(m, X_all, y_all,
                                           title=name.replace("\n", " "), problem_type='binary')

        st.markdown("""
        <div class="info-box">
        <b>💡 Key Insight:</b> Deeper networks can learn more complex decision boundaries,
        but they don't always perform better. On simple datasets, extra layers may overfit
        or just add training instability. The right depth depends on the data complexity.
        </div>
        """, unsafe_allow_html=True)

# TAB 4 — EXPERIMENT 3: Regularization
with tab_exp3:
    st.markdown("### 🔬 Experiment 3: Overfitting vs Regularization")
    st.markdown("""
    <div class="info-box">
    Deliberately overfit an MLP, then compare the effect of L2 regularization,
    Early Stopping, and Dropout. Observe the train vs validation loss gap — the
    signature of overfitting.
    </div>
    """, unsafe_allow_html=True)

    col_cfg3, col_run3 = st.columns([2, 1])
    with col_cfg3:
        e3_dataset = st.selectbox("Dataset", ["moons", "circles"], key="e3_ds")
        e3_noise = st.slider("Noise", 0.0, 0.5, 0.2, 0.05, key="e3_noise")
        e3_epochs = st.slider("Epochs", 200, 5000, 2000, 100, key="e3_ep")
        e3_arch = st.text_input("Architecture (prone to overfit)", "64,64,32", key="e3_arch")
    with col_run3:
        run_e3 = st.button("▶ Run Experiment 3", key="btn_e3")

    if run_e3:
        with st.spinner("Training 4 variants..."):
            X, y = get_binary_dataset(e3_dataset, 200, e3_noise)  # small dataset → easier to overfit
            result = prepare_data(X, y, test_size=0.3, problem_type='binary')
            X_tr, X_te, y_tr, y_te, y_tr_r, y_te_r, _ = result

            try:
                arch = [int(x.strip()) for x in e3_arch.split(",")]
            except:
                arch = [64, 64, 32]

            variants = {
                "No Reg": dict(l2_lambda=0.0, dropout_rate=0.0, es=0),
                "L2 (λ=0.01)": dict(l2_lambda=0.01, dropout_rate=0.0, es=0),
                "Dropout 0.3": dict(l2_lambda=0.0, dropout_rate=0.3, es=0),
                "Early Stop": dict(l2_lambda=0.0, dropout_rate=0.0, es=30),
            }

            results_dict = {}
            histories = {}

            for name, cfg in variants.items():
                m = MLP(2, arch, 1, 'relu', 'sigmoid', 'binary')
                hist = m.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                               epochs=e3_epochs, learning_rate=0.05,
                               l2_lambda=cfg["l2_lambda"],
                               dropout_rate=cfg["dropout_rate"],
                               early_stopping_patience=cfg["es"],
                               verbose=False)
                train_acc = np.mean(m.predict(X_tr) == y_tr_r)
                test_acc = np.mean(m.predict(X_te) == y_te_r)
                results_dict[name] = {
                    "train_acc": train_acc, "test_acc": test_acc,
                    "gap": train_acc - test_acc,
                    "epochs_run": len(hist["loss"])
                }
                histories[name] = hist

        # Summary table
        st.markdown('<div class="section-header">Overfitting Summary</div>', unsafe_allow_html=True)
        df_res = pd.DataFrame({
            "Variant": list(results_dict.keys()),
            "Train Acc": [f"{v['train_acc']:.4f}" for v in results_dict.values()],
            "Test Acc": [f"{v['test_acc']:.4f}" for v in results_dict.values()],
            "Gap (Train-Test)": [f"{v['gap']:.4f}" for v in results_dict.values()],
            "Epochs Run": [v["epochs_run"] for v in results_dict.values()],
        })
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        # Train vs val loss curves for each variant
        st.markdown('<div class="section-header">Train vs Validation Loss</div>', unsafe_allow_html=True)
        viz.plot_loss_comparison(histories, title='Overfitting vs Regularization — Loss Curves')

        # Bar chart of test accuracy
        bar_data = {k: {"accuracy": v["test_acc"]} for k, v in results_dict.items()}
        viz.plot_experiment_comparison(bar_data, "accuracy", "Test Accuracy by Regularization Strategy")

        st.markdown("""
        <div class="info-box">
        <b>💡 Key Insight:</b> <i>No Reg</i> trains to near-perfect train accuracy but its
        test accuracy lags — the gap reveals overfitting. <i>L2</i> penalizes large weights,
        reducing model complexity. <i>Dropout</i> randomly disables neurons during training,
        acting as an ensemble method. <i>Early Stopping</i> halts training when validation
        loss stops improving, preventing the model from memorizing training noise.
        </div>
        """, unsafe_allow_html=True)

# TAB 5 — THEORY
with tab_theory:
    st.markdown("###  Theory Reference")
    st.markdown("""
    <div class="info-box">
    Quick reference for the core concepts implemented in this platform.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔵 The Perceptron & Its Limits"):
        st.markdown("""
        The **Historical Perceptron** (Rosenblatt, 1958) is a binary classifier using a
        step activation and the *perceptron update rule*: adjust weights only on misclassified samples.
        It converges if and only if the data is **linearly separable** — a single hyperplane can
        divide the two classes.

        **XOR problem**: 4 points that no single line can separate → perceptron fails. This led to
        the first "AI winter" until backpropagation was rediscovered in the 1980s.

        The **Modern Perceptron** replaces the step function with a smooth activation (sigmoid, ReLU)
        and uses gradient descent — equivalent to logistic regression for sigmoid output.
        """)

    with st.expander("🟣 Multi-Layer Perceptron & Universal Approximation"):
        st.markdown("""
        An MLP with at least one hidden layer and a non-linear activation is a **universal function
        approximator** (Cybenko, 1989) — it can approximate any continuous function to arbitrary
        precision, given enough neurons.

        **Forward propagation**: Z = W·A_prev + b → A = activation(Z)

        **Backpropagation**: Apply the chain rule from output to input, computing gradients ∂L/∂W
        at each layer. Key equation: δₗ = (Wₗ₊₁ᵀ · δₗ₊₁) ⊙ activation'(Zₗ)
        """)

    with st.expander("🟢 Activation Functions"):
        st.markdown("""
        | Function | Formula | Use case |
        |---|---|---|
        | Sigmoid | 1/(1+e⁻ˣ) | Binary output layer |
        | ReLU | max(0, x) | Hidden layers (default) |
        | Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | Hidden layers, zero-centered |
        | Softmax | eˣᵢ/Σeˣⱼ | Multi-class output layer |
        | Linear | x | Regression output layer |

        **Vanishing gradient problem**: Sigmoid/tanh saturate at large values → gradients
        become tiny → deep layers stop learning. ReLU avoids this in the positive range.
        """)

    with st.expander("🟡 Loss Functions"):
        st.markdown("""
        | Problem | Loss | Formula |
        |---|---|---|
        | Regression | MSE | (1/n) Σ(ŷ − y)² |
        | Binary classification | BCE | −(y log ŷ + (1−y) log(1−ŷ)) |
        | Multi-class | CCE | −Σ yₖ log ŷₖ |

        MSE penalizes large errors quadratically. BCE/CCE are derived from maximum likelihood
        estimation and pair naturally with sigmoid/softmax outputs.
        """)

    with st.expander("🔴 Regularization & Generalization"):
        st.markdown("""
        **Overfitting**: Model memorizes training data, fails on unseen data. Sign: low train loss,
        high validation loss.

        **L2 Regularization**: Adds λ‖W‖² to the loss → penalizes large weights → smoother model.
        Equivalent to a Gaussian prior on weights (Bayesian interpretation).

        **Dropout**: Randomly sets p% of neurons to 0 during training. Forces the network to learn
        redundant representations. At inference, all neurons are active (weights scaled by 1−p).

        **Early Stopping**: Monitor validation loss each epoch. Stop when it starts rising.
        Keeps the model at the best generalization point.

        **Bias-Variance Tradeoff**:
        - High bias = underfitting (too simple model)
        - High variance = overfitting (too complex, memorizes noise)
        - Goal: sweet spot with good generalization
        """)

    with st.expander("⚡ Optimization"):
        st.markdown("""
        **Gradient Descent**: θ ← θ − α · ∇L(θ)

        | Variant | Updates on | Pros | Cons |
        |---|---|---|---|
        | Batch GD | Full dataset | Stable | Slow, memory |
        | Stochastic GD | 1 sample | Fast | Noisy |
        | Mini-batch GD | k samples | Balanced | Hyperparameter k |

        **Learning rate α**: Too high → divergence. Too low → slow convergence.
        Common values: 0.01–0.1. Use learning rate schedules or adaptive optimizers
        (Adam, RMSProp) for better results.
        """)