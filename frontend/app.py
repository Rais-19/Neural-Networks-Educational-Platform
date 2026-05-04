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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synapse",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0f1117; color: #e0e0e0; }
h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.5px; }
.stButton>button {
    background: linear-gradient(135deg, #6c63ff, #4ecdc4); color: white; border: none;
    border-radius: 8px; font-family: 'Space Mono', monospace; font-weight: 700;
    padding: 0.6rem 1.5rem; width: 100%; transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }
.metric-box { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 10px; padding: 1rem; text-align: center; margin-bottom: 0.5rem; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #6c63ff; }
.metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.section-header { border-left: 3px solid #6c63ff; padding-left: 0.75rem; margin: 1.5rem 0 1rem 0; font-family: 'Space Mono', monospace; font-size: 1rem; color: #c0c0ff; }
.info-box { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.88rem; line-height: 1.6; color: #aaa; margin-bottom: 1rem; }
.stTabs [data-baseweb="tab-list"] { background: #1a1d27; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 0.82rem; color: #888; }
.stTabs [aria-selected="true"] { background: #6c63ff !important; border-radius: 6px; color: white !important; }
div[data-testid="stSidebar"] { background: #12141f; border-right: 1px solid #2a2d3a; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 1rem 0 0.5rem 0;'>
  <h1 style='margin:0; font-size:2rem;'>🧠 Synapse</h1>
  <p style='color:#888; margin:0; font-size:0.95rem;'>
    Interactive Neural Networks Educational Platform — Built from scratch
  </p>
</div>
<hr style='border-color:#2a2d3a; margin: 0.75rem 0 1rem 0;'/>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown('<div class="section-header">Problem Type</div>', unsafe_allow_html=True)
    problem_type = st.selectbox("Problem Type", [
        "Binary Classification", "Multi-class Classification", "Regression"
    ], label_visibility="collapsed")

    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    data_source = st.radio("Source", ["Built-in", "Upload CSV"], horizontal=True)

    # Always-defined defaults so nothing is ever NameError'd downstream
    uploaded_file = None
    target_col    = None
    dataset_name  = None
    noise         = 0.15
    n_samples     = 400
    n_classes     = 3
    test_size     = 0.25

    if data_source == "Built-in":
        if problem_type == "Binary Classification":
            dataset_name = st.selectbox("Dataset", ["moons", "circles", "xor", "linear"])
            noise        = st.slider("Noise", 0.0, 0.5, 0.15, 0.05)
            n_samples    = st.slider("Samples", 100, 1000, 400, 50)

        elif problem_type == "Multi-class Classification":
            dataset_name = st.selectbox("Dataset", ["blobs", "iris"])
            if dataset_name == "blobs":
                n_classes = st.slider("Classes", 2, 5, 3)
                n_samples = st.slider("Samples", 100, 1000, 400, 50)
            else:
                n_classes, n_samples = 3, 150

        else:  # Regression
            dataset_name = st.selectbox("Dataset", ["sine", "quadratic", "linear"])
            noise        = st.slider("Noise", 0.0, 0.5, 0.15, 0.05)
            n_samples    = st.slider("Samples", 100, 1000, 300, 50)

        test_size = st.slider("Test Split", 0.1, 0.4, 0.25, 0.05)

    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        test_size     = st.slider("Test Split", 0.1, 0.4, 0.25, 0.05)

        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                df_preview = pd.read_csv(uploaded_file)

                if df_preview.empty:
                    st.error("The uploaded CSV is empty.")
                    uploaded_file = None
                else:
                    st.write("**Preview:**")
                    st.dataframe(df_preview.head(3), use_container_width=True)
                    all_cols   = df_preview.columns.tolist()
                    target_col = st.selectbox("Target Column", all_cols)

                    numeric_feats = df_preview.drop(columns=[target_col]) \
                                              .select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_feats) == 0:
                        st.error("No numeric feature columns found after removing the target.")
                        uploaded_file = None
                    else:
                        st.caption(f"✅ {len(numeric_feats)} feature(s): {', '.join(numeric_feats)}")
            except Exception as e:
                st.error(f"Could not read file: {e}")
                uploaded_file = None

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)
    model_options = ["MLP (Multi-Layer Perceptron)", "Modern Perceptron"]
    if problem_type == "Binary Classification":
        model_options.append("Historical Perceptron")
    model_type = st.selectbox("Model", model_options)

    arch_str          = None
    hidden_activation = None

    if model_type == "MLP (Multi-Layer Perceptron)":
        arch_str          = st.text_input("Hidden Layers (neurons per layer)", "16,12,8")
        hidden_activation = st.selectbox("Hidden Activation", ["relu", "tanh", "sigmoid"])
    elif model_type == "Modern Perceptron":
        hidden_activation = st.selectbox("Activation", ["sigmoid", "relu", "tanh"])

    # ── Training ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Training</div>', unsafe_allow_html=True)
    learning_rate = st.select_slider("Learning Rate",
                                     options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
                                     value=0.05)
    epochs     = st.slider("Epochs", 100, 5000, 1000, 100)
    batch_size = None
    if model_type == "MLP (Multi-Layer Perceptron)":
        if st.checkbox("Mini-batch GD", value=False):
            batch_size = st.slider("Batch Size", 8, 256, 32, 8)

    # ── Regularization ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Regularization</div>', unsafe_allow_html=True)
    l2_lambda     = st.slider("L2 λ", 0.0, 0.1, 0.0, 0.005)
    dropout_rate  = 0.0
    early_stopping = False
    es_patience   = 0
    if model_type == "MLP (Multi-Layer Perceptron)":
        dropout_rate   = st.slider("Dropout Rate", 0.0, 0.5, 0.0, 0.05)
        early_stopping = st.checkbox("Early Stopping", value=False)
        if early_stopping:
            es_patience = st.slider("Patience", 5, 100, 20, 5)

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_train, tab_exp1, tab_exp2, tab_exp3, tab_theory = st.tabs([
    "🎯 Train & Evaluate",
    "⚔️ Exp 1: Perceptron vs MLP",
    "📊 Exp 2: Layer Depth",
    "🔬 Exp 3: Regularization",
    "📚 Theory"
])

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load and prepare data from sidebar config. Always returns a clean dict."""
    pt_map = {
        "Binary Classification":      "binary",
        "Multi-class Classification": "multiclass",
        "Regression":                 "regression",
    }
    pt = pt_map[problem_type]

    if data_source == "Upload CSV":
        if uploaded_file is None:
            st.error("Please upload a valid CSV file first.")
            st.stop()
        if target_col is None:
            st.error("Please select a target column.")
            st.stop()
        try:
            uploaded_file.seek(0)          # reset pointer — sidebar already read it once
            result = load_csv_dataset(uploaded_file, target_col, pt, test_size)
        except ValueError as e:
            st.error(f"Dataset error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error loading CSV: {e}")
            st.stop()
        ds_label = os.path.splitext(uploaded_file.name)[0]

    else:
        try:
            if pt == "binary":
                X, y  = get_binary_dataset(dataset_name, n_samples, noise)
                n_cls = 2
            elif pt == "multiclass":
                nc = n_classes if dataset_name == "blobs" else 3
                ns = n_samples if dataset_name == "blobs" else 150
                X, y, n_cls = get_multiclass_dataset(dataset_name, nc, ns)
            else:
                X, y  = get_regression_dataset(dataset_name, n_samples, noise)
                n_cls = None

            result = prepare_data(X, y, test_size=test_size, problem_type=pt,
                                  n_classes=n_cls if pt == "multiclass" else None)
        except Exception as e:
            st.error(f"Error generating dataset: {e}")
            st.stop()
        ds_label = dataset_name

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, scaler = result
    return {
        "X_train":     X_train,
        "X_test":      X_test,
        "y_train":     y_train,
        "y_test":      y_test,
        "y_train_raw": y_train_raw,
        "y_test_raw":  y_test_raw,
        "pt":          pt,
        "scaler":      scaler,
        "ds_label":    ds_label,        # always a safe string
        "X_all":       np.vstack([X_train, X_test]),
        "y_all":       np.concatenate([y_train_raw, y_test_raw]),
    }


def build_model(pt, input_size, output_size=None, hidden_list=None, h_act=None, lr=None):
    if model_type == "Historical Perceptron":
        return HistoricalPerceptron(input_size=input_size,
                                    learning_rate=lr or learning_rate)
    elif model_type == "Modern Perceptron":
        return ModernPerceptron(input_size=input_size,
                                activation=h_act or hidden_activation or "sigmoid",
                                learning_rate=lr or learning_rate,
                                l2_lambda=l2_lambda)
    else:
        out_act = {"binary": "sigmoid", "multiclass": "softmax", "regression": "linear"}[pt]
        return MLP(input_size=input_size,
                   hidden_layers=hidden_list or [16, 12, 8],
                   output_size=output_size or 1,
                   hidden_activation=h_act or hidden_activation or "relu",
                   output_activation=out_act,
                   problem_type=pt)


def train_model(model, data):
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"],  data["y_test"]

    if model_type == "Historical Perceptron":
        return model.train(X_tr, y_tr.flatten().astype(int), epochs=epochs, verbose=False)
    elif model_type == "Modern Perceptron":
        return model.train(X_tr, y_tr.flatten(), epochs=epochs, verbose=False)
    else:
        return model.train(
            X_tr, y_tr, X_val=X_te, y_val=y_te,
            epochs=epochs, learning_rate=learning_rate,
            l2_lambda=l2_lambda, dropout_rate=dropout_rate,
            batch_size=batch_size,
            early_stopping_patience=es_patience if early_stopping else 0,
            verbose=False,
        )


def get_preds_and_metrics(model, data):
    pt     = data["pt"]
    X_te   = data["X_test"]
    y_te   = data["y_test_raw"]
    y_pred = model.predict(X_te)
    if pt == "regression":
        return compute_regression_metrics(y_te, y_pred), y_pred
    return compute_classification_metrics(y_te, y_pred), y_pred


def show_metrics(metrics, pt):
    if pt == "regression":
        for col, key, label in zip(st.columns(3),
                                   ["mse", "rmse", "r2"],
                                   ["MSE", "RMSE", "R² Score"]):
            with col:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value">{metrics[key]:.4f}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True)
    else:
        for col, key, label in zip(st.columns(4),
                                   ["accuracy", "precision", "recall", "f1"],
                                   ["Accuracy", "Precision", "Recall", "F1 Score"]):
            with col:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value">{metrics[key]:.4f}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    if train_btn:
        data        = load_data()
        pt          = data["pt"]
        input_size  = data["X_train"].shape[1]
        n_cls       = len(np.unique(data["y_all"]))
        output_size = n_cls if pt == "multiclass" else 1

        hidden_list = None
        if model_type == "MLP (Multi-Layer Perceptron)" and arch_str:
            try:
                hidden_list = [int(x.strip()) for x in arch_str.split(",") if x.strip()]
            except ValueError:
                st.warning("Invalid architecture — using [16,12,8].")
                hidden_list = [16, 12, 8]

        with st.spinner("Training... ⚡"):
            model   = build_model(pt, input_size, output_size, hidden_list)
            history = train_model(model, data)
            metrics, _ = get_preds_and_metrics(model, data)

        # Architecture summary
        st.markdown('<div class="section-header">Model Architecture</div>', unsafe_allow_html=True)
        if model_type == "MLP (Multi-Layer Perceptron)":
            sizes, acts = model.get_architecture_summary()
            parts = [f"Input({sizes[0]})"]
            for i, (s, a) in enumerate(zip(sizes[1:], acts)):
                parts.append(f"→ {'Hidden' if i < len(acts)-1 else 'Output'}({s}, {a})")
            st.code(" ".join(parts), language=None)
        else:
            st.code(f"{model_type} | Input({input_size}) → Output | LR={learning_rate}", language=None)

        # Metrics
        st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
        show_metrics(metrics, pt)
        st.caption(f"Epochs: {len(history['loss'])}  |  Dataset: {data['ds_label']}  |  Features: {input_size}")

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
                    st.info(f"Decision boundary requires 2 input features — your data has {input_size}.")
            with col_b:
                st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
                viz.plot_confusion_matrix(metrics["confusion_matrix"], metrics["classes"])
        else:
            st.markdown('<div class="section-header">Regression Results</div>', unsafe_allow_html=True)
            viz.plot_regression_results(data["y_all"], model.predict(data["X_all"]),
                                        title=f"Regression — {data['ds_label']}")

        st.success("✅ Training complete!")

    else:
        st.markdown("""
        <div class="info-box">
        <b>Welcome to Synapse</b> — configure your experiment in the sidebar and hit <b>Train Model</b>.<br><br>
        <b>What you can explore:</b><br>
        • Binary & multi-class classification + regression<br>
        • Historical Perceptron → Modern Perceptron → MLP<br>
        • Activation functions: ReLU, Sigmoid, Tanh, Softmax<br>
        • L2 regularization, Dropout, Early Stopping<br>
        • Mini-batch gradient descent<br>
        • Decision boundaries, confusion matrices, loss curves<br>
        • Upload your own CSV dataset
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPERIMENT 1: Perceptron vs MLP
# ══════════════════════════════════════════════════════════════════════════════
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
        e1_noise   = st.slider("Noise", 0.0, 0.4, 0.15, 0.05, key="e1_noise")
        e1_epochs  = st.slider("Epochs", 100, 3000, 1000, 100, key="e1_ep")
        e1_lr      = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05, 0.1],
                                      value=0.05, key="e1_lr")
    with col_run:
        run_e1 = st.button("▶ Run Experiment 1", key="btn_e1")

    if run_e1:
        with st.spinner("Running all models..."):
            X, y   = get_binary_dataset(e1_dataset, 400, e1_noise)
            result = prepare_data(X, y, test_size=0.25, problem_type="binary")
            X_tr, X_te, y_tr, y_te, y_tr_r, y_te_r, _ = result
            X_all = np.vstack([X_tr, X_te])
            y_all = np.concatenate([y_tr_r, y_te_r])

            hp      = HistoricalPerceptron(input_size=2, learning_rate=e1_lr)
            hist_hp = hp.train(X_tr, y_tr.flatten().astype(int), epochs=e1_epochs, verbose=False)
            acc_hp  = np.mean(hp.predict(X_te) == y_te_r)

            mp      = ModernPerceptron(input_size=2, activation="sigmoid", learning_rate=e1_lr)
            hist_mp = mp.train(X_tr, y_tr.flatten(), epochs=e1_epochs, verbose=False)
            acc_mp  = np.mean(mp.predict(X_te) == y_te_r)

            mlp      = MLP(2, [16, 12, 8], 1, "relu", "sigmoid", "binary")
            hist_mlp = mlp.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                                 epochs=e1_epochs, learning_rate=e1_lr, verbose=False)
            acc_mlp  = np.mean(mlp.predict(X_te) == y_te_r)

        results_dict = {
            "Historical\nPerceptron": {"accuracy": acc_hp},
            "Modern\nPerceptron":     {"accuracy": acc_mp},
            "MLP\n(3 hidden)":        {"accuracy": acc_mlp},
        }
        histories = {
            "Historical Perceptron": hist_hp,
            "Modern Perceptron":     hist_mp,
            "MLP (3 hidden)":        hist_mlp,
        }

        st.markdown('<div class="section-header">Accuracy Comparison</div>', unsafe_allow_html=True)
        viz.plot_experiment_comparison(results_dict, metric="accuracy",
                                       title="Perceptron vs MLP — Test Accuracy")

        st.markdown('<div class="section-header">Learning Curves</div>', unsafe_allow_html=True)
        viz.plot_loss_comparison(histories, title="Loss & Accuracy Over Training")

        st.markdown('<div class="section-header">Decision Boundaries</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        for col, name, m, acc in zip(
            [col1, col2, col3],
            ["Historical Perceptron", "Modern Perceptron", "MLP (3 hidden)"],
            [hp, mp, mlp],
            [acc_hp, acc_mp, acc_mlp]
        ):
            with col:
                st.markdown(f"**{name}** — Acc: `{acc:.4f}`")
                viz.plot_decision_boundary(m, X_all, y_all, title=name, problem_type="binary")

        st.markdown("""
        <div class="info-box">
        <b>💡 Key Insight:</b> The Perceptron draws a straight line — it cannot solve XOR/moons/circles.
        The MLP bends the boundary by composing non-linear transformations across layers.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPERIMENT 2: Layer Depth
# ══════════════════════════════════════════════════════════════════════════════
with tab_exp2:
    st.markdown("### 📊 Experiment 2: Effect of Layer Depth")
    st.markdown("""
    <div class="info-box">
    Train MLPs with 1 to 4 hidden layers on the same dataset.
    Observe how depth affects accuracy, convergence speed, and decision boundary complexity.
    </div>
    """, unsafe_allow_html=True)

    col_cfg2, col_run2 = st.columns([2, 1])
    with col_cfg2:
        e2_dataset    = st.selectbox("Dataset", ["moons", "circles", "xor"], key="e2_ds")
        e2_activation = st.selectbox("Activation", ["relu", "tanh", "sigmoid"], key="e2_act")
        e2_epochs     = st.slider("Epochs", 100, 3000, 1000, 100, key="e2_ep")
        e2_lr         = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05, 0.1],
                                         value=0.05, key="e2_lr")
    with col_run2:
        run_e2 = st.button("▶ Run Experiment 2", key="btn_e2")

    if run_e2:
        with st.spinner("Training 4 architectures..."):
            X, y   = get_binary_dataset(e2_dataset, 400, 0.15)
            result = prepare_data(X, y, test_size=0.25, problem_type="binary")
            X_tr, X_te, y_tr, y_te, y_tr_r, y_te_r, _ = result
            X_all = np.vstack([X_tr, X_te])
            y_all = np.concatenate([y_tr_r, y_te_r])

            architectures  = {
                "1 Hidden [32]":      [32],
                "2 Hidden [16,16]":   [16, 16],
                "3 Hidden [12,12,8]": [12, 12, 8],
                "4 Hidden [8,8,8,8]": [8, 8, 8, 8],
            }
            results_dict   = {}
            histories      = {}
            models_trained = {}

            for name, arch in architectures.items():
                m    = MLP(2, arch, 1, e2_activation, "sigmoid", "binary")
                hist = m.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                               epochs=e2_epochs, learning_rate=e2_lr, verbose=False)
                acc  = np.mean(m.predict(X_te) == y_te_r)
                results_dict[name]   = {"accuracy": acc}
                histories[name]      = hist
                models_trained[name] = m

        st.markdown('<div class="section-header">Accuracy by Depth</div>', unsafe_allow_html=True)
        viz.plot_experiment_comparison(results_dict, "accuracy", "Accuracy vs. Depth")

        st.markdown('<div class="section-header">Learning Curves by Depth</div>', unsafe_allow_html=True)
        viz.plot_loss_comparison(histories, title="Training Dynamics — Layer Depth")

        st.markdown('<div class="section-header">Decision Boundaries by Depth</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (name, m) in zip(cols, models_trained.items()):
            with col:
                viz.plot_decision_boundary(m, X_all, y_all, title=name, problem_type="binary")

        st.markdown("""
        <div class="info-box">
        <b>💡 Key Insight:</b> Deeper is not always better.
        The right depth depends on data complexity — extra layers can overfit on simple problems.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPERIMENT 3: Regularization
# ══════════════════════════════════════════════════════════════════════════════
with tab_exp3:
    st.markdown("### 🔬 Experiment 3: Overfitting vs Regularization")
    st.markdown("""
    <div class="info-box">
    Deliberately overfit an MLP, then compare L2, Dropout, and Early Stopping.
    Observe the train vs validation loss gap — the signature of overfitting.
    </div>
    """, unsafe_allow_html=True)

    col_cfg3, col_run3 = st.columns([2, 1])
    with col_cfg3:
        e3_dataset = st.selectbox("Dataset", ["moons", "circles"], key="e3_ds")
        e3_noise   = st.slider("Noise", 0.0, 0.5, 0.2, 0.05, key="e3_noise")
        e3_epochs  = st.slider("Epochs", 200, 5000, 2000, 100, key="e3_ep")
        e3_arch    = st.text_input("Architecture (prone to overfit)", "64,64,32", key="e3_arch")
    with col_run3:
        run_e3 = st.button("▶ Run Experiment 3", key="btn_e3")

    if run_e3:
        with st.spinner("Training 4 variants..."):
            X, y   = get_binary_dataset(e3_dataset, 200, e3_noise)
            result = prepare_data(X, y, test_size=0.3, problem_type="binary")
            X_tr, X_te, y_tr, y_te, y_tr_r, y_te_r, _ = result

            try:
                arch = [int(x.strip()) for x in e3_arch.split(",") if x.strip()]
            except ValueError:
                arch = [64, 64, 32]

            variants = {
                "No Reg":      dict(l2_lambda=0.0,  dropout_rate=0.0, es=0),
                "L2 (λ=0.01)":dict(l2_lambda=0.01, dropout_rate=0.0, es=0),
                "Dropout 0.3": dict(l2_lambda=0.0,  dropout_rate=0.3, es=0),
                "Early Stop":  dict(l2_lambda=0.0,  dropout_rate=0.0, es=30),
            }
            results_dict = {}
            histories    = {}

            for name, cfg in variants.items():
                m    = MLP(2, arch, 1, "relu", "sigmoid", "binary")
                hist = m.train(X_tr, y_tr, X_val=X_te, y_val=y_te,
                               epochs=e3_epochs, learning_rate=0.05,
                               l2_lambda=cfg["l2_lambda"],
                               dropout_rate=cfg["dropout_rate"],
                               early_stopping_patience=cfg["es"],
                               verbose=False)
                tr_acc = np.mean(m.predict(X_tr) == y_tr_r)
                te_acc = np.mean(m.predict(X_te) == y_te_r)
                results_dict[name] = {
                    "train_acc":  tr_acc,
                    "test_acc":   te_acc,
                    "gap":        tr_acc - te_acc,
                    "epochs_run": len(hist["loss"]),
                }
                histories[name] = hist

        st.markdown('<div class="section-header">Overfitting Summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Variant":    list(results_dict.keys()),
            "Train Acc":  [f"{v['train_acc']:.4f}"  for v in results_dict.values()],
            "Test Acc":   [f"{v['test_acc']:.4f}"   for v in results_dict.values()],
            "Gap":        [f"{v['gap']:.4f}"         for v in results_dict.values()],
            "Epochs Run": [v["epochs_run"]           for v in results_dict.values()],
        }), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Train vs Validation Loss</div>', unsafe_allow_html=True)
        viz.plot_loss_comparison(histories, title="Overfitting vs Regularization — Loss Curves")
        viz.plot_experiment_comparison(
            {k: {"accuracy": v["test_acc"]} for k, v in results_dict.items()},
            "accuracy", "Test Accuracy by Regularization Strategy")

        st.markdown("""
        <div class="info-box">
        <b>💡 Key Insight:</b> No Reg memorizes training data (high train acc, lower test acc).
        L2 constrains weights. Dropout forces redundant learning. Early Stopping halts before memorization.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — THEORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.markdown("### 📚 Theory Reference")
    st.markdown('<div class="info-box">Quick reference for every core concept in this platform.</div>',
                unsafe_allow_html=True)

    with st.expander("🔵 The Perceptron & Its Limits"):
        st.markdown("""
        The **Historical Perceptron** (Rosenblatt, 1958) uses a step activation and the
        *perceptron update rule*: adjust weights only on misclassified samples.
        Converges only if data is **linearly separable**.

        **XOR problem**: no single line can separate the 4 points → perceptron fails → first AI winter.

        The **Modern Perceptron** replaces step with sigmoid/ReLU and uses gradient descent.
        """)

    with st.expander("🟣 Multi-Layer Perceptron & Universal Approximation"):
        st.markdown("""
        An MLP with ≥1 hidden layer and non-linear activations is a **universal function approximator**
        (Cybenko, 1989).

        **Forward prop**: Z = W·A_prev + b → A = activation(Z)

        **Backprop**: chain rule → δₗ = (Wₗ₊₁ᵀ · δₗ₊₁) ⊙ activation'(Zₗ)
        """)

    with st.expander("🟢 Activation Functions"):
        st.markdown("""
        | Function | Formula | Use case |
        |---|---|---|
        | Sigmoid | 1/(1+e⁻ˣ) | Binary output |
        | ReLU | max(0,x) | Hidden layers (default) |
        | Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | Hidden, zero-centered |
        | Softmax | eˣᵢ/Σeˣⱼ | Multi-class output |
        | Linear | x | Regression output |

        **Vanishing gradient**: Sigmoid/tanh saturate → tiny gradients → deep layers stop learning. ReLU avoids this.
        """)

    with st.expander("🟡 Loss Functions"):
        st.markdown("""
        | Problem | Loss | Formula |
        |---|---|---|
        | Regression | MSE | (1/n) Σ(ŷ−y)² |
        | Binary | BCE | −(y log ŷ + (1−y) log(1−ŷ)) |
        | Multi-class | CCE | −Σ yₖ log ŷₖ |
        """)

    with st.expander("🔴 Regularization & Generalization"):
        st.markdown("""
        **Overfitting**: low train loss, high val loss — model memorized noise.

        - **L2**: adds λ‖W‖² → penalizes large weights
        - **Dropout**: randomly zeros p% of neurons during training
        - **Early Stopping**: halt when val loss stops improving

        **Bias-Variance**: high bias = underfitting, high variance = overfitting.
        """)

    with st.expander("⚡ Optimization"):
        st.markdown("""
        **Gradient Descent**: θ ← θ − α·∇L(θ)

        | Variant | Updates on | Pros | Cons |
        |---|---|---|---|
        | Batch GD | Full dataset | Stable | Slow |
        | SGD | 1 sample | Fast | Noisy |
        | Mini-batch | k samples | Balanced | Extra hyperparameter |

        Typical learning rate range: 0.001–0.1.
        """)