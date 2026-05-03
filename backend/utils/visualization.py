import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import seaborn as sns


PALETTE = {
    'bg': '#0f1117',
    'surface': '#1a1d27',
    'accent1': '#6c63ff',
    'accent2': '#ff6584',
    'accent3': '#43e97b',
    'text': '#e0e0e0',
    'grid': '#2a2d3a',
}

def _apply_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PALETTE['surface'])
    ax.tick_params(colors=PALETTE['text'], labelsize=9)
    ax.spines[:].set_color(PALETTE['grid'])
    if title:
        ax.set_title(title, color=PALETTE['text'], fontsize=12, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=PALETTE['text'], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE['text'], fontsize=10)
    ax.grid(True, color=PALETTE['grid'], linewidth=0.5, linestyle='--', alpha=0.6)


def plot_training_history(history, title='Training History'):
    has_val = len(history.get('val_loss', [])) > 0
    cols = 2
    fig, axes = plt.subplots(1, cols, figsize=(12, 4))
    fig.patch.set_facecolor(PALETTE['bg'])

    # Loss
    ax = axes[0]
    ax.plot(history['loss'], color=PALETTE['accent1'], linewidth=2, label='Train Loss')
    if has_val:
        ax.plot(history['val_loss'], color=PALETTE['accent2'], linewidth=2,
                linestyle='--', label='Val Loss')
    _apply_style(ax, 'Loss vs Epochs', 'Epoch', 'Loss')
    ax.legend(facecolor=PALETTE['surface'], labelcolor=PALETTE['text'], framealpha=0.8)

    # Accuracy
    ax = axes[1]
    ax.plot(history['accuracy'], color=PALETTE['accent3'], linewidth=2, label='Train Accuracy')
    if has_val:
        ax.plot(history['val_accuracy'], color=PALETTE['accent2'], linewidth=2,
                linestyle='--', label='Val Accuracy')
    _apply_style(ax, 'Accuracy vs Epochs', 'Epoch', 'Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=PALETTE['surface'], labelcolor=PALETTE['text'], framealpha=0.8)

    plt.suptitle(title, color=PALETTE['text'], fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_decision_boundary(model, X, y, title='Decision Boundary', problem_type='binary'):
    if X.shape[1] != 2:
        st.warning("Decision boundary only available for 2D input data.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['surface'])

    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    proba = model.predict_proba(grid)
    if problem_type == 'binary':
        Z = proba.reshape(xx.shape) if proba.ndim == 1 else proba[:, 0].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=50, cmap='RdYlGn', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors='white', linewidths=2)
    elif problem_type == 'multiclass':
        Z = np.argmax(proba, axis=1).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap='tab10', alpha=0.5)
    else:  # regression
        Z = proba.reshape(xx.shape)
        cf = ax.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.7)
        plt.colorbar(cf, ax=ax)

    # Scatter
    colors_pts = [PALETTE['accent1'], PALETTE['accent2'], PALETTE['accent3'],
                  '#ffd700', '#ff8c00']
    classes = np.unique(y)
    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_pts[i % len(colors_pts)],
                   edgecolors='white', linewidths=0.5, s=40, label=f'Class {int(c)}', zorder=5)

    _apply_style(ax, title, 'Feature 1', 'Feature 2')
    ax.legend(facecolor=PALETTE['surface'], labelcolor=PALETTE['text'], framealpha=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(PALETTE['bg'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=[f'Pred {c}' for c in classes],
                yticklabels=[f'True {c}' for c in classes],
                ax=ax, linewidths=0.5, linecolor=PALETTE['bg'])
    ax.set_facecolor(PALETTE['surface'])
    ax.tick_params(colors=PALETTE['text'])
    ax.set_title(title, color=PALETTE['text'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', color=PALETTE['text'])
    ax.set_ylabel('True', color=PALETTE['text'])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_regression_results(y_true, y_pred, title='Regression Results'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(PALETTE['bg'])

    # Predicted vs Actual
    ax = axes[0]
    ax.scatter(y_true, y_pred, color=PALETTE['accent1'], alpha=0.6, s=30, edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'w--', linewidth=1.5, label='Perfect fit')
    _apply_style(ax, 'Predicted vs Actual', 'Actual', 'Predicted')
    ax.legend(facecolor=PALETTE['surface'], labelcolor=PALETTE['text'])

    # Residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, color=PALETTE['accent2'], alpha=0.6, s=30, edgecolors='none')
    ax.axhline(0, color='white', linewidth=1.5, linestyle='--')
    _apply_style(ax, 'Residuals', 'Predicted', 'Residual')

    plt.suptitle(title, color=PALETTE['text'], fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_experiment_comparison(results_dict, metric='accuracy', title='Model Comparison'):
    """Bar chart comparing multiple models."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['surface'])

    names = list(results_dict.keys())
    values = [results_dict[n][metric] for n in names]
    colors = [PALETTE['accent1'], PALETTE['accent2'], PALETTE['accent3'],
              '#ffd700', '#ff8c00'][:len(names)]

    bars = ax.bar(names, values, color=colors, edgecolor=PALETTE['bg'], linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', color=PALETTE['text'],
                fontsize=10, fontweight='bold')

    _apply_style(ax, title, 'Model', metric.capitalize())
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_loss_comparison(histories_dict, title='Loss Comparison'):
    """Overlay loss curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(PALETTE['bg'])
    colors = [PALETTE['accent1'], PALETTE['accent2'], PALETTE['accent3'], '#ffd700']

    for ax_idx, (metric, label) in enumerate([('loss', 'Loss'), ('accuracy', 'Accuracy')]):
        ax = axes[ax_idx]
        for i, (name, hist) in enumerate(histories_dict.items()):
            if metric in hist and hist[metric]:
                ax.plot(hist[metric], color=colors[i % len(colors)],
                        linewidth=2, label=name)
        _apply_style(ax, f'{label} Comparison', 'Epoch', label)
        ax.legend(facecolor=PALETTE['surface'], labelcolor=PALETTE['text'], framealpha=0.8)

    plt.suptitle(title, color=PALETTE['text'], fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()