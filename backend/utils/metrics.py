import numpy as np


#  Classification

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t]][class_to_idx[p]] += 1
    return cm, classes

def precision_recall_f1(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

# Regression 

def mse_metric(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse_metric(y_true, y_pred):
    return np.sqrt(mse_metric(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-9)

def compute_classification_metrics(y_true, y_pred):
    acc = accuracy(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    cm, classes = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'confusion_matrix': cm, 'classes': classes}

def compute_regression_metrics(y_true, y_pred):
    return {
        'mse': mse_metric(y_true, y_pred),
        'rmse': rmse_metric(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }