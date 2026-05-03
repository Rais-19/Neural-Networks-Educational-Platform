import numpy as np
from sklearn.datasets import (make_moons, make_circles, make_blobs,
                              make_classification, load_iris)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


def get_binary_dataset(name='moons', n_samples=400, noise=0.15, random_state=42):
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif name == 'xor':
        rng = np.random.RandomState(random_state)
        X = rng.randn(n_samples, 2)
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
        X += rng.randn(*X.shape) * noise
    elif name == 'linear':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   random_state=random_state)
    else:
        raise ValueError(f"Unknown binary dataset: {name}")
    return X, y


def get_multiclass_dataset(name='blobs', n_classes=3, n_samples=400, random_state=42):
    if name == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=n_classes,
                          n_features=2, random_state=random_state, cluster_std=1.2)
    elif name == 'iris':
        data = load_iris()
        X, y = data.data[:, :2], data.target   # first 2 features for 2D viz
        n_classes = 3
    else:
        raise ValueError(f"Unknown multiclass dataset: {name}")
    return X, y, n_classes


def get_regression_dataset(name='sine', n_samples=300, noise=0.15, random_state=42):
    rng = np.random.RandomState(random_state)
    if name == 'sine':
        X = rng.uniform(-np.pi, np.pi, (n_samples, 1))
        y = np.sin(X).flatten() + rng.randn(n_samples) * noise
    elif name == 'quadratic':
        X = rng.uniform(-3, 3, (n_samples, 1))
        y = (0.5 * X ** 2 - X + 1).flatten() + rng.randn(n_samples) * noise
    elif name == 'linear':
        X = rng.uniform(-3, 3, (n_samples, 1))
        y = (2 * X + 1).flatten() + rng.randn(n_samples) * noise
    else:
        raise ValueError(f"Unknown regression dataset: {name}")
    return X, y


def prepare_data(X, y, test_size=0.25, problem_type='binary', n_classes=None, random_state=42):
    """Scale features and split. Returns train/test + one-hot y for multiclass."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state)

    y_train_raw = y_train.copy()
    y_test_raw = y_test.copy()

    if problem_type == 'multiclass':
        n_cls = n_classes or len(np.unique(y))
        enc = OneHotEncoder(sparse_output=False, categories='auto')
        y_train = enc.fit_transform(y_train.reshape(-1, 1))
        y_test = enc.transform(y_test.reshape(-1, 1))
    elif problem_type == 'binary':
        y_train = y_train.astype(float).reshape(-1, 1)
        y_test = y_test.astype(float).reshape(-1, 1)
    else:  # regression
        y_train = y_train.astype(float).reshape(-1, 1)
        y_test = y_test.astype(float).reshape(-1, 1)

    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, scaler


def load_csv_dataset(file, target_column, problem_type='binary', test_size=0.25):
    df = pd.read_csv(file)
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found.")
    X = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).values
    y = df[target_column].values
    n_classes = len(np.unique(y)) if problem_type == 'multiclass' else None
    return prepare_data(X, y, test_size=test_size,
                        problem_type=problem_type, n_classes=n_classes)