import numpy as np
from sklearn.datasets import (make_moons, make_circles, make_blobs,
                              make_classification, load_iris)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


# ── Built-in generators ───────────────────────────────────────────────────────

def get_binary_dataset(name='moons', n_samples=400, noise=0.15, random_state=42):
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5,
                            random_state=random_state)
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
        X, y = data.data[:, :2], data.target
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


# ── Core prepare function ─────────────────────────────────────────────────────

def prepare_data(X, y, test_size=0.25, problem_type='binary',
                 n_classes=None, random_state=42):
    """
    Scale X and split into train/test.
    Returns: X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, scaler
    - y_train / y_test  : model-ready (float, one-hot for multiclass)
    - y_train_raw / y_test_raw : integer labels for metrics
    """
    # Guarantee X is a clean float64 2D array
    X = np.array(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Guarantee y is a 1D numpy array of integers (label-encoded if needed)
    y = _encode_target(y, problem_type)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state)

    y_train_raw = y_train.copy()
    y_test_raw  = y_test.copy()

    if problem_type == 'multiclass':
        enc     = OneHotEncoder(sparse_output=False, categories='auto')
        y_train = enc.fit_transform(y_train.reshape(-1, 1))
        y_test  = enc.transform(y_test.reshape(-1, 1))
    else:
        # binary and regression both become float column vectors
        y_train = y_train.astype(float).reshape(-1, 1)
        y_test  = y_test.astype(float).reshape(-1, 1)

    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, scaler


def _encode_target(y, problem_type):
    """
    Convert any target array to a clean 1D integer (or float for regression) array.
    Handles: ints, floats, strings, booleans, pandas Series, mixed types.
    """
    # Convert pandas Series / anything iterable to numpy
    if hasattr(y, 'values'):
        y = y.values
    y = np.asarray(y).flatten()

    if problem_type == 'regression':
        try:
            return y.astype(np.float64)
        except (ValueError, TypeError):
            raise ValueError(
                "Regression target column contains non-numeric values. "
                "Please choose a numeric target column."
            )

    # Classification: encode to integers 0, 1, 2, ...
    # If already integer-like, just cast
    try:
        y_int = y.astype(np.int64)
        # Check the cast didn't silently corrupt floats like 0.5 → 0
        if np.allclose(y.astype(float), y_int.astype(float)):
            # Re-map to 0-based consecutive labels in case classes are e.g. [2,5,7]
            le = LabelEncoder()
            return le.fit_transform(y_int).astype(np.int64)
    except (ValueError, TypeError):
        pass

    # String / boolean / mixed → LabelEncoder
    le = LabelEncoder()
    return le.fit_transform(y.astype(str)).astype(np.int64)


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_csv_dataset(file, target_column, problem_type='binary', test_size=0.25):
    """
    Load any CSV file and prepare it for training.
    Handles: string targets, non-consecutive integer labels, missing values,
             non-numeric feature columns, file pointer already consumed.
    """
    # Always reset the file pointer — sidebar may have already read the file
    if hasattr(file, 'seek'):
        file.seek(0)

    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")

    if df.empty:
        raise ValueError("CSV file is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # ── Features ──────────────────────────────────────────────────────────────
    feature_df = df.drop(columns=[target_column])

    # Drop columns that are entirely non-numeric (e.g. ID, name strings)
    numeric_df = feature_df.select_dtypes(include=[np.number])
    dropped    = [c for c in feature_df.columns if c not in numeric_df.columns]
    if dropped:
        # Not an error — just silently drop non-numeric columns
        pass

    if numeric_df.shape[1] == 0:
        raise ValueError(
            "No numeric feature columns found after removing the target. "
            f"Non-numeric columns dropped: {dropped or 'none'}. "
            "Please ensure your CSV has at least one numeric feature column."
        )

    # Drop rows where ANY feature or target is NaN
    target_series = df[target_column]
    combined      = numeric_df.copy()
    combined["__target__"] = target_series.values
    combined = combined.dropna()

    if combined.empty:
        raise ValueError("No rows remaining after dropping rows with missing values.")

    X = combined.drop(columns=["__target__"]).values   # shape (n, p) — guaranteed 2D
    y = combined["__target__"].values                   # shape (n,)

    if X.shape[0] < 5:
        raise ValueError(
            f"Only {X.shape[0]} valid rows found — need at least 5 to train."
        )

    # ── Validate class count vs problem type ──────────────────────────────────
    n_unique = len(np.unique(y.astype(str)))
    if problem_type == 'binary' and n_unique != 2:
        raise ValueError(
            f"Binary Classification expects exactly 2 unique target values, "
            f"but found {n_unique}: {np.unique(y.astype(str))[:5].tolist()}. "
            f"Switch to Multi-class Classification or choose a different target column."
        )
    if problem_type == 'multiclass' and n_unique < 2:
        raise ValueError(
            f"Multi-class Classification expects ≥2 classes, found {n_unique}."
        )

    n_classes = n_unique if problem_type == 'multiclass' else None

    return prepare_data(X, y, test_size=test_size,
                        problem_type=problem_type, n_classes=n_classes)