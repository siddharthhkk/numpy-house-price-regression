"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import math
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    row = len(X)
    col = len(X[0])
    col_mean =[]
    for i in range(col):
        col_sum = 0.0
        cnt = 0.0
        for j in range(row):
            if not math.isnan(X[j][i]):
                col_sum += X[j][i]
                cnt += 1
        mean = (col_sum / cnt) if cnt > 0 else 0.0
        col_mean.append(mean)
    result = []
    for i in range(row):
        rows = []
        for j in range(col):
            if math.isnan(X[i][j]): rows.append(col_mean[j])
            else: rows.append(X[i][j])
        result.append(rows)
    return result
    pass

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3-q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (lower, upper)
    pass

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    arr = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            val = X[i][j]
            if val < lower[j]: row.append(lower[j])
            elif val > upper[j]: row.append(upper[j])
            else: row.append(val)
        arr.append(row)
    return arr
    pass

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator / (denominator + eps)
    pass

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    arr = []
    for i in range(len(X)):
        row = list(X[i]) + [col[i]]
        arr.append(row)
    return np.array(arr)
    
    pass

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    uniq, inv = np.unique(labels, return_inverse=True)
    result = np.zeros((len(labels), len(uniq)), dtype=float)
    result[np.arange(len(labels)), inv] = 1.0
    return result
    pass

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return mean, std
    pass

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    xr = X - mean
    xst = xr / std
    return xst
    pass

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    xo = np.ones((X.shape[0], 1))
    return np.hstack([xo, X])
    pass

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    x = np.arange(n_samples)
    s = np.random.default_rng(seed)
    s.shuffle(x)
    return x
    pass

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = np.array(indices[0:n_train])
    val_idx = np.array(indices[n_train: n_train + n_val])
    test_idx = np.array(indices[n_train + n_val :])
    return train_idx, val_idx, test_idx
    pass

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X_sub = np.array([X[indi] for indi in indices])
    y_sub = np.array([y[indi]for indi in indices])
    return X_sub, y_sub
    pass

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    a = X.T @ X
    b = X.T @ y
    theta = np.linalg.solve(a, b)
    return theta
    pass

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    ans = X @ theta
    return ans
    pass

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_true - y_pred))
    pass

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    rmse = (1 / len(y_true)) * np.sum(np.abs(y_pred - y_true)**2)
    return float(np.sqrt(rmse))
    pass

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_res = np.sum((y_true - y_pred)**2)
    SS_tot = np.sum((y_true - np.mean(y_true))**2)
    if SS_tot == 0: return 0
    return 1 - (SS_res / SS_tot)
    pass

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    mean = np.mean(y_true - y_pred)
    std = np.std(y_true - y_pred)
    abss = np.median(np.abs(y_true - y_pred))
    return {'mean': mean, 'std': std, 'median_abs': abss}
    pass

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X_clean = np.array(X, dtype=float)
    with np.errstate(all='ignore'):
        col_means = np.nanmean(X_clean, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_inds = np.isnan(X_clean)
    X_clean = np.where(nan_inds, col_means, X_clean)
    q25 = np.percentile(X_clean, 25, axis=0)
    q75 = np.percentile(X_clean, 75, axis=0)
    iqr = q75 - q25
    lower_bounds = q25 - iqr_k * iqr
    upper_bounds = q75 + iqr_k * iqr
    X_clean = np.clip(X_clean, lower_bounds, upper_bounds)
    return X_clean
    pass

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    num = X_num[:, ratio_num_idx]
    denom = X_num[:, ratio_den_idx]
    ratiof = make_ratio_feature(num, denom, eps=1e-08)
    res = append_column(X_num, ratiof)
    if cat_labels is not None:
        pp = one_hot_encode(cat_labels)
        for i in range(pp.shape[1]):
            res = append_column(res, pp[:, i])
    return res
    pass

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n_samples = X.shape[0]
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx, test_idx = partition_indices(indices, train_ratio, val_ratio)
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)
    return {
        'X_test': X_test, 'X_train': X_train, 'X_val': X_val,
        'y_test': y_test, 'y_train': y_train, 'y_val': y_val
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    X_train = splits['X_train']
    mean, std = fit_standardizer(X_train)
    std_splits = {}
    for key, matrix in splits.items():
        if key.startswith('X_'):
            X_scaled = apply_standardizer(matrix, mean, std)
            std_splits[key] = add_bias_column(X_scaled)
        else: std_splits[key] = matrix
            
    return std_splits, mean, std
    pass

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

