"""Bootstrap utilities for uncertainty estimation."""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def bootstrap_linear_regression(X, y, n_bootstrap=1000, random_seed=42):
    """Perform bootstrap analysis for linear regression.

    Args:
        X: Feature matrix (n_samples, n_features) or 1D array
        y: Target values (n_samples,)
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility

    Returns:
        dict with bootstrap results:
            - slopes: array of bootstrap slope estimates
            - intercepts: array of bootstrap intercept estimates
            - slope_mean: mean of slopes
            - slope_std: std of slopes
            - slope_ci: [2.5%, 97.5%] confidence interval
            - intercept_mean: mean of intercepts
            - intercept_ci: [2.5%, 97.5%] confidence interval
            - predictions: bootstrap predictions on original X (n_bootstrap, n_samples)
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    np.random.seed(random_seed)
    n_samples = len(y)

    bootstrap_slopes = []
    bootstrap_intercepts = []
    bootstrap_predictions = []

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Check if we have enough unique X values for regression
        if X.shape[1] == 1 and len(np.unique(X_boot[:, 0])) < 2:
            continue  # Skip this iteration

        # Fit model on bootstrap sample
        model_boot = LinearRegression()
        model_boot.fit(X_boot, y_boot)

        bootstrap_slopes.append(model_boot.coef_)
        bootstrap_intercepts.append(model_boot.intercept_)

        # Store predictions on original data
        y_pred_boot = model_boot.predict(X)
        bootstrap_predictions.append(y_pred_boot)

    bootstrap_slopes = np.array(bootstrap_slopes)
    bootstrap_intercepts = np.array(bootstrap_intercepts)
    bootstrap_predictions = np.array(bootstrap_predictions)

    # Calculate confidence intervals
    slope_ci = np.percentile(bootstrap_slopes, [2.5, 97.5], axis=0)
    intercept_ci = np.percentile(bootstrap_intercepts, [2.5, 97.5])

    # For predictions
    pred_mean = bootstrap_predictions.mean(axis=0)
    pred_std = bootstrap_predictions.std(axis=0)
    pred_ci_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
    pred_ci_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)

    return {
        'slopes': bootstrap_slopes,
        'intercepts': bootstrap_intercepts,
        'slope_mean': bootstrap_slopes.mean(axis=0),
        'slope_std': bootstrap_slopes.std(axis=0),
        'slope_ci': slope_ci,
        'intercept_mean': bootstrap_intercepts.mean(),
        'intercept_ci': intercept_ci,
        'predictions': bootstrap_predictions,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_ci_lower': pred_ci_lower,
        'pred_ci_upper': pred_ci_upper
    }


def bootstrap_scipy_regression(X, y, n_bootstrap=1000, random_seed=42):
    """Perform bootstrap analysis using scipy's linregress.

    Args:
        X: 1D array of feature values
        y: 1D array of target values
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility

    Returns:
        dict with bootstrap results (similar to bootstrap_linear_regression)
    """
    np.random.seed(random_seed)
    n_samples = len(y)

    bootstrap_slopes = []
    bootstrap_intercepts = []

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Need at least 2 unique x values
        if len(np.unique(X_boot)) > 1:
            slope, intercept, _, _, _ = stats.linregress(X_boot, y_boot)
            bootstrap_slopes.append(slope)
            bootstrap_intercepts.append(intercept)

    bootstrap_slopes = np.array(bootstrap_slopes)
    bootstrap_intercepts = np.array(bootstrap_intercepts)

    return {
        'slopes': bootstrap_slopes,
        'intercepts': bootstrap_intercepts,
        'slope_mean': bootstrap_slopes.mean(),
        'slope_std': bootstrap_slopes.std(),
        'slope_ci': np.percentile(bootstrap_slopes, [2.5, 97.5]),
        'intercept_mean': bootstrap_intercepts.mean(),
        'intercept_ci': np.percentile(bootstrap_intercepts, [2.5, 97.5])
    }


def compute_tradeoff_from_coefficients(coef_1, coef_2, bootstrap_coefs=None):
    """Compute tradeoff between two coefficients.

    For a linear model: y = beta_1 * x_1 + beta_2 * x_2 + intercept
    The tradeoff is: how much does x_1 need to change to compensate for a unit change in x_2?
    Tradeoff = -beta_2 / beta_1

    Args:
        coef_1: Coefficient for variable 1
        coef_2: Coefficient for variable 2
        bootstrap_coefs: Optional (n_bootstrap, 2) array of bootstrap coefficients

    Returns:
        dict with:
            - tradeoff: point estimate
            - tradeoff_mean: bootstrap mean (if bootstrap_coefs provided)
            - tradeoff_median: bootstrap median (if bootstrap_coefs provided)
            - tradeoff_ci: [2.5%, 97.5%] CI (if bootstrap_coefs provided)
    """
    tradeoff = coef_2 / coef_1

    result = {'tradeoff': tradeoff}

    if bootstrap_coefs is not None:
        bootstrap_tradeoffs = bootstrap_coefs[:, 1] / bootstrap_coefs[:, 0]
        result.update({
            'tradeoff_mean': bootstrap_tradeoffs.mean(),
            'tradeoff_median': np.median(bootstrap_tradeoffs),
            'tradeoff_ci': np.percentile(bootstrap_tradeoffs, [2.5, 97.5]),
            'bootstrap_tradeoffs': bootstrap_tradeoffs
        })

    return result
