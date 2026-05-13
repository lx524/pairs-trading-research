import math

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def _aligned_pair(y, x):
    data = pd.concat([y, x], axis=1).dropna()
    if data.shape[0] < 3:
        raise ValueError("At least three aligned observations are required.")
    return data.iloc[:, 0], data.iloc[:, 1]


def estimate_hedge_ratio(y, x):
    """Estimate alpha and beta in y = alpha + beta * x + residual."""
    y_aligned, x_aligned = _aligned_pair(y, x)
    design = np.column_stack([np.ones(len(x_aligned)), x_aligned.to_numpy()])
    alpha, beta = np.linalg.lstsq(design, y_aligned.to_numpy(), rcond=None)[0]
    return float(alpha), float(beta)


def construct_spread(y, x, alpha, beta):
    y_aligned, x_aligned = _aligned_pair(y, x)
    spread = y_aligned - alpha - beta * x_aligned
    spread.name = f"{y.name or 'y'}_{x.name or 'x'}_spread"
    return spread


def engle_granger_pvalue(y, x):
    y_aligned, x_aligned = _aligned_pair(y, x)
    _stat, pvalue, _crit = coint(y_aligned, x_aligned)
    return float(pvalue)


def adf_pvalue(series):
    clean = series.dropna()
    if clean.shape[0] < 10:
        return float("nan")
    _stat, pvalue, *_rest = adfuller(clean, autolag="AIC")
    return float(pvalue)


def half_life(spread):
    clean = spread.dropna()
    lagged = clean.shift(1).dropna()
    delta = clean.diff().dropna()
    lagged, delta = lagged.align(delta, join="inner")
    if len(lagged) < 3:
        return float("nan")

    design = np.column_stack([np.ones(len(lagged)), lagged.to_numpy()])
    _intercept, slope = np.linalg.lstsq(design, delta.to_numpy(), rcond=None)[0]
    if slope >= 0:
        return float("inf")
    return float(-math.log(2) / slope)
