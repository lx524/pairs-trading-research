import numpy as np
import pandas as pd

from pairs_trading.statistics import estimate_hedge_ratio


def test_estimate_hedge_ratio_recovers_synthetic_relationship():
    x = pd.Series(np.linspace(1.0, 10.0, 100), name="x")
    y = 0.75 + 1.6 * x

    alpha, beta = estimate_hedge_ratio(y, x)

    assert abs(alpha - 0.75) < 1e-10
    assert abs(beta - 1.6) < 1e-10
