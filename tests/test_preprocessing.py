import numpy as np
import pandas as pd

from pairs_trading.preprocessing import simple_returns


def test_simple_returns_are_close_to_close_percentage_changes():
    prices = pd.DataFrame({"TLT": [100.0, 105.0, 102.9]}, index=pd.date_range("2024-01-01", periods=3))

    returns = simple_returns(prices)

    np.testing.assert_allclose(returns["TLT"].to_numpy(), [0.05, -0.02])
