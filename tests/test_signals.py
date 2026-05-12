import numpy as np
import pandas as pd

from pairs_trading.signals import generate_positions, rolling_zscore


def test_rolling_zscore_uses_shifted_rolling_statistics():
    spread = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2024-01-01", periods=4))

    zscore = rolling_zscore(spread, lookback=2)

    expected = (3.0 - 1.5) / np.std([1.0, 2.0], ddof=1)
    assert np.isnan(zscore.iloc[0])
    assert np.isnan(zscore.iloc[1])
    assert abs(zscore.iloc[2] - expected) < 1e-12


def test_generate_positions_enters_exits_and_stops():
    zscore = pd.Series(
        [np.nan, -2.1, -1.0, -0.4, 2.2, 3.6],
        index=pd.date_range("2024-01-01", periods=6),
    )

    positions = generate_positions(zscore, entry_z=2.0, exit_z=0.5, stop_z=3.5)

    assert positions.tolist() == [0, 1, 1, 0, -1, 0]
