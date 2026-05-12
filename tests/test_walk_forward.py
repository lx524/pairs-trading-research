import pandas as pd

from pairs_trading.backtest import walk_forward_one_pair
from pairs_trading import pair_selection


def test_walk_forward_selection_does_not_use_test_data(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=8)
    prices = pd.DataFrame({"Y": range(8), "X": range(10, 18)}, index=dates, dtype=float)
    seen_windows = []

    def fake_select_best_pair(train_prices, config):
        seen_windows.append((train_prices.index.min(), train_prices.index.max(), len(train_prices)))
        return {}

    monkeypatch.setattr(pair_selection, "select_best_pair", fake_select_best_pair)

    walk_forward_one_pair(prices, {"train_days": 4, "test_days": 2})

    assert seen_windows == [(dates[0], dates[3], 4), (dates[2], dates[5], 4)]
