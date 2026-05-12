import pandas as pd

from pairs_trading import backtest


def test_backtest_applies_next_day_returns(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"Y": [100.0, 110.0, 121.0], "X": [100.0, 100.0, 100.0]}, index=dates)

    monkeypatch.setattr(backtest, "rolling_zscore", lambda spread, lookback: pd.Series([0.0, 0.0, 0.0], index=dates))
    monkeypatch.setattr(backtest, "generate_positions", lambda *args, **kwargs: pd.Series([1, 0, 0], index=dates))

    result = backtest.backtest_pair(prices, "Y", "X", 0.0, 0.0, {"lookback": 2}, 0.0, 0.0)

    assert result["net_returns"].iloc[0] == 0.0
    assert abs(result["net_returns"].iloc[1] - 0.10) < 1e-12
    assert result["net_returns"].iloc[2] == 0.0


def test_transaction_costs_are_applied_when_positions_change(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"Y": [100.0, 100.0, 100.0, 100.0, 100.0], "X": [100.0] * 5}, index=dates)

    monkeypatch.setattr(backtest, "rolling_zscore", lambda spread, lookback: pd.Series([0.0] * 5, index=dates))
    monkeypatch.setattr(backtest, "generate_positions", lambda *args, **kwargs: pd.Series([0, 1, 1, 0, 0], index=dates))

    result = backtest.backtest_pair(prices, "Y", "X", 0.0, 0.0, {"lookback": 2}, 100.0, 0.0)

    assert result["costs"].iloc[2] == 0.01
    assert result["costs"].iloc[4] == 0.01
    assert abs(result["metrics"]["total_transaction_costs"] - 0.02) < 1e-12
