from __future__ import annotations

from pathlib import Path

import pandas as pd


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted ETF OHLCV data from yfinance.

    yfinance is called with auto_adjust=True, so the returned Close column is
    adjusted for splits and dividends.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to download market data.") from exc

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("No data returned from yfinance.")
    return raw


def save_prices(prices: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=True)


def load_prices(path: str | Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    return prices.sort_index()
