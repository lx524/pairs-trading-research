from pathlib import Path

import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end=None):
    """Download adjusted ETF OHLCV data from yfinance.

    yfinance is called with auto_adjust=True, so the returned Close column is
    adjusted for splits and dividends.
    """
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


def save_prices(prices, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=True)


def load_prices(path):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    return prices.sort_index()
