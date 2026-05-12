from __future__ import annotations

import numpy as np
import pandas as pd


def extract_close(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract adjusted Close prices from yfinance output.

    With auto_adjust=True, yfinance's Close field is adjusted for splits and
    dividends. This function supports both single-ticker and multi-ticker
    yfinance column layouts.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        first_level = set(raw.columns.get_level_values(0))
        last_level = set(raw.columns.get_level_values(-1))
        if "Close" in first_level:
            close = raw["Close"]
        elif "Close" in last_level:
            close = raw.xs("Close", axis=1, level=-1)
        else:
            raise KeyError("Could not find Close column in MultiIndex data.")
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
    else:
        close = raw.copy()

    close = close.copy()
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close.columns = [str(col) for col in close.columns]
    return close


def align_prices(prices: pd.DataFrame, min_obs: int = 500) -> pd.DataFrame:
    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.where(clean > 0)
    clean = clean.dropna(axis=1, thresh=min_obs)
    clean = clean.ffill(limit=2)
    clean = clean.dropna(how="any")
    return clean


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all")


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return log_prices(prices).diff().dropna(how="all")
