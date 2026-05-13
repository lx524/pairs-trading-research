from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_path(path):
    if path is None:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _finish(fig, path):
    out = _prepare_path(path)
    if out is not None:
        fig.savefig(out, bbox_inches="tight", dpi=150)
    return fig


def plot_pair_prices(prices, y, x, path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    normalized = prices[[y, x]].dropna() / prices[[y, x]].dropna().iloc[0]
    normalized.plot(ax=ax)
    ax.set_title(f"Normalized Adjusted Close: {y} / {x}")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    return _finish(fig, path)


def plot_spread(spread, path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    spread.plot(ax=ax, label="Spread")
    spread.rolling(60).mean().plot(ax=ax, label="60D mean")
    ax.set_title("Cointegration Spread")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, path)


def plot_zscore(zscore, path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    zscore.plot(ax=ax, label="Z-score")
    for level in (-2.0, -0.5, 0.5, 2.0):
        ax.axhline(level, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Spread Z-score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, path)


def plot_equity_curve(equity, path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    equity.plot(ax=ax)
    ax.set_title("Net Equity Curve")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    return _finish(fig, path)


def plot_gross_vs_net(gross_equity, net_equity, path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    gross_equity.plot(ax=ax, label="Gross")
    net_equity.plot(ax=ax, label="Net after costs")
    ax.set_title("Gross vs Net Equity")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, path)


def plot_drawdown(equity, path=None):
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    fig, ax = plt.subplots(figsize=(10, 4))
    drawdown.plot(ax=ax)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    return _finish(fig, path)


def plot_selected_pairs_table(selected_pairs, path=None):
    display_cols = [
        col
        for col in ["train_start", "train_end", "test_start", "test_end", "pair", "coint_pvalue", "adf_pvalue", "test_return"]
        if col in selected_pairs.columns
    ]
    table = selected_pairs[display_cols].copy()
    for col in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[col]):
            table[col] = table[col].dt.date.astype(str)
        elif pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].round(4)
    fig, ax = plt.subplots(figsize=(12, max(2, 0.35 * len(table) + 1)))
    ax.axis("off")
    ax.table(cellText=table.astype(str).values, colLabels=table.columns, loc="center")
    ax.set_title("Walk-Forward Selected Pairs")
    return _finish(fig, path)
