from itertools import combinations

import numpy as np
import pandas as pd

from pairs_trading.preprocessing import log_returns
from pairs_trading.statistics import (
    adf_pvalue,
    construct_spread,
    engle_granger_pvalue,
    estimate_hedge_ratio,
    half_life,
)


def candidate_pairs(tickers):
    return list(combinations(tickers, 2))


def correlation_screen(
    returns,
    pairs,
    min_corr,
    excluded_pairs=None,
):
    rows = []
    corr = returns.corr()
    excluded = {tuple(sorted(pair)) for pair in (excluded_pairs or [])}
    for y, x in pairs:
        if tuple(sorted((y, x))) in excluded:
            continue
        if y not in corr.index or x not in corr.columns:
            continue
        value = float(corr.loc[y, x])
        if np.isfinite(value) and value >= min_corr:
            rows.append({"y": y, "x": x, "correlation": value})
    columns = ["y", "x", "correlation"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("correlation", ascending=False).reset_index(drop=True)


def evaluate_pair(log_price_data, y, x):
    pair = log_price_data[[y, x]].dropna()
    alpha, beta = estimate_hedge_ratio(pair[y], pair[x])
    spread = construct_spread(pair[y], pair[x], alpha, beta)
    coint_pvalue = engle_granger_pvalue(pair[y], pair[x])
    spread_adf_pvalue = adf_pvalue(spread)
    spread_half_life = half_life(spread)
    return {
        "y": y,
        "x": x,
        "alpha": alpha,
        "beta": beta,
        "coint_pvalue": coint_pvalue,
        "adf_pvalue": spread_adf_pvalue,
        "half_life": spread_half_life,
    }


def select_best_pair(train_prices, config):
    min_corr = float(config.get("min_corr", 0.75))
    max_coint_pvalue = float(config.get("max_coint_pvalue", 0.05))
    max_adf_pvalue = float(config.get("max_adf_pvalue", 0.05))
    max_half_life = float(config.get("max_half_life", 60))
    excluded_pairs = [tuple(pair) for pair in config.get("excluded_pairs", [])]

    tickers = list(train_prices.columns)
    returns = log_returns(train_prices)
    screened = correlation_screen(returns, candidate_pairs(tickers), min_corr, excluded_pairs=excluded_pairs)
    if screened.empty:
        return {}

    log_price_data = np.log(train_prices)
    rows = []
    for row in screened.itertuples(index=False):
        evaluated = evaluate_pair(log_price_data, row.y, row.x)
        evaluated["correlation"] = float(row.correlation)
        rows.append(evaluated)

    if not rows:
        return {}

    table = pd.DataFrame(rows)
    table = table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["coint_pvalue", "adf_pvalue", "half_life", "beta"]
    )
    table = table[
        (table["coint_pvalue"] <= max_coint_pvalue)
        & (table["adf_pvalue"] <= max_adf_pvalue)
        & (table["half_life"] > 0)
        & (table["half_life"] <= max_half_life)
    ]
    if table.empty:
        return {}

    table = table.assign(
        score=table["coint_pvalue"] + table["adf_pvalue"] + table["half_life"] / 252.0 - table["correlation"] * 0.05
    )
    best = table.sort_values("score").iloc[0].to_dict()
    return best
