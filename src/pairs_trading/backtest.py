import pandas as pd
import numpy as np

from pairs_trading.metrics import performance_summary
from pairs_trading.preprocessing import simple_returns
from pairs_trading.signals import generate_positions, rolling_zscore
from pairs_trading.statistics import construct_spread


def _pair_weights(position, beta):
    gross = 1.0 + abs(beta)
    weights = pd.DataFrame(index=position.index)
    weights["y_weight"] = position / gross
    weights["x_weight"] = -position * beta / gross
    return weights


def _build_trades(position, returns):
    trades = []
    open_date = None
    open_side = 0
    pnl_parts = []

    for date, side in position.items():
        prev_side = open_side
        if prev_side == 0 and side != 0:
            open_date = date
            open_side = int(side)
            pnl_parts = []
        elif prev_side != 0 and side == 0 and open_date is not None:
            trades.append(
                {
                    "entry_date": open_date,
                    "exit_date": date,
                    "side": open_side,
                    "holding_period": len(pnl_parts),
                    "pnl": float(sum(pnl_parts)),
                }
            )
            open_date = None
            open_side = 0
            pnl_parts = []
        elif prev_side != 0 and side != prev_side and open_date is not None:
            trades.append(
                {
                    "entry_date": open_date,
                    "exit_date": date,
                    "side": open_side,
                    "holding_period": len(pnl_parts),
                    "pnl": float(sum(pnl_parts)),
                }
            )
            open_date = date
            open_side = int(side)
            pnl_parts = []

        if open_side != 0 and date in returns.index:
            pnl_parts.append(float(returns.loc[date]))

    return pd.DataFrame(trades)


def backtest_pair(
    prices,
    y,
    x,
    alpha,
    beta,
    signal_config,
    cost_bps,
    slippage_bps,
    signal_start=None,
):
    pair_prices = prices[[y, x]].dropna()
    spread = construct_spread(np.log(pair_prices[y]), np.log(pair_prices[x]), alpha, beta)
    zscore = rolling_zscore(spread, int(signal_config.get("lookback", 60)))
    tradable_zscore = zscore.loc[signal_start:] if signal_start is not None else zscore
    desired_position = generate_positions(
        tradable_zscore,
        entry_z=float(signal_config.get("entry_z", 2.0)),
        exit_z=float(signal_config.get("exit_z", 0.5)),
        stop_z=float(signal_config.get("stop_z", 3.5)),
        max_holding_days=signal_config.get("max_holding_days", 30),
    ).reindex(pair_prices.index, fill_value=0)

    returns = simple_returns(pair_prices).reindex(pair_prices.index).fillna(0.0)
    executed_position = desired_position.shift(1).fillna(0).astype(float)
    weights = _pair_weights(executed_position, beta)
    gross_returns = weights["y_weight"] * returns[y] + weights["x_weight"] * returns[x]

    traded_notional = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    total_cost_bps = float(cost_bps) + float(slippage_bps)
    costs = traded_notional * total_cost_bps / 10_000.0
    net_returns = gross_returns - costs
    equity = (1.0 + net_returns).cumprod()
    gross_equity = (1.0 + gross_returns).cumprod()

    positions = pd.DataFrame(
        {
            "desired_position": desired_position,
            "executed_position": executed_position,
            "y_weight": weights["y_weight"],
            "x_weight": weights["x_weight"],
        }
    )
    trades = _build_trades(executed_position, net_returns)
    metrics = performance_summary(net_returns, trades=trades, positions=weights)
    metrics.update(
        {
            "gross_return": float(gross_equity.iloc[-1] - 1.0) if not gross_equity.empty else float("nan"),
            "net_return": float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan"),
            "total_transaction_costs": float(costs.sum()),
        }
    )

    return {
        "pair": (y, x),
        "spread": spread,
        "zscore": zscore,
        "positions": positions,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "costs": costs,
        "equity": equity,
        "gross_equity": gross_equity,
        "trades": trades,
        "metrics": metrics,
    }


def make_walk_forward_windows(prices, train_days, test_days):
    windows = []
    dates = prices.index
    start = 0
    while start + train_days + test_days <= len(dates):
        train_start = dates[start]
        train_end = dates[start + train_days - 1]
        test_start = dates[start + train_days]
        test_end = dates[start + train_days + test_days - 1]
        windows.append((train_start, train_end, test_start, test_end))
        start += test_days
    return windows


def walk_forward_one_pair(prices, config):
    from pairs_trading import pair_selection

    train_days = int(config.get("train_days", config.get("walk_forward", {}).get("train_days", 504)))
    test_days = int(config.get("test_days", config.get("walk_forward", {}).get("test_days", 126)))
    signal_config = config.get("signals", {})
    costs = config.get("costs", {})
    cost_bps = float(costs.get("commission_bps", config.get("cost_bps", 0.5)))
    slippage_bps = float(costs.get("slippage_bps", config.get("slippage_bps", 1.0)))

    all_returns = []
    all_gross_returns = []
    all_costs = []
    selected_rows = []
    pair_results = []

    for train_start, train_end, test_start, test_end in make_walk_forward_windows(prices, train_days, test_days):
        train_prices = prices.loc[train_start:train_end]
        test_prices = prices.loc[test_start:test_end]
        selected = pair_selection.select_best_pair(train_prices, config.get("pair_selection", config))

        if not selected:
            empty = pd.Series(0.0, index=test_prices.index)
            all_returns.append(empty)
            all_gross_returns.append(empty)
            all_costs.append(empty)
            selected_rows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "pair": None,
                    "reason": "no_pair_selected",
                }
            )
            continue

        y = selected["y"]
        x = selected["x"]
        history_and_test = prices.loc[train_start:test_end, [y, x]]
        result = backtest_pair(
            history_and_test,
            y,
            x,
            selected["alpha"],
            selected["beta"],
            signal_config,
            cost_bps,
            slippage_bps,
            signal_start=train_end,
        )
        test_returns = result["net_returns"].loc[test_start:test_end]
        test_gross_returns = result["gross_returns"].loc[test_start:test_end]
        test_costs = result["costs"].loc[test_start:test_end]
        test_position = result["positions"]["executed_position"].loc[test_start:test_end]
        test_entries = ((test_position != 0) & (test_position.shift(1).fillna(0) == 0)).sum()
        all_returns.append(test_returns)
        all_gross_returns.append(test_gross_returns)
        all_costs.append(test_costs)
        pair_results.append(result)
        selected_rows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "pair": f"{y}/{x}",
                "y": y,
                "x": x,
                "correlation": selected.get("correlation"),
                "coint_pvalue": selected.get("coint_pvalue"),
                "adf_pvalue": selected.get("adf_pvalue"),
                "half_life": selected.get("half_life"),
                "alpha": selected.get("alpha"),
                "beta": selected.get("beta"),
                "gross_return": float((1.0 + test_gross_returns).prod() - 1.0),
                "net_return": float((1.0 + test_returns).prod() - 1.0),
                "test_costs": float(test_costs.sum()),
                "number_of_trades": int(test_entries),
            }
        )

    net_returns = pd.concat(all_returns).sort_index() if all_returns else pd.Series(dtype=float)
    gross_returns = pd.concat(all_gross_returns).sort_index() if all_gross_returns else pd.Series(dtype=float)
    costs_series = pd.concat(all_costs).sort_index() if all_costs else pd.Series(dtype=float)
    equity = (1.0 + net_returns).cumprod()
    gross_equity = (1.0 + gross_returns).cumprod()
    selected_pairs = pd.DataFrame(selected_rows)
    metrics = performance_summary(net_returns)
    metrics["gross_return"] = float(gross_equity.iloc[-1] - 1.0) if not gross_equity.empty else float("nan")
    metrics["net_return"] = float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan")
    metrics["total_transaction_costs"] = float(costs_series.sum()) if not costs_series.empty else 0.0
    metrics["number_of_trades"] = int(selected_pairs.get("number_of_trades", pd.Series(dtype=float)).sum())

    return {
        "net_returns": net_returns,
        "gross_returns": gross_returns,
        "equity": equity,
        "gross_equity": gross_equity,
        "costs": costs_series,
        "selected_pairs": selected_pairs,
        "pair_results": pair_results,
        "metrics": metrics,
    }
