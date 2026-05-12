from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def performance_summary(
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    positions: pd.DataFrame | pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict:
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    cumulative = float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan")
    years = len(returns) / periods_per_year if len(returns) else float("nan")
    ann_return = float((1.0 + cumulative) ** (1.0 / years) - 1.0) if years and years > 0 else float("nan")
    ann_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year)) if len(returns) else float("nan")
    sharpe = float(ann_return / ann_vol) if ann_vol and ann_vol > 0 else float("nan")

    summary = {
        "cumulative_return": cumulative,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(equity) if not equity.empty else float("nan"),
    }

    if trades is not None and not trades.empty:
        summary.update(
            {
                "number_of_trades": int(len(trades)),
                "hit_rate": float((trades["pnl"] > 0).mean()) if "pnl" in trades else float("nan"),
                "average_holding_period": float(trades["holding_period"].mean())
                if "holding_period" in trades
                else float("nan"),
            }
        )
    else:
        summary.update({"number_of_trades": 0, "hit_rate": float("nan"), "average_holding_period": float("nan")})

    if positions is not None:
        pos = positions if isinstance(positions, pd.DataFrame) else positions.to_frame("position")
        turnover = pos.diff().abs().sum(axis=1).sum()
        gross_exposure_days = pos.abs().sum(axis=1).sum()
        summary["turnover"] = float(turnover / gross_exposure_days) if gross_exposure_days > 0 else 0.0

    return summary
