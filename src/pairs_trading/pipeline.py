from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pairs_trading.backtest import backtest_pair, walk_forward_one_pair
from pairs_trading.data import download_prices, save_prices
from pairs_trading.preprocessing import align_prices, extract_close
from pairs_trading.statistics import estimate_hedge_ratio
from pairs_trading.visualization import (
    plot_drawdown,
    plot_equity_curve,
    plot_gross_vs_net,
    plot_pair_prices,
    plot_selected_pairs_table,
    plot_spread,
    plot_zscore,
)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_single_pair(prices, y: str = "TLT", x: str = "IEF", signal_config: dict | None = None, costs: dict | None = None) -> dict:
    signal_config = signal_config or {}
    costs = costs or {}
    log_pair = np.log(prices[[y, x]])
    alpha, beta = estimate_hedge_ratio(log_pair[y], log_pair[x])
    return backtest_pair(
        prices,
        y,
        x,
        alpha,
        beta,
        signal_config,
        cost_bps=float(costs.get("commission_bps", 0.5)),
        slippage_bps=float(costs.get("slippage_bps", 1.0)),
    )


def run_full_pipeline(
    universe_path: str | Path = "config/universe.yaml",
    backtest_path: str | Path = "config/backtest.yaml",
) -> dict:
    universe = load_yaml(universe_path)
    config = load_yaml(backtest_path)
    run_config = copy.deepcopy(config)
    sanity_pairs = universe.get("sanity_check_pairs", [])
    clone_pairs = sanity_pairs + [["IVV", "VOO"]]
    run_config.setdefault("pair_selection", {})
    run_config["pair_selection"].setdefault("excluded_pairs", clone_pairs)
    raw = download_prices(universe["tickers"], start=config["start"], end=config.get("end"))
    prices = align_prices(extract_close(raw))
    save_prices(prices, "data/processed/adjusted_close.csv")
    single_pair = run_single_pair(
        prices,
        *universe.get("recommended_first_pair", ["TLT", "IEF"]),
        signal_config=run_config.get("signals", {}),
        costs=run_config.get("costs", {}),
    )
    walk_forward = walk_forward_one_pair(prices, run_config)
    return {"prices": prices, "single_pair": single_pair, "walk_forward": walk_forward}


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def _format_float(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    values = frame.map(lambda value: "n/a" if pd.isna(value) else str(value))
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(str(item) for item in row) + " |" for row in values.to_numpy()]
    return "\n".join([header, separator, *rows])


def _metrics_frame(metrics: dict) -> pd.DataFrame:
    order = [
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "hit_rate",
        "average_holding_period",
        "turnover",
        "gross_return",
        "net_return",
        "total_transaction_costs",
    ]
    rows = []
    for metric in order:
        if metric not in metrics:
            continue
        value = metrics[metric]
        if metric in {
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
            "hit_rate",
            "gross_return",
            "net_return",
            "total_transaction_costs",
        }:
            formatted = _format_pct(float(value))
        elif metric in {"number_of_trades"}:
            formatted = str(int(value))
        else:
            formatted = _format_float(float(value))
        rows.append({"metric": metric, "value": formatted})
    return pd.DataFrame(rows)


def _window_metrics(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    if selected_pairs.empty:
        return selected_pairs
    cols = [
        "test_start",
        "test_end",
        "pair",
        "gross_return",
        "net_return",
        "test_costs",
        "number_of_trades",
        "correlation",
        "coint_pvalue",
        "adf_pvalue",
        "half_life",
    ]
    available = [col for col in cols if col in selected_pairs.columns]
    return selected_pairs[available].copy()


def _write_report(
    prices: pd.DataFrame,
    single_pair: dict,
    walk_forward: dict,
    single_metrics: pd.DataFrame,
    wf_metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
    figure_paths: dict[str, str],
    report_path: str | Path,
) -> None:
    selected = walk_forward["selected_pairs"].copy()
    selected_display = selected.copy()
    for col in ["train_start", "train_end", "test_start", "test_end"]:
        if col in selected_display:
            selected_display[col] = pd.to_datetime(selected_display[col]).dt.date.astype(str)
    display_cols = [
        col
        for col in [
            "test_start",
            "test_end",
            "pair",
            "correlation",
            "coint_pvalue",
            "adf_pvalue",
            "half_life",
            "gross_return",
            "net_return",
            "test_costs",
        ]
        if col in selected_display
    ]
    selected_display = selected_display[display_cols].tail(12)

    for frame in [selected_display, window_metrics]:
        for col in frame.select_dtypes(include=["float", "float64"]).columns:
            frame[col] = frame[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "n/a")

    failure_text = "No negative walk-forward window was available in this run, so the main residual risk is that passing in-sample filters may still fail in future regimes."
    if not selected.empty and "net_return" in selected:
        candidates = selected.dropna(subset=["net_return"]).sort_values("net_return")
        if not candidates.empty:
            worst = candidates.iloc[0]
            pair = worst.get("pair", "n/a")
            test_start = pd.to_datetime(worst["test_start"]).date()
            test_end = pd.to_datetime(worst["test_end"]).date()
            failure_text = (
                f"The weakest walk-forward window was `{pair}` from {test_start} to {test_end}. "
                f"It returned {_format_pct(float(worst['net_return']))} net of costs versus "
                f"{_format_pct(float(worst.get('gross_return', np.nan)))} gross, with "
                f"{_format_pct(float(worst.get('test_costs', np.nan)))} paid in modeled costs. "
                "This is a useful failure case because the pair was selected using in-sample "
                "correlation, cointegration, ADF, and half-life filters, but the out-of-sample "
                "period still did not necessarily reward the mean-reversion assumption. Possible "
                "explanations include a macro regime shift, unstable spread behavior, slow "
                "mean reversion, or transaction costs absorbing the gross edge."
            )

    text = f"""# ETF Pairs Trading Research Report

## Research Question

This project studies whether cointegration-based pairs trading relationships persist out of sample after transaction costs and walk-forward validation.

The work is framed as research, not as evidence of a production-ready profitable strategy.

## Data And Universe

The run used {len(prices.columns)} ETFs from {prices.index.min().date()} to {prices.index.max().date()}. Prices were downloaded with `yfinance` using `auto_adjust=True`, so the `Close` column is adjusted for splits and dividends.

The recommended first pair is `TLT / IEF` because both ETFs are linked to Treasury rates and fixed-income duration exposure. `SPY / IVV`, `SPY / VOO`, and `IVV / VOO` are useful sanity checks but are excluded from automated walk-forward selection so the report does not showcase near-duplicate S&P 500 ETF clones.

## Methodology

The pipeline estimates hedge ratios with OLS on log prices, constructs the spread as the OLS residual, and tests relationships using Engle-Granger cointegration plus an ADF test on the spread.

Signals are based on rolling z-scores of the spread. Rolling means and standard deviations are shifted so the current observation is not used to define its own signal.

## Backtest Design

The backtest uses close-to-close daily returns. Positions are shifted by one day before returns are applied, reflecting that a signal formed at today's close cannot earn today's close-to-close return.

Transaction costs include both commission and slippage assumptions and are applied whenever pair weights change.

## Single-Pair Results: TLT / IEF

Figures:

- Price chart: `{figure_paths['single_prices']}`
- Spread chart: `{figure_paths['single_spread']}`
- Z-score chart: `{figure_paths['single_zscore']}`
- Equity curve: `{figure_paths['single_equity']}`
- Drawdown chart: `{figure_paths['single_drawdown']}`

Metrics:

{_markdown_table(single_metrics)}

## Walk-Forward Results

Each window selects one pair using training data only. The selected pair is then traded in the following test window. Test performance is not used for pair selection, hedge-ratio estimation, or statistical testing.

Figures:

- Selected-pair table: `{figure_paths['wf_selected_pairs']}`
- Out-of-sample equity curve: `{figure_paths['wf_equity']}`
- Drawdown chart: `{figure_paths['wf_drawdown']}`
- Gross vs net comparison: `{figure_paths['wf_gross_vs_net']}`

Overall walk-forward metrics:

{_markdown_table(wf_metrics)}

Recent selected-pair windows:

{_markdown_table(selected_display)}

Performance by window is saved to `reports/walk_forward_window_metrics.csv`.

## Failure-Case Analysis

{failure_text}

Failure cases are not an embarrassment in this project; they are the point of the research process. They show whether an apparently reasonable in-sample statistical relationship survives a realistic out-of-sample test after costs.

## Limitations

- Daily ETF data does not model intraday execution.
- yfinance data is convenient but not institutional-grade.
- The ETF universe is small and may introduce selection bias.
- Transaction costs are simplified.
- This is not a production trading system.

## Reproducibility

Run the full MVP pipeline with:

```bash
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python -m pairs_trading.pipeline
```

Run the test suite with:

```bash
.venv/bin/python -m pytest -p no:capture
```
"""
    Path(report_path).write_text(text, encoding="utf-8")


def generate_mvp_outputs(
    universe_path: str | Path = "config/universe.yaml",
    backtest_path: str | Path = "config/backtest.yaml",
    reports_dir: str | Path = "reports",
) -> dict:
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    results = run_full_pipeline(universe_path, backtest_path)
    prices = results["prices"]
    single_pair = results["single_pair"]
    walk_forward = results["walk_forward"]
    y, x = single_pair["pair"]

    figure_paths = {
        "single_prices": str(figures_dir / "single_pair_tlt_ief_prices.png"),
        "single_spread": str(figures_dir / "single_pair_tlt_ief_spread.png"),
        "single_zscore": str(figures_dir / "single_pair_tlt_ief_zscore.png"),
        "single_equity": str(figures_dir / "single_pair_tlt_ief_equity.png"),
        "single_drawdown": str(figures_dir / "single_pair_tlt_ief_drawdown.png"),
        "wf_selected_pairs": str(figures_dir / "walk_forward_selected_pairs.png"),
        "wf_equity": str(figures_dir / "walk_forward_equity.png"),
        "wf_drawdown": str(figures_dir / "walk_forward_drawdown.png"),
        "wf_gross_vs_net": str(figures_dir / "walk_forward_gross_vs_net.png"),
    }

    plot_pair_prices(prices, y, x, figure_paths["single_prices"])
    plot_spread(single_pair["spread"], figure_paths["single_spread"])
    plot_zscore(single_pair["zscore"], figure_paths["single_zscore"])
    plot_equity_curve(single_pair["equity"], figure_paths["single_equity"])
    plot_drawdown(single_pair["equity"], figure_paths["single_drawdown"])
    plot_selected_pairs_table(walk_forward["selected_pairs"].tail(12), figure_paths["wf_selected_pairs"])
    plot_equity_curve(walk_forward["equity"], figure_paths["wf_equity"])
    plot_drawdown(walk_forward["equity"], figure_paths["wf_drawdown"])
    plot_gross_vs_net(walk_forward["gross_equity"], walk_forward["equity"], figure_paths["wf_gross_vs_net"])

    single_metrics = _metrics_frame(single_pair["metrics"])
    wf_metrics = _metrics_frame(walk_forward["metrics"])
    window_metrics = _window_metrics(walk_forward["selected_pairs"])

    single_metrics.to_csv(reports_dir / "single_pair_metrics.csv", index=False)
    wf_metrics.to_csv(reports_dir / "walk_forward_metrics.csv", index=False)
    window_metrics.to_csv(reports_dir / "walk_forward_window_metrics.csv", index=False)
    walk_forward["selected_pairs"].to_csv(reports_dir / "walk_forward_selected_pairs.csv", index=False)

    _write_report(
        prices,
        single_pair,
        walk_forward,
        single_metrics,
        wf_metrics,
        window_metrics,
        figure_paths,
        reports_dir / "final_report.md",
    )

    return {
        **results,
        "figure_paths": figure_paths,
        "single_metrics": single_metrics,
        "walk_forward_metrics": wf_metrics,
        "window_metrics": window_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ETF pairs trading MVP pipeline.")
    parser.add_argument("--universe", default="config/universe.yaml")
    parser.add_argument("--backtest", default="config/backtest.yaml")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    outputs = generate_mvp_outputs(args.universe, args.backtest, args.reports_dir)
    print("MVP pipeline completed.")
    print(f"Single-pair metrics: {Path(args.reports_dir) / 'single_pair_metrics.csv'}")
    print(f"Walk-forward metrics: {Path(args.reports_dir) / 'walk_forward_metrics.csv'}")
    print(f"Final report: {Path(args.reports_dir) / 'final_report.md'}")
    print("Figures:")
    for path in outputs["figure_paths"].values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
