# ETF Pairs Trading Research Report

## Research Question

This project studies whether cointegration-based pairs trading relationships persist out of sample after transaction costs and walk-forward validation.

The work is framed as research, not as evidence of a production-ready profitable strategy.

## Data And Universe

The run used 19 ETFs from 2015-01-02 to 2026-05-12. Prices were downloaded with `yfinance` using `auto_adjust=True`, so the `Close` column is adjusted for splits and dividends.

The recommended first pair is `TLT / IEF` because both ETFs are linked to Treasury rates and fixed-income duration exposure. `SPY / IVV`, `SPY / VOO`, and `IVV / VOO` are useful sanity checks but are excluded from automated walk-forward selection so the report does not showcase near-duplicate S&P 500 ETF clones.

## Methodology

The pipeline estimates hedge ratios with OLS on log prices, constructs the spread as the OLS residual, and tests relationships using Engle-Granger cointegration plus an ADF test on the spread.

Signals are based on rolling z-scores of the spread. Rolling means and standard deviations are shifted so the current observation is not used to define its own signal.

## Backtest Design

The backtest uses close-to-close daily returns. Positions are shifted by one day before returns are applied, reflecting that a signal formed at today's close cannot earn today's close-to-close return.

Transaction costs include both commission and slippage assumptions and are applied whenever pair weights change.

## Single-Pair Results: TLT / IEF

Figures:

- Price chart: `reports/figures/single_pair_tlt_ief_prices.png`
- Spread chart: `reports/figures/single_pair_tlt_ief_spread.png`
- Z-score chart: `reports/figures/single_pair_tlt_ief_zscore.png`
- Equity curve: `reports/figures/single_pair_tlt_ief_equity.png`
- Drawdown chart: `reports/figures/single_pair_tlt_ief_drawdown.png`

Metrics:

| metric | value |
| --- | --- |
| cumulative_return | 5.83% |
| annualized_return | 0.50% |
| annualized_volatility | 1.58% |
| sharpe_ratio | 0.3167 |
| max_drawdown | -3.86% |
| number_of_trades | 74 |
| hit_rate | 62.16% |
| average_holding_period | 14.5676 |
| turnover | 0.1359 |
| gross_return | 8.22% |
| net_return | 5.83% |
| total_transaction_costs | 2.24% |

## Walk-Forward Results

Each window selects one pair using training data only. The selected pair is then traded in the following test window. Test performance is not used for pair selection, hedge-ratio estimation, or statistical testing.

Figures:

- Selected-pair table: `reports/figures/walk_forward_selected_pairs.png`
- Out-of-sample equity curve: `reports/figures/walk_forward_equity.png`
- Drawdown chart: `reports/figures/walk_forward_drawdown.png`
- Gross vs net comparison: `reports/figures/walk_forward_gross_vs_net.png`

Overall walk-forward metrics:

| metric | value |
| --- | --- |
| cumulative_return | -0.07% |
| annualized_return | -0.01% |
| annualized_volatility | 3.41% |
| sharpe_ratio | -0.0023 |
| max_drawdown | -10.89% |
| number_of_trades | 39 |
| hit_rate | n/a |
| average_holding_period | n/a |
| gross_return | 1.06% |
| net_return | -0.07% |
| total_transaction_costs | 1.12% |

Recent selected-pair windows:

| test_start | test_end | pair | correlation | coint_pvalue | adf_pvalue | half_life | gross_return | net_return | test_costs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020-01-06 | 2020-07-06 | SPY/XLK | 0.9370 | 0.0084 | 0.0017 | 13.8100 | 0.0125 | 0.0095 | 0.0030 |
| 2020-07-07 | 2021-01-04 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2021-01-05 | 2021-07-06 | XLK/XLV | 0.8343 | 0.0152 | 0.0033 | 16.0896 | 0.0076 | 0.0070 | 0.0006 |
| 2021-07-07 | 2022-01-03 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2022-01-04 | 2022-07-06 | XLK/XLY | 0.8799 | 0.0070 | 0.0014 | 12.5722 | 0.0272 | 0.0260 | 0.0012 |
| 2022-07-07 | 2023-01-04 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2023-01-05 | 2023-07-07 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2023-07-10 | 2024-01-05 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2024-01-08 | 2024-07-09 | IWM/XLF | 0.8274 | 0.0055 | 0.0011 | 12.0252 | 0.0111 | 0.0108 | 0.0003 |
| 2024-07-10 | 2025-01-07 | XLB/XLI | 0.8570 | 0.0120 | 0.0026 | 12.0845 | -0.0156 | -0.0173 | 0.0016 |
| 2025-01-08 | 2025-07-11 | IWM/XLF | 0.7542 | 0.0023 | 0.0004 | 10.2715 | -0.0475 | -0.0482 | 0.0007 |
| 2025-07-14 | 2026-01-09 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Performance by window is saved to `reports/walk_forward_window_metrics.csv`.

## Failure-Case Analysis

The weakest walk-forward window was `IWM/XLF` from 2025-01-08 to 2025-07-11. It returned -4.82% net of costs versus -4.75% gross, with 0.07% paid in modeled costs. This is a useful failure case because the pair was selected using in-sample correlation, cointegration, ADF, and half-life filters, but the out-of-sample period still did not necessarily reward the mean-reversion assumption. Possible explanations include a macro regime shift, unstable spread behavior, slow mean reversion, or transaction costs absorbing the gross edge.

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
