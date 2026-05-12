# ETF Pairs Trading Research Pipeline

This project studies whether cointegration-based pairs trading relationships persist out of sample after transaction costs and walk-forward validation.

It is a research pipeline, not a claim of a profitable production trading strategy. The goal is to demonstrate market intuition, statistical reasoning, backtesting discipline, and clean Python implementation.

## MVP Scope

The implemented MVP focuses on:

- Single-pair diagnostics, with `TLT / IEF` as the recommended first pair.
- One-pair-per-window walk-forward validation.
- Transaction-cost-aware close-to-close backtesting.
- Core charts and metrics.
- Honest discussion of assumptions, lookahead controls, results, and failure cases.

Multi-pair portfolio backtesting, rolling hedge ratios, dashboards, databases, APIs, Docker, live trading, and broker integration are intentionally out of scope for the MVP.

## ETF Universe

The starter universe is a small set of liquid ETFs:

```text
SPY, IVV, VOO, QQQ, IWM,
TLT, IEF, GLD, SLV, XLE, USO,
XLK, XLF, XLV, XLI, XLY, XLP, XLU, XLB
```

`SPY / IVV` and `SPY / VOO` are useful sanity-check pairs because they are nearly identical ETFs. They should not be the main showcased result because the relationship is too trivial.

Recommended analysis pairs:

- `TLT / IEF`
- `GLD / SLV`
- `XLK / QQQ`
- `XLE / USO`
- `XLF / IWM`

`TLT / IEF` is the recommended first pair because both ETFs are economically linked through Treasury-rate exposure. It is relevant to fixed income, rates, and global markets, while being less trivial than `SPY / IVV`.

## Data Source

Data is downloaded from `yfinance` using:

```python
yf.download(tickers, start=start, end=end, auto_adjust=True)
```

Because `auto_adjust=True`, the returned `Close` column is adjusted for splits and dividends. The project treats this adjusted `Close` as the research price series.

## Methodology

1. Download adjusted ETF close prices.
2. Clean and align prices across the ETF universe.
3. Compute log returns for correlation screening.
4. Estimate the hedge ratio on log prices using OLS:

   ```text
   log(Y_t) = alpha + beta * log(X_t) + residual_t
   ```

5. Treat the residual as the spread.
6. Test pair relationships with Engle-Granger cointegration and an ADF test on the spread.
7. Build z-score signals from shifted rolling spread statistics.
8. Backtest with next-day execution and transaction costs.
9. Repeat pair selection and backtesting through rolling walk-forward windows.

## Lookahead-Bias Controls

- Pair selection uses training data only.
- Hedge ratios are estimated using training data only.
- Cointegration and ADF tests use training data only.
- Rolling z-scores use shifted rolling mean and standard deviation.
- Positions are shifted before returns are applied.
- Test-window performance is never used to choose pairs.

## Backtesting Assumptions

- Daily close-to-close returns.
- Long spread means long `Y` and short `beta * X`.
- Short spread means short `Y` and long `beta * X`.
- Gross exposure is normalized for the pair.
- Default commission is `0.5 bps`.
- Default slippage is `1.0 bps`.
- Costs are applied when positions change.

## Metrics

The pipeline reports cumulative return, annualized return, annualized volatility, Sharpe ratio, max drawdown, number of trades, hit rate, average holding period, turnover, gross return, net return, total transaction costs, and walk-forward performance by window.

## Failure-Case Analysis

Failure-case analysis is required. A useful failure case is a walk-forward window where the selected pair passes in-sample tests but loses money out of sample or stops mean-reverting. The report should discuss whether the issue appears to come from regime change, unstable hedge ratio, weak stationarity, slow mean reversion, or transaction costs overwhelming gross returns.

## How To Run

Install the package in editable mode:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Run tests:

```bash
.venv/bin/python -m pytest -p no:capture
```

Generate all MVP charts, CSV tables, and the final report:

```bash
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python -m pairs_trading.pipeline
```

Use the notebooks for exploration:

```text
notebooks/01_single_pair_diagnostics.ipynb
notebooks/02_walk_forward_results.ipynb
```

Reusable logic lives in `src/pairs_trading`.

## Interview Talking Points

- Correlation is not enough because two assets can move together without their price spread being mean-reverting.
- Cointegration means a linear combination of two non-stationary price series may be stationary.
- The spread should be stationary because the strategy assumes deviations from equilibrium revert.
- The hedge ratio is estimated with OLS on training-window log prices.
- Rolling z-score must avoid lookahead bias by using shifted rolling mean and standard deviation.
- Positions are shifted before applying returns because signals formed at today’s close cannot earn today’s close-to-close return.
- Walk-forward validation tests whether relationships persist out of sample.
- Transaction costs and slippage can turn attractive gross results into weak or negative net results.
- Failure cases reveal regime changes, unstable relationships, and overfitting risk.
- This is a research pipeline, not a production trading system.

## Stretch Goals

- Multi-pair portfolio backtest with equal-weight active pairs.
- Rolling hedge ratio.
- Individual stock universe.
- Sector-aware stock pair selection.
- Parameter sensitivity analysis.
- Dashboard.

## Out Of Scope

- Databases.
- APIs.
- Docker.
- Live trading.
- Intraday execution.
- Broker integration.
- Production risk system.
