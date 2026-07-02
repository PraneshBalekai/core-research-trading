# Notebooks

Notebooks for exploring ideas from books and research papers.

---

---

## [Timing Mean Reversion Using IV](iv_mean_reversion.ipynb)

Backtests supporting ATM implied volatility ([reference](https://www.crackingmarkets.com/iv-mean-reversion/)) from IBKR can act as a timing signal for price mean-reversion.


## [Volatility Range Momentum](rolling_vol_range_momentum.ipynb)

Reproducting a published volatility range breakout strategy ([paper link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172)). The idea: use rolling intraday volatility to construct dynamic upper/lower bounds around the VWAP — when price breaks above or below these bands, enter long or short. All positions are closed by end of day, eliminating overnight risk.

---

## [Volatility Estimators](volatility_estimators.ipynb)

Alternative volatility estimators from *Volatility Trading* by Euan Sinclair, applied to 30-minute AAPL bars with dividend adjustments.
