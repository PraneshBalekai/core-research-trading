# Research - Trading Repository

A python mono-repo with implementations that explore research topics.
Integrates with IBKR, Binance APIs to collect data.

The goal is to build this into a platform that can run semi-systematic trades;

## Repo Structure

```
IB_PRD/
├── cio/          # Core IO package, factory functions and specific data loader / writer module from / to different sources
├── core/         # Core strategy logic and models - most of the core implementations live here
├── etl/          # Light-weight wrappers around data_loaders / data_writers to collect historical data (and real-time delayed)
├── external/     # External integrations (brokers, data providers)
└── trading/      # TODO: Trading execution and order management
```

Each sub-package (`cio`, `core`, `etl`, `external`, `trading`) is an independent Python package with its own `pyproject.toml`.

## Notebooks
- Findings and exploratory notebooks are under [core/core/notebooks](core/core/notebooks)
