from functools import cached_property

import numpy as np
import pandas as pd


class VolatilityEstimator:
    """Estimates volatility from adjusted price data."""

    def __init__(
        self,
        px: pd.DataFrame,
        symbol: str,
        n_days: int = 30,
        px_freq: str | None = None,
    ):
        """
        Args:
            px: DataFrame with adjusted_prices from MarketData.adj_prices.
                Expected columns: open, high, low, close, volume, count
            symbol: Instrument identifier.
            n_days: Number of calendar days for the rolling volatility window.
            px_freq: Index frequency string (e.g. "30m", "1D"). Used to select
                the correct annualization factor.
        """

        self.symbol = symbol
        self.px = px
        self.n_days = n_days
        self.window = f"{n_days}D"
        self.px_freq = px_freq

    @cached_property
    def freq_adj_factor(self) -> int:
        """Number of observations per trading day for the given px_freq.

        Used to adjust the rolling window height when computing the overlapping
        adjustment factor. E.g. 13 for 30-minute bars (6.5 hours / 30min),
        1 for daily bars.
        """
        return {"30m": 13, "1D": 1}[self.px_freq]

    @cached_property
    def overlapping_adj_factor(self) -> float:
        """Adjustment factor that unbiases variance under overlapping rolling windows.

        When data is sampled more frequently than the rolling window step size,
        consecutive window estimates share observations, introducing autocorrelation
        that biases the variance downward. This factor corrects for that bias per
        the formula in Politis & White (2004).

        Returns 1.0 for non-overlapping data (e.g. 1D bars on a daily frequency).
        """
        h = self.freq_adj_factor * self.n_days
        n = len(self.px)
        return 1 / (1 - (h / n) + ((h**2 - 1) / (3 * n**2)))

    def _min_periods_mask(self, result: pd.Series) -> pd.Series:
        """Mask values that fall within the minimum periods lookback window.

        The first ``n_days`` calendar days are masked because the rolling window
        does not have enough historical data to produce a full-period estimate.
        This prevents stale/undersized volatility readings from being treated as
        valid estimates.

        Returns a boolean mask where True indicates the value should be excluded.
        """
        start_date = result.index.date[0]
        end_date = start_date + pd.Timedelta(days=self.n_days)

        date_rng = pd.date_range(start=start_date, end=end_date, freq="D")

        result_dates = pd.to_datetime(result.index.date)
        return result[~result_dates.isin(date_rng)]

    def estimate(self) -> pd.Series:
        """Placeholder — implemented by subclasses."""
        raise NotImplementedError

    @cached_property
    def adjusted_estimate(self) -> pd.Series:
        """Return annualized volatility by calling estimate() and scaling.

        Annualizes the result and applies the overlapping data adjustment factor
        to unbias the variance.
        """
        result = self.estimate
        ann_factor = np.sqrt(self.freq_adj_factor * 252)
        olap_factor = np.sqrt(self.overlapping_adj_factor)
        print(
            f"annualization_factor={ann_factor:.3f}, overlapping_adj_factor={olap_factor:.3f}"
        )
        return result * ann_factor * olap_factor


class CloseCloseEstimator(VolatilityEstimator):
    """Close-to-close realized volatility estimator.

    Only the last row of each day is kept so the first day captures the overnight
    "open jump" component. If all rows were kept, subsequent days' rolling volatility
    would include intraday returns while the first day would miss its overnight jump —
    creating a systematic bias where day-1 volatility is understated relative to the rest.
    """

    def __init__(
        self,
        px: pd.DataFrame,
        symbol: str,
        n_days: int = 30,
        px_freq: str | None = None,
    ):
        first_date = px.index.date[0]
        last_row_of_first = px[px.index.date == first_date].iloc[-1:]
        rest = px[px.index.date > first_date]
        self.px = pd.concat([last_row_of_first, rest])
        super().__init__(self.px, symbol, n_days, px_freq)

    @property
    def returns(self) -> pd.Series:
        """Compute log returns from adjusted close prices."""
        return np.log(self.px["close"] / self.px["close"].shift(1)).dropna()

    @cached_property
    def estimate(self) -> pd.Series:
        """Compute rolling close-to-close realized volatility."""
        result = self.returns.rolling(self.window, min_periods=1).std()
        return self._min_periods_mask(result)


class GarmanKlassEstimator(VolatilityEstimator):
    """Garman-Klass realized volatility estimator using open-high-low-close prices.

    Captures the full intraday price range and is more efficient than close-to-close,
    providing better volatility estimates while remaining unbiased under a Brownian motion
    assumption.
    """

    @cached_property
    def estimate(self) -> pd.Series:
        """Compute rolling Garman-Klass realized volatility."""
        hl = np.log(self.px["high"] / self.px["low"]) ** 2
        ccl = np.log(self.px["close"] / self.px["close"].shift(1)) ** 2
        result = 0.5 * hl - (2 * np.log(2) - 1) * ccl
        result = np.sqrt(result.rolling(self.window, min_periods=1).mean())
        return self._min_periods_mask(result)


class RogerSatchellEstimator(VolatilityEstimator):
    """Roger-Satchell realized volatility estimator using open-high-low-close prices.

    Captures drift term separately and is unbiased for log-normal prices.
    """

    @cached_property
    def estimate(self) -> pd.Series:
        """Compute rolling Roger-Satchell realized volatility."""
        hc = np.log(self.px["high"] / self.px["close"])
        ho = np.log(self.px["high"] / self.px["open"])
        lc = np.log(self.px["low"] / self.px["close"])
        lo = np.log(self.px["low"] / self.px["open"])
        rs = hc * ho + lc * lo
        result = np.sqrt(rs.rolling(self.window, min_periods=1).mean())
        return self._min_periods_mask(result)


class YangZhangEstimator(VolatilityEstimator):
    """Yang-Zhang realized volatility estimator combining overnight and intraday components.

    Combines a open-to-close estimator, an overnight estimator, and a Rogers-Satchell
    component. The overnight and open-to-close terms are weighted by an optimal constant
    that minimizes bias under a jump-diffusion process.
    """

    @cached_property
    def estimate(self) -> pd.Series:
        """Compute rolling Yang-Zhang realized volatility.

        Deviates from the snippet in the book as the book's formula systematically biases high.
        """
        ocl = np.log(self.px["open"] / self.px["close"].shift(1))
        co = np.log(self.px["close"] / self.px["open"])

        hc = np.log(self.px["high"] / self.px["close"])
        ho = np.log(self.px["high"] / self.px["open"])
        lc = np.log(self.px["low"] / self.px["close"])
        lo = np.log(self.px["low"] / self.px["open"])
        rs = (hc * ho) + (lc * lo)

        n = ocl.rolling(self.window, min_periods=1).count()
        mean_ocl = ocl.rolling(self.window, min_periods=1).mean()
        mean_co = co.rolling(self.window, min_periods=1).mean()
        sum_rs = rs.rolling(self.window, min_periods=1).sum()

        k = 0.34 / (1.34 + ((n + 1) / (n - 1)))

        sigma2_o = (ocl - mean_ocl).rolling(self.window, min_periods=1).var()
        sigma2_c = (co - mean_co).rolling(self.window, min_periods=1).var()

        sigma2 = (
            (sigma2_o / (n - 1))
            + (k * sigma2_c / (n - 1))
            + ((1 - k) * sum_rs / (n - 1))
        )
        result = np.sqrt(sigma2)
        return self._min_periods_mask(result)


class ParkinsonEstimator(VolatilityEstimator):
    """Parkinson realized volatility estimator using high-low range."""

    @cached_property
    def estimate(self) -> pd.Series:
        """Compute rolling Parkinson realized volatility."""
        hl = np.log(self.px["high"] / self.px["low"]) ** 2
        f = lambda w: np.sqrt(w.mean() / (4 * np.log(2)))
        result = hl.rolling(self.window, min_periods=1).apply(f)
        return self._min_periods_mask(result)
