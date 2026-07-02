import matplotlib.pyplot as plt
import pandas as pd


class MarketData:
    """Wrapper around a price DataFrame with resampling support."""

    def __init__(
        self,
        df: pd.DataFrame,
        freq: str,
        symbol: str,
        dividends: dict[str, float] | None = None,
    ):
        """
        Args:
            df: DataFrame with DatetimeIndex and data columns.
            freq: Pandas frequency string (e.g. "1d", "30m", "5m").
        """
        self._df = df.copy()
        self.freq = freq
        self.symbol = symbol
        self.dividends = dividends if dividends is not None else {}

    @property
    def adj_factor(self) -> pd.DataFrame:
        """Compute adjustment factor from dividends.

        For each dividend date, finds the close price on that date and computes
        adj_factor = (1 - dividend / close). Result is forward-filled to cover
        all dates in the DataFrame.
        """
        div_df = pd.DataFrame(
            list(self.dividends.items()), columns=["date", "dividend"]
        )
        div_df["dividend"] = div_df["dividend"].astype(float)

        div_df["date"] = pd.to_datetime(div_df["date"])
        if self._df.index.tz is not None:
            div_df["date"] = div_df["date"].dt.tz_localize(self._df.index.tz)
        div_df = div_df.set_index("date").sort_index()

        closes = self._df["close"].reindex(div_df.index, method="ffill")
        div_df["close"] = closes

        div_df["adj_factor"] = 1 - div_df["dividend"] / div_df["close"]
        return div_df[["adj_factor"]]

    @property
    def adj_prices(self) -> pd.DataFrame:
        """Apply cumulative adjustment factor to open, high, low, close columns."""
        adj = (
            self.adj_factor.iloc[::-1]
            .cumprod()
            .iloc[::-1]
            .reindex(self._df.index, method="ffill")
            .fillna(1)
        )
        adj = adj["adj_factor"]  # Series with self._df.index
        print(self.adj_factor.iloc[::-1].cumprod().iloc[::-1])
        result = self._df[["open", "high", "low", "close"]].multiply(adj, axis=0)
        result["volume"] = self._df["volume"]
        result["count"] = self._df["count"]
        return result

    def plot(self):
        """Plot close and adjusted close prices."""
        fig, ax = plt.subplots()
        ax.plot(self._df.index, self._df["close"], label="close")
        ax.plot(self._df.index, self.adj_prices["close"], label="adjusted_close")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(f"{self.symbol} - Close vs Adjusted Close")
        ax.legend()
        plt.show()
