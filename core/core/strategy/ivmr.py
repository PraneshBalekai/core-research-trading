from __future__ import annotations

import numpy as np
import pandas as pd


def combine_close_prices(
    stock_df: pd.DataFrame,
    iv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract close prices from stock and implied volatility dataframes.

    Parameters
    ----------
    stock_df : pd.DataFrame
        Stock OHLC dataframe with datetime index and 'close' column.
    iv_df : pd.DataFrame
        Implied volatility OHLC dataframe with datetime index and 'close' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with datetime index and two columns:
        'close' (stock close price) and 'iv_close' (implied volatility close).
    """
    start_date = max(stock_df.index.min(), iv_df.index.min())
    stock_filtered = stock_df.loc[start_date:]
    iv_filtered = iv_df.loc[start_date:]
    result = pd.concat(
        [
            stock_filtered["close"],
            iv_filtered["close"],
            stock_filtered["open"].shift(-1),
        ],
        axis=1,
    )
    result.columns = ["close", "iv_close", "next_open"]
    return result


def calculate_sma_bands(
    df: pd.DataFrame,
    sma_long: int = 200,
    sma_short: int = 20,
    base_column: str = "sma_short",
    k: float = 1,
) -> pd.DataFrame:
    """
    Calculate SMA bands with IV-adjusted bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index, 'close' column, and 'iv_close' column.
    sma_long : int
        Rolling window for long-term SMA. Default is 200.
    sma_short : int
        Rolling window for short-term SMA. Default is 20.
    base_column : str
        The column to use as the base for calculating bands. Default is "sma_short".
        Options: "sma_short", "sma_long".
    k : float
        IV Band width. Default is 1.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional columns: sma_long, sma_short, iv_daily_close,
        upper_band, and lower_band.
    """
    result = df.copy()
    result["sma_long"] = result["close"].rolling(window=sma_long).mean()
    result["sma_short"] = result["close"].rolling(window=sma_short).mean()
    result["iv_daily_close"] = result["iv_close"] / (252**0.5)
    base = result[base_column]
    result["upper_band"] = base + (base * k * result["iv_daily_close"])
    result["lower_band"] = base - (base * k * result["iv_daily_close"])
    return result


def calculate_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate entry signals based on close price relative to previous close and lower band.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns: close, lower_band, sma_long.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional 'entry' column (1 when entry signal, NaN otherwise).
    """
    result = df.copy()
    result["entry"] = (
        (
            (result["close"] < result["close"].shift(1))
            & (result["close"] < result["lower_band"])
            & (result["close"] > result["sma_long"])
            & (result["next_open"] < result["lower_band"])
            & (result["next_open"] > result["sma_long"])
        )
        .astype(int)
        .replace(0, np.nan)
    )
    return result


def load_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate exit signals and position based on close relative to SMA 20.

    We are in a position if we've seen an entry signal and haven't exited yet.
    We exit when either condition is met while in a position:
    - exit: close >= sma_short * 0.995 (regular exit)
    - stop_loss: sma_short < entry_price OR close < sma_long (stop loss triggered)
    - stop_loss: position held for >= 10 days (max holding period)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns: close, sma_short, entry.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional columns:
        - 'exit': 1 when we exit a position (close >= sma_short * 0.995), NaN otherwise
        - 'stop_loss': 1 when stop loss triggered
            (sma_short < entry_price OR close < sma_long OR holding >= 10 days), NaN otherwise
        - 'position': 1 when in a position, 0 otherwise
    """
    result = df.copy()
    result["exit"] = np.nan
    result["stop_loss"] = np.nan
    result["position"] = 0

    in_position = False
    entry_price = None
    # entry_date = None
    stop_loss_cooldown = 0  # days since stop loss exit

    for i in range(len(result)):
        if stop_loss_cooldown > 0:
            stop_loss_cooldown -= 1

        if result["entry"].iloc[i] == 1 and not in_position and stop_loss_cooldown == 0:
            # Entry signal - we enter a position (only if not already in one and not in cooldown)
            in_position = True
            entry_price = result["next_open"].iloc[i]
            # entry_date = result.index[i]
            result.iloc[i, result.columns.get_loc("position")] = 1
        elif result["entry"].iloc[i] == 1 and in_position:
            # Already in position, null out this redundant entry signal
            result.iloc[i, result.columns.get_loc("entry")] = np.nan
            result.iloc[i, result.columns.get_loc("position")] = 1
        elif (
            result["entry"].iloc[i] == 1 and not in_position and stop_loss_cooldown > 0
        ):
            # In stop loss cooldown period, null out this entry signal
            result.iloc[i, result.columns.get_loc("entry")] = np.nan
        elif (
            in_position
            and result["close"].iloc[i] >= result["sma_short"].iloc[i] * 0.995
        ):
            # In position and close >= sma_short - regular exit
            result.iloc[i, result.columns.get_loc("exit")] = 1
            result.iloc[i, result.columns.get_loc("position")] = 0
            in_position = False
            entry_price = None
            # entry_date = None
        elif (
            in_position
            and entry_price is not None
            and result["sma_short"].iloc[i] < entry_price
        ):
            # In position and sma_short dropped below entry price - stop loss triggered
            result.iloc[i, result.columns.get_loc("stop_loss")] = 1
            result.iloc[i, result.columns.get_loc("position")] = 0
            in_position = False
            entry_price = None
            # entry_date = None
            stop_loss_cooldown = 5  # 5 day cooldown before re-entering
        elif in_position and result["close"].iloc[i] < result["sma_long"].iloc[i]:
            # In position and close dropped below sma_long - stop loss triggered
            result.iloc[i, result.columns.get_loc("stop_loss")] = 1
            result.iloc[i, result.columns.get_loc("position")] = 0
            in_position = False
            entry_price = None
            # entry_date = None
            stop_loss_cooldown = 5  # 5 day cooldown before re-entering
        # elif in_position and entry_date is not None and (result.index[i] - entry_date).days >= 8:
        #     # In position and held for >= 10 days - max holding period stop loss
        #     result.iloc[i, result.columns.get_loc("stop_loss")] = 1
        #     result.iloc[i, result.columns.get_loc("position")] = 0
        #     in_position = False
        #     entry_price = None
        #     entry_date = None
        #     stop_loss_cooldown = 5  # 5 day cooldown before re-entering
        elif in_position:
            # Still in position
            result.iloc[i, result.columns.get_loc("position")] = 1
        # else: position stays 0, exit/stop_loss stay NaN

    return result
