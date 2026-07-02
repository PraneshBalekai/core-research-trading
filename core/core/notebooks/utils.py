# flake8: noqa
# TODO: Generalize and move to core/visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_close_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str = "Series 1",
    label2: str = "Series 2",
) -> plt.Figure:
    """
    Plot the 'close' fields from two dataframes as line graphs.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe with datetime index and 'close' column.
    df2 : pd.DataFrame
        Second dataframe with datetime index and 'close' column.
    label1 : str, optional
        Label for the first series in the legend. Default is "Series 1".
    label2 : str, optional
        Label for the second series in the legend. Default is "Series 2".

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df1.index, df1["close"], label=label1)
    ax.plot(df2.index, df2["close"], label=label2)
    if "next_open" in df1.columns:
        ax.plot(
            df1.index,
            df1["next_open"],
            label=f"{label1} Next Open",
            color="blue",
            alpha=0.5,
            linewidth=0.5,
            linestyle="dotted",
        )
    if "next_open" in df2.columns:
        ax.plot(
            df2.index,
            df2["next_open"],
            label=f"{label2} Next Open",
            color="orange",
            alpha=0.5,
            linewidth=0.5,
            linestyle="dotted",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.set_title("Close Price Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


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
    df: pd.DataFrame, base_column: str = "sma20", k: float = 1
) -> pd.DataFrame:
    """
    Calculate SMA bands with IV-adjusted bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index, 'close' column, and 'iv_close' column.
    base_column : str
        The column to use as the base for calculating bands. Default is "sma20".
        Options: "sma20", "sma200".
    k : float
        IV Band width. Default is 1.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional columns: sma200, sma20, iv_daily_close,
        upper_band, and lower_band.
    """
    result = df.copy()
    result["sma200"] = result["close"].rolling(window=200).mean()
    result["sma20"] = result["close"].rolling(window=20).mean()
    result["iv_daily_close"] = result["iv_close"] / (252**0.5)
    base = result[base_column]
    result["upper_band"] = base + (base * k * result["iv_daily_close"])
    result["lower_band"] = base - (base * k * result["iv_daily_close"])
    return result


def plot_sma_bands(df: pd.DataFrame) -> plt.Figure:
    """
    Plot SMA bands with close price.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns: sma200, sma20, close,
        upper_band, lower_band.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["sma200"], label="SMA 200", color="blue")
    ax.plot(df.index, df["sma20"], label="SMA 20", color="orange")
    ax.plot(df.index, df["close"], label="Close", color="black", linewidth=1)
    ax.plot(df.index, df["upper_band"], label="Upper Band", color="red", linestyle="--")
    ax.plot(
        df.index, df["lower_band"], label="Lower Band", color="green", linestyle="--"
    )
    if "next_open" in df.columns:
        ax.plot(
            df.index,
            df["next_open"],
            label="Next Open",
            color="gray",
            alpha=0.5,
            linewidth=0.5,
            linestyle="dotted",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("SMA Bands with IV-Adjusted Bounds")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if "iv_daily_close" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            df.index,
            df["iv_daily_close"],
            label="IV Daily Close",
            color="red",
            alpha=0.25,
        )
        ax2.set_ylabel("IV Daily Close")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig


def plot_sma_bands_interactive(df: pd.DataFrame) -> go.Figure:
    """
    Plot SMA bands with close price using Plotly for interactive viewing.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns: sma200, sma20, close,
        upper_band, lower_band.

    Returns
    -------
    go.Figure
        The Plotly figure object.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["sma200"], name="SMA 200", line=dict(color="blue"))
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["sma20"], name="SMA 20", line=dict(color="orange"))
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["close"], name="Close", line=dict(color="black", width=1)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["upper_band"],
            name="Upper Band",
            line=dict(color="red", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["lower_band"],
            name="Lower Band",
            line=dict(color="green", dash="dot"),
        )
    )
    if "next_open" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["next_open"],
                name="Next Open",
                line=dict(color="gray", width=1, dash="dot"),
                opacity=0.5,
            )
        )

    fig.update_layout(
        title="SMA Bands with IV-Adjusted Bounds",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )
    return fig


def calculate_entry_signals(
    df: pd.DataFrame, iv_daily_close_threshold: float = 0.010
) -> pd.DataFrame:
    """
    Calculate entry signals based on close price relative to previous close and lower band.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns: close, lower_band, sma200.
    iv_daily_close_threshold : float, optional
        Minimum IV daily close value required for entry. Default is 0.010.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional 'entry' column set to 1 where:
        - close is less than the previous close
        - close is less than the lower band
        - close is above sma200
        Otherwise 'entry' is set to 0.
    """
    result = df.copy()
    result["entry"] = (
        (
            (result["close"] < result["close"].shift(1))
            & (result["close"] < result["lower_band"])
            & (result["close"] > result["sma200"])
            & (result["next_open"] < result["lower_band"])
            & (result["next_open"] > result["sma200"])
            & (result["iv_daily_close"] > iv_daily_close_threshold)
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
    - exit: close >= sma20 * 0.995 (regular exit)
    - stop_loss: sma20 < entry_price OR close < sma200 (stop loss triggered)
    - stop_loss: position held for >= 10 days (max holding period)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns: close, sma20, entry.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional columns:
        - 'exit': 1 when we exit a position (close >= sma20 * 0.995), NaN otherwise
        - 'stop_loss': 1 when stop loss triggered (sma20 < entry_price OR close < sma200 OR holding >= 10 days), NaN otherwise
        - 'position': 1 when in a position, 0 otherwise
    """
    result = df.copy()
    result["exit"] = np.nan
    result["stop_loss"] = np.nan
    result["position"] = 0

    in_position = False
    entry_price = None
    entry_date = None
    stop_loss_cooldown = 0  # days since stop loss exit

    for i in range(len(result)):
        if stop_loss_cooldown > 0:
            stop_loss_cooldown -= 1

        if result["entry"].iloc[i] == 1 and not in_position and stop_loss_cooldown == 0:
            # Entry signal - we enter a position (only if not already in one and not in cooldown)
            in_position = True
            entry_price = result["next_open"].iloc[i]
            entry_date = result.index[i]
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
        elif in_position and result["close"].iloc[i] >= result["sma20"].iloc[i] * 0.995:
            # In position and close >= sma20 - regular exit
            result.iloc[i, result.columns.get_loc("exit")] = 1
            result.iloc[i, result.columns.get_loc("position")] = 0
            in_position = False
            entry_price = None
            entry_date = None
        elif (
            in_position
            and entry_price is not None
            and result["sma20"].iloc[i] < entry_price
        ):
            # In position and sma20 dropped below entry price - stop loss triggered
            result.iloc[i, result.columns.get_loc("stop_loss")] = 1
            result.iloc[i, result.columns.get_loc("position")] = 0
            in_position = False
            entry_price = None
            entry_date = None
            stop_loss_cooldown = 5  # 5 day cooldown before re-entering
        elif in_position and result["close"].iloc[i] < result["sma200"].iloc[i]:
            # In position and close dropped below sma200 - stop loss triggered
            result.iloc[i, result.columns.get_loc("stop_loss")] = 1
            result.iloc[i, result.columns.get_loc("position")] = 0
            in_position = False
            entry_price = None
            entry_date = None
            stop_loss_cooldown = 5  # 5 day cooldown before re-entering
        # elif in_position and entry_date is not None and (result.index[i] - entry_date).days >= 10:
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


def strategy_return_distribution(
    df: pd.DataFrame, n: int = 15, position_scaling: bool = False
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Calculate and plot the distribution of total returns per trade.

    A trade is defined as the period from an entry (position 0->1) to an exit (position 1->0).
    Total return is the percentage return from the entry close price to the exit close price.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns: close, position, sma20.
    n : int, optional
        Number of days for signal decay analysis. Default is 15.
    position_scaling : bool, optional
        If True, scales total return by signal (total_return = total_return * signal).
        Default is False.

    Returns
    -------
    tuple[pd.DataFrame, go.Figure]
        - DataFrame with trade returns (columns: entry_date, exit_date, entry_price, exit_price, total_return, signal, holding_period)
        - Plotly figure with 8 subplots in a single column: histogram, scatter, bar chart, scatter, scatter, cumulative return curve, underwater plot, signal decay.
    """
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None
    entry_sma20 = None
    entry_lower_band = None

    for i in range(len(df)):
        if df["position"].iloc[i] == 1 and not in_trade:
            in_trade = True
            entry_date = df.index[i]
            entry_price = df["next_open"].iloc[i]
            entry_sma20 = df["sma20"].iloc[i]
            entry_lower_band = df["lower_band"].iloc[i]
        elif df["position"].iloc[i] == 0 and in_trade:
            exit_date = df.index[i]
            exit_price = df["next_open"].iloc[i]
            total_return = (exit_price - entry_price) / entry_price * 100
            signal = abs((entry_price - entry_sma20) / (entry_sma20 - entry_lower_band))
            if position_scaling:
                total_return = total_return * signal
            holding_period = (exit_date - entry_date).days
            # Calculate IV cumulative return for the same duration
            iv_entry_price = (
                df.loc[entry_date, "iv_close"]
                if "iv_close" in df.columns and entry_date in df.index
                else None
            )
            iv_exit_price = (
                df.loc[exit_date, "iv_close"]
                if "iv_close" in df.columns and exit_date in df.index
                else None
            )
            iv_cumulative_return = (
                ((iv_exit_price - iv_entry_price) / iv_entry_price * 100)
                if iv_entry_price and iv_exit_price
                else np.nan
            )
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "total_return": total_return,
                    "signal": signal,
                    "holding_period": holding_period,
                    "iv_cumulative_return": iv_cumulative_return,
                }
            )
            in_trade = False
            entry_date = None
            entry_price = None
            entry_sma20 = None
            entry_lower_band = None

    trades_df = pd.DataFrame(trades)
    trades_df_sorted = trades_df.sort_values("exit_date").copy()
    trades_df_sorted["cumulative_return"] = (
        trades_df_sorted["total_return"] / 100
    ).cumsum()

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=8,
        cols=1,
        subplot_titles=(
            "Trade Return Distribution",
            "Signal vs Total Return",
            "Average Return by Holding Period",
            "Total Return vs IV Cumulative Return",
            "Avg Holding Period vs IV Cumulative Return",
            "Cumulative Return Curve",
            "Underwater Plot (Drawdown from Peak)",
            "Signal Decay",
        ),
        vertical_spacing=0.05,
    )

    # Histogram (row 1)
    if len(trades_df) > 0:
        fig.add_trace(
            go.Histogram(
                x=trades_df["total_return"],
                nbinsx=30,
                name="Returns",
                marker=dict(color="steelblue", line=dict(color="black")),
                hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        mean_val = trades_df["total_return"].mean()
        median_val = trades_df["total_return"].median()
        fig.add_vline(x=mean_val, line=dict(color="red", dash="dash"), row=1, col=1)
        fig.add_vline(
            x=median_val, line=dict(color="orange", dash="dash"), row=1, col=1
        )
        fig.add_annotation(
            x=0.02,
            y=0.95,
            xref="x domain",
            yref="y domain",
            text=f"Mean: {mean_val:.2f}%<br>Median: {median_val:.2f}%",
            showarrow=False,
            font=dict(color="red", size=10),
            align="left",
            bgcolor="white",
            borderpad=4,
            row=1,
            col=1,
        )

    # Scatter: signal vs return (row 2)
    if len(trades_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=trades_df["signal"],
                y=trades_df["total_return"],
                mode="markers",
                name="Trades",
                marker=dict(color="steelblue", line=dict(color="black")),
                text=trades_df["entry_date"].dt.strftime("%Y-%m-%d"),
                hovertemplate="Date: %{text}<br>Signal: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        slope, intercept = np.polyfit(trades_df["signal"], trades_df["total_return"], 1)
        x_line = np.linspace(trades_df["signal"].min(), trades_df["signal"].max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line + intercept,
                mode="lines",
                name=f"Trend (slope: {slope:.2f})",
                line=dict(color="red", dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=0.98,
            y=0.95,
            xref="x domain",
            yref="y domain",
            text=f"Slope: {slope:.2f}",
            showarrow=False,
            font=dict(color="red", size=11),
            align="right",
            bgcolor="white",
            borderpad=4,
            row=2,
            col=1,
        )

    # Bar chart: avg return by holding period (row 3)
    if len(trades_df) > 0:
        agg_by_holding = (
            trades_df.groupby("holding_period")
            .agg(avg_return=("total_return", "mean"), count=("total_return", "count"))
            .reset_index()
        )
        overall_mean = trades_df["total_return"].mean()
        fig.add_trace(
            go.Bar(
                x=agg_by_holding["holding_period"],
                y=agg_by_holding["avg_return"],
                name="Avg Return",
                marker=dict(color="steelblue", line=dict(color="black")),
                text=[f"n={int(c)}" for c in agg_by_holding["count"]],
                textposition="outside",
                hovertemplate="Holding: %{x} days<br>Avg Return: %{y:.2f}%<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=3, col=1)
        fig.add_hline(
            y=overall_mean,
            line=dict(color="red", dash="dash"),
            annotation_text=f"Mean: {overall_mean:.2f}%",
            row=3,
            col=1,
        )

    # Scatter: Total Return vs IV Cumulative Return (row 4)
    if len(trades_df) > 0 and "iv_cumulative_return" in trades_df.columns:
        valid_iv = trades_df["iv_cumulative_return"].notna()
        if valid_iv.any():
            fig.add_trace(
                go.Scatter(
                    x=trades_df.loc[valid_iv, "total_return"],
                    y=trades_df.loc[valid_iv, "iv_cumulative_return"],
                    mode="markers",
                    name="Trades",
                    marker=dict(color="steelblue", line=dict(color="black")),
                    text=trades_df.loc[valid_iv, "entry_date"].dt.strftime("%Y-%m-%d"),
                    hovertemplate="Date: %{text}<br>Total Return: %{x:.2f}%<br>IV Cumulative Return: %{y:.2f}%<extra></extra>",
                ),
                row=4,
                col=1,
            )
            # Regression trend line
            x_vals = trades_df.loc[valid_iv, "total_return"].values
            y_vals = trades_df.loc[valid_iv, "iv_cumulative_return"].values
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=slope * x_line + intercept,
                    mode="lines",
                    name=f"Trend (slope: {slope:.2f})",
                    line=dict(color="red", dash="dash"),
                ),
                row=4,
                col=1,
            )
            fig.add_annotation(
                x=0.98,
                y=0.95,
                xref="x domain",
                yref="y domain",
                text=f"Slope: {slope:.2f}",
                showarrow=False,
                font=dict(color="red", size=11),
                align="right",
                bgcolor="white",
                borderpad=4,
                row=4,
                col=1,
            )

    # Scatter: Avg Holding Period vs IV Cumulative Return (row 5)
    if len(trades_df) > 0 and "iv_cumulative_return" in trades_df.columns:
        valid_iv = trades_df["iv_cumulative_return"].notna()
        if valid_iv.any():
            agg_by_holding = (
                trades_df[valid_iv]
                .groupby("holding_period")
                .agg(
                    avg_iv_return=("iv_cumulative_return", "mean"),
                    count=("iv_cumulative_return", "count"),
                )
                .reset_index()
            )
            fig.add_trace(
                go.Scatter(
                    x=agg_by_holding["holding_period"],
                    y=agg_by_holding["avg_iv_return"],
                    mode="markers+lines",
                    name="Avg IV Return",
                    marker=dict(color="steelblue", line=dict(color="black"), size=8),
                    text=[f"n={int(c)}" for c in agg_by_holding["count"]],
                    hovertemplate="Holding: %{x} days<br>Avg IV Return: %{y:.2f}%<extra></extra>",
                ),
                row=5,
                col=1,
            )
            fig.add_hline(y=0, line=dict(color="black", width=0.8), row=5, col=1)

    # Cumulative return curve (row 6)
    if len(trades_df_sorted) > 0:
        fig.add_trace(
            go.Scatter(
                x=trades_df_sorted["exit_date"],
                y=trades_df_sorted["cumulative_return"] * 100,
                mode="lines+markers",
                name="Cumulative Return",
                line=dict(color="blue", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,0,255,0.2)",
                hovertemplate="Date: %{x}<br>Cumulative: %{y:.2f}%<extra></extra>",
            ),
            row=6,
            col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=6, col=1)

    # Underwater plot (row 7)
    if len(trades_df_sorted) > 0:
        cumulative = trades_df_sorted["cumulative_return"].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) * 100
        max_dd_idx = np.argmin(drawdown)
        fig.add_trace(
            go.Scatter(
                x=trades_df_sorted["exit_date"],
                y=drawdown,
                mode="lines",
                name="Drawdown",
                line=dict(color="darkred", width=1),
                fill="tozeroy",
                fillcolor="rgba(255,0,0,0.5)",
                hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
            ),
            row=7,
            col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=7, col=1)
        if len(drawdown) > 0:
            fig.add_annotation(
                x=trades_df_sorted["exit_date"].iloc[max_dd_idx],
                y=drawdown[max_dd_idx],
                text=f"Max DD: {drawdown[max_dd_idx]:.2f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=30,
                ay=-30,
                font=dict(size=9),
                row=7,
                col=1,
            )

    # Signal decay plot (row 8)
    if len(trades_df) > 0:
        decay_returns = []
        for n_days in range(1, n + 1):
            daily_returns = []
            for _, trade in trades_df.iterrows():
                entry_date = trade["entry_date"]
                entry_idx = df.index.get_loc(entry_date)
                if entry_idx + n_days < len(df):
                    n_day_date = df.index[entry_idx + n_days]
                    entry_price = trade["entry_price"]
                    n_day_price = df.loc[n_day_date, "close"]
                    cum_return = (n_day_price - entry_price) / entry_price * 100
                    daily_returns.append(cum_return)
            if daily_returns:
                decay_returns.append(np.mean(daily_returns))
            else:
                decay_returns.append(np.nan)

        fig.add_trace(
            go.Scatter(
                x=list(range(1, n + 1)),
                y=decay_returns,
                mode="lines+markers",
                name="Signal Decay",
                line=dict(color="purple"),
                hovertemplate="Day: %{x}<br>Avg Cumulative Return: %{y:.2f}%<extra></extra>",
            ),
            row=8,
            col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=8, col=1)

    fig.update_layout(
        title="Strategy Return Analysis",
        height=3200,
        width=800,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Total Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Signal", row=2, col=1)
    fig.update_yaxes(title_text="Total Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Holding Period (Days)", row=3, col=1)
    fig.update_yaxes(title_text="Avg Return (%)", row=3, col=1)
    fig.update_xaxes(title_text="Total Return (%)", row=4, col=1)
    fig.update_yaxes(title_text="IV Cumulative Return (%)", row=4, col=1)
    fig.update_xaxes(title_text="Holding Period (Days)", row=5, col=1)
    fig.update_yaxes(title_text="IV Cumulative Return (%)", row=5, col=1)
    fig.update_xaxes(title_text="Exit Date", row=6, col=1)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=6, col=1)
    fig.update_xaxes(title_text="Date", row=7, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=7, col=1)
    fig.update_xaxes(title_text="Days Since Entry", row=8, col=1)
    fig.update_yaxes(title_text="Avg Cumulative Return (%)", row=8, col=1)

    return trades_df, fig


def plot_strategy_interactive(df: pd.DataFrame) -> go.Figure:
    """
    Plot SMA bands strategy with entry signals.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index and columns: sma200, close, upper_band,
        lower_band, entry (optional).

    Returns
    -------
    go.Figure
        The Plotly figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df.index, y=df["sma200"], name="SMA 200", line=dict(color="blue"))
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["sma20"], name="SMA 20", line=dict(color="orange"))
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["close"], name="Close", line=dict(color="black", width=1)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["upper_band"],
            name="Upper Band",
            line=dict(color="red", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["lower_band"],
            name="Lower Band",
            line=dict(color="green", dash="dot"),
        )
    )
    if "next_open" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["next_open"],
                name="Next Open",
                line=dict(color="gray", width=1, dash="dot"),
                opacity=0.5,
            )
        )

    if "entry" in df.columns:
        entry_signals = df[df["entry"].notna()]
        entry_y = (
            entry_signals["next_open"]
            if "next_open" in entry_signals.columns
            else entry_signals["close"]
        )
        fig.add_trace(
            go.Scatter(
                x=entry_signals.index,
                y=entry_y,
                name="Entry",
                mode="markers",
                marker=dict(symbol="arrow-up", color="green", size=12),
            )
        )

    if "exit" in df.columns:
        exit_signals = df[df["exit"].notna()]
        fig.add_trace(
            go.Scatter(
                x=exit_signals.index,
                y=exit_signals["close"],
                name="Exit",
                mode="markers",
                marker=dict(symbol="arrow-down", color="orange", size=12),
            )
        )

    if "stop_loss" in df.columns:
        stop_loss_signals = df[df["stop_loss"].notna()]
        fig.add_trace(
            go.Scatter(
                x=stop_loss_signals.index,
                y=stop_loss_signals["close"],
                name="Stop Loss",
                mode="markers",
                marker=dict(symbol="arrow-down", color="red", size=12),
            )
        )

    fig.update_layout(
        title="Strategy - SMA Bands with Entry/Exit Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )

    if "iv_daily_close" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["iv_daily_close"],
                name="IV Daily Close",
                line=dict(color="red"),
                opacity=0.5,
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(title="IV Daily Close", overlaying="y", side="right")
        )

    return fig
