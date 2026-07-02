from __future__ import annotations

import json

import click
import pandas as pd

from core.strategy.ivmr import (
    calculate_entry_signals,
    calculate_sma_bands,
    combine_close_prices,
)
from etl.update_historical_data import main as update_historical_data
from trading.broker import get_broker, get_positions
from trading.consts import IBKR


def load_data_from_etl_config(etl_config_path: str) -> pd.DataFrame:
    """Load parquet data from an ETL config file."""
    with open(etl_config_path, "r") as f:
        config = json.load(f)
    return pd.read_parquet(config["writer_config"]["filename"])


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-c",
    "--config-path",
    type=str,
    default="/Users/praneshbalekai/Desktop/IB_PRD/trading/configs/ivmr/config.json",
    help="Path to config file",
)
@click.option(
    "-d",
    "--is-docker-run",
    is_flag=True,
    help="Flag to set if this script is run in docker",
)
def first_run(config_path: str, is_docker_run: bool):
    print(f"Input args: config_path={config_path}, is_docker_run={is_docker_run}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Init broker session
    ibkr_app = get_broker(broker_name=IBKR, is_docker_run=is_docker_run)

    try:
        positions = get_positions(broker_app=ibkr_app)
        print(f"Positions: {positions}")

        for symbol, symbol_config in config.items():
            print(f"\n=== Processing {symbol} ===")
            print(f"Config: {symbol_config}")

            etl_config = symbol_config["etl_config"]
            print(f"Loading stock data from: {etl_config['stock']}")
            update_historical_data(etl_config["stock"], ibkr_app=ibkr_app)
            ibkr_app.historical_data = pd.DataFrame()
            print(f"Loading IV data from: {etl_config['iv']}")
            update_historical_data(etl_config["iv"], ibkr_app=ibkr_app)
            ibkr_app.historical_data = pd.DataFrame()

            stock_df = load_data_from_etl_config(etl_config["stock"])
            iv_df = load_data_from_etl_config(etl_config["iv"])

            combined_df = combine_close_prices(stock_df, iv_df)
            bands_df = calculate_sma_bands(combined_df)

            position = positions.get(symbol, 0)
            if position == 0:
                entry_signals_df = calculate_entry_signals(bands_df)
                entry_signal = entry_signals_df["entry"].iloc[-1] == 1
                print(f"Entry signal for {symbol}: {entry_signal}")
            else:
                print(
                    f"Skipping entry calculation for {symbol}: has position {position}"
                )
    except Exception as e:
        ibkr_app.disconnect()
        raise e

    ibkr_app.disconnect()


@cli.command()
@click.option(
    "-c",
    "--config-path",
    type=str,
    default="/Users/praneshbalekai/Desktop/IB_PRD/trading/configs/ivmr/config.json",
    help="Path to config file",
)
@click.option(
    "-d",
    "--is-docker-run",
    is_flag=True,
    help="Flag to set if this script is run in docker",
)
def second_run(config_path: str, is_docker_run: bool):
    print(f"Input args: config_path={config_path}, is_docker_run={is_docker_run}")


if __name__ == "__main__":
    cli()
