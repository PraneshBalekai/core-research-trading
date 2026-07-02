from __future__ import annotations

import glob
import threading
import time
from abc import ABC, abstractmethod
from urllib.parse import urlencode

import pandas as pd
import requests
from ibapi.contract import Contract

import cio.constants as c
import external.binance as binance
from core.ibkr import IBKRSession


class BaseLoader(ABC):
    def __init__(self, config: dict):
        """Base Loader class to load data from different sources based on config.

        To begin with, we will stick to cross-sectional dataframes for data and all data classes
        will returns a None / pd.DataFrame

        Args:
            config (dict): Dictionary with values related to the source and other details
                of the data loader
        """
        self.config = config

    @abstractmethod
    def load_data(self):
        pass


class ParquetDataFrameLoader(BaseLoader):
    """Loads data from a parquet file as a dataframe.

    Example config:
    {
        "loader_class": "ParquetDataFrameLoader",
        "filename": "../data/instruments_token.parquet"
    }
    """

    def load_data(self):
        data = pd.read_parquet(self.config["filename"])
        return data


class BinanceHistoricalDataLoader(BaseLoader):
    """Loads historicla data from Binance Marketdata endpoint.

    Example config:
    {
        "loader_class": "BinanceHistoricalDataLoader",
        "endpoint_type": None # KEY | SIGNATURE
        "params": {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": "1745769121",
            "endTime": "1745769121",
        }
    }
    """

    def load_data(self):
        endpoint = "/api/v3/klines"
        query_string = urlencode(self.config["params"])

        # TODO: endpoint_type == "KEY"
        signature = None
        if (
            "endpoint_type" in self.config
            and self.config["endpoint_type"] == "SIGNATURE"
        ):
            signature = binance.get_query_signature(query_string)

        if signature is None:
            url = f"{binance.BASE_URL}{endpoint}?{query_string}"
        else:
            url = f"{binance.BASE_URL}{endpoint}?{query_string}&signature={signature}"

        response = requests.get(url, headers=binance.DEFAULT_HEADERS)

        response.raise_for_status()

        # convert into DF
        df = pd.DataFrame(
            response.json(),
            columns=[
                "kline_open_time",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "kline_close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "NIL",
            ],
        )

        # Do the following in the ETL script. Save copy of raw data.
        # Timezone aware datetime index
        # required fields: close, open, low, high, volume, count

        # Return raw data
        return df


class ParquetCustomLoader(BaseLoader):
    """Loads data from multiple parquet files in a directory with symbol column.

    Files are expected to be named as {symbol.lower()}_*.parquet

    Example config:
    {
        "loader_class": "ParquetCustomLoader",
        "input_dir": "../data/parquet_files/"
    }
    """

    def load_data(self):
        input_dir = self.config["input_dir"]
        pattern = f"{input_dir}/*.parquet"
        files = glob.glob(pattern)

        frames = []
        for filepath in files:
            filename = filepath.split("/")[-1]
            symbol = filename.split("_")[0].upper()
            df = pd.read_parquet(filepath)
            df["symbol"] = symbol
            frames.append(df)

        return pd.concat(frames)


class IBKRHistoricalDataLoader(BaseLoader):
    """Loads historical data from IBKR based on config.

    This IBKR implementation connects to the app, loads and disconnects. This makes the app
    synchronous. What is an ideal solution that can use the same app but can pass custom hanndler functions
    for each function, for ex, load historical data, send order etc?

    Example config:
    {
        "loader_class": "IBKRHistoricalDataLoader",
        "contract": {
            "symbol": "SPY",
            "secType": "STK",
            "exchange": "SMART",
            "currency": "USD"
        },
        "ibkr_params": {
            "endDateTime": "20241211 09:30:00 US/Eastern",
            "durationStr": "1 D",
            "barSizeSetting": "1 min",
            "whatToShow": "TRADES",
            "useRTH": True,
            "formatDate": 2, #2 stands for epoch seconds - use `localize_index` under loader params
            "keepUpToDate": False,
            "chartOptions": []
        }
    }
    """

    def load_data(self, ibkr_app=None):
        # Use provided app or create a new one
        should_disconnect = ibkr_app is None
        if ibkr_app is None:
            ibkr_app = IBKRSession()
            ibkr_app.connect("127.0.0.1", 4002, 0)
            threading.Thread(target=ibkr_app.run).start()
            time.sleep(1)

        # Send Query
        ibkr_params = self.config["ibkr_params"]
        ibkr_params["reqId"] = ibkr_app.nextId()
        ibkr_params["contract"] = Contract()
        for k, v in self.config["contract"].items():
            setattr(ibkr_params["contract"], k, v)
        ibkr_app.reqHistoricalData(**ibkr_params)

        # Run while loop until historical_query_end is set to True
        while not ibkr_app.historical_query_end:
            time.sleep(1)
        time.sleep(15)
        # Reset this after query end, in case other queries need to be run
        ibkr_app.historical_query_end = False

        if should_disconnect:
            ibkr_app.disconnect()
        return ibkr_app.historical_data


def load_data(config: dict, **kwargs):
    """Based on config, call relevant data loader function.

    Args:
        config (dict): Config Dict object
    """
    source = config[c.loader_class]
    if source == "ParquetDataFrameLoader":
        loader = ParquetDataFrameLoader(config)
    elif source == "ParquetCustomLoader":
        loader = ParquetCustomLoader(config)
    elif source == "IBKRHistoricalDataLoader":
        loader = IBKRHistoricalDataLoader(config)
    elif source == "BinanceHistoricalDataLoader":
        loader = BinanceHistoricalDataLoader(config)
    else:
        raise ValueError("Not a valid data source for data loader")

    data = loader.load_data(**kwargs)
    return data
