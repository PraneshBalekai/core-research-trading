from __future__ import annotations

import argparse
import datetime
import json

import pandas as pd
from dynaconf import Dynaconf

from cio.data_loader import load_data
from cio.data_writer import write_data

parser = argparse.ArgumentParser(description="Path of config file to pass to script")
parser.add_argument("--config_path", type=str, help="Path to config file")


def main(config_path: str, **kwargs):
    """Loads config from config_path, gets historical data for multiple symbols and writes to target files.

    Example config: {
        "symbols": ["QQQ", "SPY", "TLT"],
        "base_config": {
            "loader_config": {
                "loader_class": "IBKRHistoricalDataLoader",
                "contract": {
                    "secType": "STK",
                    "exchange": "SMART",
                    "currency": "USD"
                },
                "ibkr_params": {
                    "endDateTime": "@format {this.endDateTime} 09:30:00 US/Eastern",
                    "durationStr": "22 Y",
                    "barSizeSetting": "1 day",
                    "whatToShow": "TRADES",
                    "useRTH": True,
                    "formatDate": 2,
                    "keepUpToDate": False,
                    "chartOptions": []
                }
            },
            "script_config": {
                "timezone": "US/Eastern",
                "format_date": "%Y%m%d"
            },
            "writer_params": {
                "output_dir": "/Users/praneshbalekai/Desktop/IB_PRD/data",
                "append_if_exists": True,
                "sort_index": True,
                "deduplicate_index": True
            }
        }
    }
    """
    # load all config params
    with open(config_path) as f:
        config = json.load(f)

    symbols = config["symbols"]
    base_config = config["base_config"]
    writer_params = base_config.get("writer_params", {})
    output_dir = writer_params.get(
        "output_dir", "/Users/praneshbalekai/Desktop/IB_PRD/data"
    )

    for symbol in symbols:
        print(f"Processing symbol: {symbol}")

        # Build config for this symbol
        dc = Dynaconf()

        # Set endDateTime
        endDateTime = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        dc["endDateTime"] = endDateTime

        # Build loader_config for this symbol
        loader_config = json.loads(json.dumps(base_config["loader_config"]))
        loader_config["contract"]["symbol"] = symbol
        dc["loader_config"] = loader_config

        # Load data
        data = load_data(dc["loader_config"], **kwargs)

        # Change to timezone aware timestamp
        script_config = base_config["script_config"]
        if "format_date" in script_config:
            data.index = pd.to_datetime(data.index, format=script_config["format_date"])
            data.index = pd.Series(data.index).dt.tz_localize(script_config["timezone"])
        else:
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = pd.Series(data.index).dt.tz_localize("UTC")
            data.index = pd.Series(data.index).dt.tz_convert(script_config["timezone"])

        # Build writer_config for this symbol
        filename = f"{output_dir}/{symbol.lower()}_daily.parquet"
        writer_config = {
            "writer_class": "ParquetWriter",
            "filename": filename,
            "writer_params": {
                k: v for k, v in writer_params.items() if k != "output_dir"
            },
        }

        write_data(data, writer_config)
        print(f"Wrote {len(data)} rows to {filename}")

    return


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Input args: {args.__dict__}")

    main(args.config_path)
