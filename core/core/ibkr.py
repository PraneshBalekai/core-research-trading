# historical
import warnings

import pandas as pd
from ibapi.client import EClient, OrderId
from ibapi.wrapper import EWrapper

warnings.filterwarnings("ignore")


class IBKRSession(EClient, EWrapper):
    """Unified IBKR session for historical data and trading operations."""

    def __init__(self):
        self.orderId = 0
        EClient.__init__(self, self)
        self.positions = {}
        self.position_query_end = False
        self.orders = {}
        self.historical_data = pd.DataFrame()
        self.historical_query_end = False

    def nextValidId(self, orderId: OrderId):
        self.orderId = orderId

    def nextId(self):
        self.orderId += 1
        return self.orderId

    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(
            f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}"
        )

    # --- Historical Data ---
    def historicalData(self, reqId, bar):
        df = {}
        bar_data = {
            "close": bar.close,
            "open": bar.open,
            "low": bar.low,
            "high": bar.high,
            "volume": bar.volume,
            "count": bar.barCount,
        }
        df[bar.date] = bar_data
        df = pd.DataFrame.from_dict(df, orient="index")
        self.historical_data = pd.concat([self.historical_data, df])

    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical Data Ended for {reqId}. Started at {start}, ending at {end}")
        self.cancelHistoricalData(reqId)
        self.historical_query_end = True

    def headTimestamp(self, reqId: int, headTimestamp: str):
        print(f"reqId: {reqId}, headTimestamp: {headTimestamp}")
        self.cancelHeadTimeStamp(reqId)

    # --- Trading / Positions ---
    def position(self, account, contract, position, avgCost):
        super().position(account, contract, position, avgCost)
        self.positions[contract.symbol] = {
            "position": position,
            "avgCost": avgCost,
            "account": account,
        }

    def positionEnd(self):
        self.position_query_end = True
        print("Position batch ended")

    def orderStatus(
        self,
        orderId,
        status,
        filled,
        remaining,
        avgFillPrice,
        permId,
        parentId,
        lastFillPrice,
        clientId,
        whyHeld,
        mktCapPrice="",
    ):
        super().orderStatus(
            orderId,
            status,
            filled,
            remaining,
            avgFillPrice,
            permId,
            parentId,
            lastFillPrice,
            clientId,
            whyHeld,
            mktCapPrice,
        )
        self.orders[orderId] = {
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avgFillPrice": avgFillPrice,
            "lastFillPrice": lastFillPrice,
        }

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        self.orders[orderId] = {
            "contract": contract,
            "order": order,
            "orderState": orderState,
        }

    def openOrderEnd(self):
        print("Open order batch ended")

    def execDetails(self, reqId, contract, execution):
        super().execDetails(reqId, contract, execution)
        print(
            f"Execution details - Symbol: {contract.symbol}, Quantity: {execution.shares}, Price: {execution.price}"
        )
