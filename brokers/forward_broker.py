from pybacktestchain.broker import Backtest, Broker
from dataclasses import dataclass
import pandas as pd
from datetime import datetime


@dataclass
class ForwardContract:
    purchase_date: pd.Timestamp
    maturity_date: pd.Timestamp
    forward_price: float
    position_type: str  # 'BUY' or 'SELL'


class ForwardTradingBroker(Broker):
    def __init__(self, cash: float):
        """
        Initializes the broker with a starting cash balance and necessary attributes.
        """
        super().__init__(cash)
        self.positions = []  # List of open forward contracts
        self.behaviour = []  # List of realized profits and losses
        self.positions_taken = []
        self.list_payoff = []

    def trade_forward(
        self, date, forward_price, maturity_date, predicted_spot_price, actual_spot
    ):
        """
        Buys or sells a forward contract based on the predicted spot price.

        Args:
            date (pd.Timestamp): The current date.
            forward_price (float): The forward price for a one-month contract.
            maturity_date (pd.Timestamp): The maturity date of the forward.
            predicted_spot_price (float): The predicted spot price in one month.
        """
        if predicted_spot_price > forward_price:
            # Buy forward contract
            self.positions.append(
                ForwardContract(
                    purchase_date=date,
                    maturity_date=maturity_date,
                    forward_price=forward_price,
                    position_type="BUY",
                )
            )
            self.log_transaction(date, "BUY_FORWARD", 1, forward_price, self.cash)

        elif predicted_spot_price < forward_price:
            # Sell forward contract
            self.positions.append(
                ForwardContract(
                    purchase_date=date,
                    maturity_date=maturity_date,
                    forward_price=forward_price,
                    position_type="SELL",
                )
            )
            self.log_transaction(
                date, "SELL_FORWARD", 1, forward_price, 0
            )  # No upfront cost for selling

    def close_expired_positions(self, current_date, actual_spot_price):
        """
        Closes all forward contracts that mature on or before the current date.

        Args:
            current_date (pd.Timestamp): The current date.
            actual_spot_price (float): The actual spot price at the maturity date.
        """
        for pos in self.positions:
            if pos.position_type == "BUY":
                payoff = actual_spot_price - pos.forward_price
            elif pos.position_type == "SELL":
                payoff = pos.forward_price - actual_spot_price
            self.cash += payoff
            self.list_payoff.append(self.cash)

            self.behaviour.append(pos.position_type)
            self.log_transaction(
                current_date, "CLOSE_;FORWARD", 1, pos.forward_price, payoff
            )
            break

    def run_backtest(self, data: pd.DataFrame):
        """
        Runs a backtest using the provided data.

        Args:
            data (pd.DataFrame): A DataFrame containing columns: 'date', 'spot_price', 'forward_price', 'predicted_spot'.
        """
        data["maturity_date"] = pd.to_datetime(data["Date"]) + pd.Timedelta(days=30)

        for _, row in data.iterrows():
            current_date = row["Date"]
            forward_price = row["Brent Futures"]
            predicted_spot = row["predicted_spot_price"]
            maturity_date = row["maturity_date"]
            actual_spot = row["actual_values"]
            # Trade based on prediction
            self.trade_forward(
                current_date, forward_price, maturity_date, predicted_spot, actual_spot
            )

        # Close expired positions
        self.close_expired_positions(current_date, data["actual_values"])
        self.cash = pd.DataFrame(self.cash)["actual_values"].iloc[-1]
        # Final portfolio value
        return self.cash

