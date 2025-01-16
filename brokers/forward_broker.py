from pybacktestchain.broker import Backtest,Broker
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

    def trade_forward(self, date, forward_price, maturity_date, predicted_spot_price,actual_spot):
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
            self.log_transaction(date, "SELL_FORWARD", 1, forward_price, 0)  # No upfront cost for selling

