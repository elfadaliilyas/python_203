from pybacktestchain.broker import Broker
from dataclasses import dataclass
import pandas as pd

@dataclass
class ForwardContract:
    purchase_date: pd.Timestamp
    maturity_date: pd.Timestamp
    forward_price: float
    position_type: str  # 'BUY' or 'SELL'
