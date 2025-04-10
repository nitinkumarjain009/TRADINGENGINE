"""
Backtesting engine
"""
import pandas as pd
import numpy as np
import logging
from models.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self, symbol, data, initial_capital=100000, commission=0.001):
        self.symbol = symbol
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
