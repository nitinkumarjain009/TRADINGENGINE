"""
Configuration settings for the stock analysis application
"""
import os
import pytz

# API Keys and IDs
TELEGRAM_BOT_TOKEN = "8017759392:AAEwM-W-y83lLXTjlPl8sC_aBmizuIrFXnU"
TELEGRAM_CHAT_ID = "711856868" 
TELEGRAM_GROUP_CHANNEL = "@Stockniftybot"

# Data Sources
STOCK_LIST_CSV = "nifty50_stocks.csv"
DEFAULT_STOCKS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", 
    "ITC.NS", "KOTAKBANK.NS", "LT.NS", "HINDUNILVR.NS", "SBIN.NS"
]  # Fallback if CSV not available

# Time Settings
CACHE_DURATION_SECONDS = 3600  # 1 hour
SCAN_INTERVAL_MINUTES = 10
INDIA_TIMEZONE = pytz.timezone('Asia/Kolkata')
DATA_FETCH_PERIOD = "6mo"
BACKTEST_PERIOD = "1y"

# Market Hours
MARKET_OPEN_HOUR = 9  # 9 AM
MARKET_OPEN_MINUTE = 15  # 9:15 AM
MARKET_CLOSE_HOUR = 15  # 3 PM
MARKET_CLOSE_MINUTE = 30  # 3:30 PM

# Portfolio Settings
INITIAL_CASH = 100000
DEFAULT_BACKTEST_SYMBOL = "RELIANCE.NS"

# Flask server settings
PORT = int(os.environ.get('PORT', 8080))
HOST = '0.0.0.0'
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
