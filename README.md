# Stock Trading Bot

A web application that provides stock trading recommendations based on technical analysis for Nifty 50 stocks. The application fetches stock data, analyzes it using various technical indicators, generates buy/sell recommendations, displays them on a webpage, and sends notifications to Telegram.

## Features

- Fetches stock data from Yahoo Finance
- Analyzes stocks using technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ADX)
- Generates buy/sell recommendations with target prices
- Displays interactive charts with Plotly
- Sends notifications to Telegram for strong buy/sell signals
- Includes paper trading model with backtesting capabilities
- Responsive web interface built with Bootstrap

## Deployed Application

The application is deployed at: https://robot-pdwz.onrender.com/

## Technical Indicators Used

1. **Simple Moving Averages (SMA)** - 20-day and 50-day moving averages to identify trends
2. **Moving Average Convergence Divergence (MACD)** - Momentum indicator showing relationship between two moving averages
3. **Relative Strength Index (RSI)** - Momentum oscillator measuring speed and change of price movements
4. **Bollinger Bands** - Volatility indicator showing price channels
5. **Average Directional Index (ADX)** - Trend strength indicator

## Signal Generation Logic

- **Strong Buy** signals are generated when multiple indicators align positively
- **Buy** signals occur when some indicators are positive
- **Hold** is the default when indicators are mixed
- **Sell** signals occur when some indicators are negative
- **Strong Sell** signals are generated when multiple indicators align negatively

## Backtesting Engine

The application includes a backtesting engine that simulates how the trading strategy would have performed historically. Key metrics include:

- Number of trades
- Win rate
- Profit factor
- Total profit
- Performance compared to buy-and-hold strategy

## Telegram Integration

The bot sends notifications to:
- Individual chat: Using chat ID 711856868
- Group channel: @Stockniftybot

## Deployment

### Prerequisites

- Python 3.8+
- Pip

### Local Development

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-trading-bot.git
   cd stock-trading-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the application at http://localhost:5000

### Deploying to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Python Version: 3.11

4. Add the following environment variables:
   - `TELEGRAM_TOKEN`: Your Telegram bot token
   - `TELEGRAM_CHAT_ID`: Your Telegram chat ID
   - `TELEGRAM_CHANNEL`: Your Telegram channel name

## Project Structure

```
stock-trading-bot/
├── app.py                 # Main application file
├── nifty50_stocks.csv     # List of Nifty 50 stocks
├── requirements.txt       # Python dependencies
├── render.yaml            # Render configuration
└── templates/             # HTML templates
    ├── index.html         # Main page template
    └── backtest.html      # Backtest results template
```

## Future Enhancements

- Add user authentication
- Allow customization of technical indicators
- Implement portfolio tracking
- Add more advanced statistical analysis
- Include fundamental analysis data
