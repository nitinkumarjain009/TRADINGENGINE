# app.py
import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import json
from flask import Flask, render_template
import threading
import time
import csv
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)

# Configuration variables
TELEGRAM_TOKEN = "8017759392:AAEwM-W-y83lLXTjlPl8sC_aBmizuIrFXnU"
TELEGRAM_CHAT_ID = "711856868"
TELEGRAM_CHANNEL = "@Stockniftybot"
CSV_FILE = "nifty50_stocks.csv"
UPDATE_INTERVAL = 3600  # Update data every hour

# Global variables
recommendations = []
last_update_time = None
backtest_results = {}

def send_telegram_message(message, chat_id=TELEGRAM_CHAT_ID):
    """Send message to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        print(f"Telegram API error: {e}")
        return None

def send_to_channel(message):
    """Send message to Telegram channel"""
    return send_telegram_message(message, TELEGRAM_CHANNEL)

def get_stock_list():
    """Read stock list from CSV file"""
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            return df['Symbol'].tolist() if 'Symbol' in df.columns else []
        else:
            print(f"CSV file {CSV_FILE} not found, using default Nifty 50 stocks")
            # Default Nifty 50 stocks if CSV file is not found
            return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                   "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS"]
    except Exception as e:
        print(f"Error reading stock list: {e}")
        return []

def calculate_indicators(data):
    """Calculate technical indicators"""
    # Calculate SMA
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate EMA
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['std'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA20'] + (data['std'] * 2)
    data['Lower_Band'] = data['MA20'] - (data['std'] * 2)
    
    # Calculate ADX
    plus_dm = np.zeros_like(data['High'])
    minus_dm = np.zeros_like(data['High'])
    
    for i in range(1, len(data)):
        h_diff = data['High'].iloc[i] - data['High'].iloc[i-1]
        l_diff = data['Low'].iloc[i-1] - data['Low'].iloc[i]
        
        if h_diff > l_diff and h_diff > 0:
            plus_dm[i] = h_diff
        else:
            plus_dm[i] = 0
            
        if l_diff > h_diff and l_diff > 0:
            minus_dm[i] = l_diff
        else:
            minus_dm[i] = 0
            
    data['Plus_DM'] = plus_dm
    data['Minus_DM'] = minus_dm
    
    # Calculate True Range
    data['TR'] = np.maximum(
        np.maximum(
            data['High'] - data['Low'],
            np.abs(data['High'] - data['Close'].shift(1))
        ),
        np.abs(data['Low'] - data['Close'].shift(1))
    )
    
    # Calculate ATR, +DI, -DI and ADX
    period = 14
    data['ATR'] = data['TR'].rolling(window=period).mean()
    data['Plus_DI'] = 100 * (data['Plus_DM'].rolling(window=period).mean() / data['ATR'])
    data['Minus_DI'] = 100 * (data['Minus_DM'].rolling(window=period).mean() / data['ATR'])
    
    data['DX'] = 100 * np.abs((data['Plus_DI'] - data['Minus_DI']) / (data['Plus_DI'] + data['Minus_DI']))
    data['ADX'] = data['DX'].rolling(window=period).mean()
    
    return data

def generate_signals(data):
    """Generate buy/sell signals based on indicators"""
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    signals['Signal'] = 0
    
    # SMA Crossover
    signals.loc[(data['SMA20'] > data['SMA50']) & (data['SMA20'].shift(1) <= data['SMA50'].shift(1)), 'SMA_Signal'] = 1  # Buy
    signals.loc[(data['SMA20'] < data['SMA50']) & (data['SMA20'].shift(1) >= data['SMA50'].shift(1)), 'SMA_Signal'] = -1  # Sell
    
    # MACD Signal
    signals.loc[(data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1)), 'MACD_Signal'] = 1  # Buy
    signals.loc[(data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1)), 'MACD_Signal'] = -1  # Sell
    
    # RSI Signal
    signals.loc[data['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold - Buy
    signals.loc[data['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought - Sell
    
    # Bollinger Bands
    signals.loc[data['Close'] < data['Lower_Band'], 'BB_Signal'] = 1  # Price below lower band - Buy
    signals.loc[data['Close'] > data['Upper_Band'], 'BB_Signal'] = -1  # Price above upper band - Sell
    
    # ADX Signal (Trend Strength)
    signals.loc[(data['ADX'] > 25) & (data['Plus_DI'] > data['Minus_DI']), 'ADX_Signal'] = 1  # Strong uptrend
    signals.loc[(data['ADX'] > 25) & (data['Plus_DI'] < data['Minus_DI']), 'ADX_Signal'] = -1  # Strong downtrend
    
    # Fill NaN values with 0
    signals = signals.fillna(0)
    
    # Composite Signal (average of all signals)
    signal_columns = ['SMA_Signal', 'MACD_Signal', 'RSI_Signal', 'BB_Signal', 'ADX_Signal']
    signals['Composite'] = signals[signal_columns].sum(axis=1)
    
    # Classification
    signals['Recommendation'] = 'Hold'
    signals.loc[signals['Composite'] >= 2, 'Recommendation'] = 'Strong Buy'
    signals.loc[(signals['Composite'] < 2) & (signals['Composite'] > 0), 'Recommendation'] = 'Buy'
    signals.loc[signals['Composite'] <= -2, 'Recommendation'] = 'Strong Sell'
    signals.loc[(signals['Composite'] > -2) & (signals['Composite'] < 0), 'Recommendation'] = 'Sell'
    
    # Calculate target prices (5% movement for strong signals, 3% for regular signals)
    current_price = data['Close'].iloc[-1]
    signals['Target_Price'] = current_price
    
    signals.loc[signals['Recommendation'] == 'Strong Buy', 'Target_Price'] = round(current_price * 1.05, 2)
    signals.loc[signals['Recommendation'] == 'Buy', 'Target_Price'] = round(current_price * 1.03, 2)
    signals.loc[signals['Recommendation'] == 'Strong Sell', 'Target_Price'] = round(current_price * 0.95, 2)
    signals.loc[signals['Recommendation'] == 'Sell', 'Target_Price'] = round(current_price * 0.97, 2)
    
    return signals.iloc[-1]

def analyze_stock(symbol):
    """Analyze a single stock and return recommendation"""
    try:
        # Add .NS suffix if not present (for NSE stocks)
        if not symbol.endswith(('.NS', '.BO')):
            symbol = f"{symbol}.NS"
        
        # Get historical data
        stock = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)  # Get 100 days of data
        hist = stock.history(start=start_date, end=end_date)
        
        if len(hist) < 50:  # Not enough data
            return None
        
        # Calculate indicators
        data = calculate_indicators(hist)
        
        # Generate signal
        signal = generate_signals(data)
        
        # Get company name
        try:
            info = stock.info
            company_name = info.get('longName', symbol)
        except:
            company_name = symbol
        
        # Prepare charts
        fig = create_stock_chart(hist, data, symbol)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        recommendation = {
            'symbol': symbol,
            'company_name': company_name,
            'current_price': round(hist['Close'].iloc[-1], 2),
            'recommendation': signal['Recommendation'],
            'target_price': signal['Target_Price'],
            'rsi': round(data['RSI'].iloc[-1], 2),
            'macd': round(data['MACD'].iloc[-1], 3),
            'adx': round(data['ADX'].iloc[-1], 2) if not np.isnan(data['ADX'].iloc[-1]) else None,
            'sma20': round(data['SMA20'].iloc[-1], 2),
            'sma50': round(data['SMA50'].iloc[-1], 2),
            'chart': chart_json,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return recommendation
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def create_stock_chart(hist, data, symbol):
    """Create stock chart with indicators using Plotly"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=('Price & Indicators', 'MACD', 'RSI'),
                        row_heights=[0.6, 0.2, 0.2])
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add SMA lines
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA20'],
            name='SMA 20',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA50'],
            name='SMA 50',
            line=dict(color='orange')
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Upper_Band'],
            name='Upper BB',
            line=dict(color='rgba(250, 0, 0, 0.5)')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Lower_Band'],
            name='Lower BB',
            line=dict(color='rgba(0, 250, 0, 0.5)')
        ),
        row=1, col=1
    )
    
    # Add MACD
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            name='MACD',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Signal'],
            name='Signal',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    # MACD histogram
    hist_data = data['MACD'] - data['Signal']
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=hist_data,
            name='Histogram',
            marker=dict(
                color=np.where(hist_data >= 0, 'green', 'red'),
                opacity=0.5
            )
        ),
        row=2, col=1
    )
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    # Add RSI levels
    fig.add_trace(
        go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[70, 70],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[30, 30],
            mode='lines',
            line=dict(color='green', dash='dash'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig

def run_backtest(symbol, days=180):
    """Run a backtest on historical data"""
    try:
        # Add .NS suffix if not present (for NSE stocks)
        if not symbol.endswith(('.NS', '.BO')):
            symbol = f"{symbol}.NS"
        
        # Get historical data
        stock = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)  # Get days of data
        hist = stock.history(start=start_date, end=end_date)
        
        if len(hist) < 50:  # Not enough data
            return {"error": "Not enough historical data"}
        
        # Initialize variables
        positions = []
        capital = 100000  # Starting with 100k
        position_size = 0.1  # 10% of capital per trade
        in_position = False
        entry_price = 0
        
        # Calculate indicators for all data
        data = calculate_indicators(hist)
        
        # For each day after the first 50 days (need enough data for indicators)
        for i in range(50, len(data)):
            daily_data = data.iloc[:i+1]
            signal = generate_signals(daily_data).to_dict()
            current_price = daily_data['Close'].iloc[-1]
            date = daily_data.index[-1].strftime('%Y-%m-%d')
            
            # Trading logic
            if not in_position and signal['Recommendation'] in ['Buy', 'Strong Buy']:
                # Enter a position
                shares = int((capital * position_size) / current_price)
                entry_price = current_price
                in_position = True
                positions.append({
                    'date': date,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'value': shares * current_price,
                    'signal': signal['Recommendation']
                })
            
            elif in_position and signal['Recommendation'] in ['Sell', 'Strong Sell']:
                # Exit position
                last_pos = positions[-1]
                shares = last_pos['shares']
                positions.append({
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'value': shares * current_price,
                    'signal': signal['Recommendation'],
                    'profit': shares * (current_price - entry_price),
                    'profit_pct': (current_price / entry_price - 1) * 100
                })
                in_position = False
        
        # Calculate performance
        buy_trades = [p for p in positions if p['action'] == 'BUY']
        sell_trades = [p for p in positions if p['action'] == 'SELL']
        
        if len(buy_trades) == 0:
            return {
                "symbol": symbol,
                "trades": 0,
                "profitable_trades": 0,
                "profit_factor": 0,
                "total_profit": 0,
                "max_drawdown": 0,
                "positions": positions
            }
        
        trades = len(sell_trades)
        
        if trades == 0:
            return {
                "symbol": symbol,
                "trades": 0,
                "profitable_trades": 0,
                "profit_factor": 0,
                "total_profit": 0,
                "max_drawdown": 0,
                "positions": positions
            }
        
        profitable_trades = len([t for t in sell_trades if t.get('profit', 0) > 0])
        win_rate = profitable_trades / trades if trades > 0 else 0
        
        total_profit = sum([t.get('profit', 0) for t in sell_trades])
        profitable_amount = sum([t.get('profit', 0) for t in sell_trades if t.get('profit', 0) > 0])
        loss_amount = sum([abs(t.get('profit', 0)) for t in sell_trades if t.get('profit', 0) < 0])
        
        profit_factor = profitable_amount / loss_amount if loss_amount > 0 else float('inf')
        
        # Calculate max drawdown
        buy_hold_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        
        # Prepare performance chart
        daily_returns = hist['Close'].pct_change()
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        performance_fig = go.Figure()
        performance_fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns * 100,
                name='Buy and Hold',
                line=dict(color='blue')
            )
        )
        
        # Add trade markers
        for pos in positions:
            marker_color = 'green' if pos['action'] == 'BUY' else 'red'
            idx = pd.to_datetime(pos['date'])
            if idx in cumulative_returns.index:
                performance_fig.add_trace(
                    go.Scatter(
                        x=[idx],
                        y=[cumulative_returns.loc[idx] * 100],
                        mode='markers',
                        marker=dict(color=marker_color, size=10),
                        name=pos['action'],
                        text=f"{pos['action']} at {pos['price']}"
                    )
                )
        
        performance_fig.update_layout(
            title=f"{symbol} Backtest Performance",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        return {
            "symbol": symbol,
            "trades": trades,
            "profitable_trades": profitable_trades,
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "total_profit": round(total_profit, 2),
            "buy_hold_return": round(buy_hold_return, 2),
            "positions": positions,
            "chart": json.dumps(performance_fig, cls=plotly.utils.PlotlyJSONEncoder)
        }
    except Exception as e:
        print(f"Error running backtest for {symbol}: {e}")
        return {"error": str(e)}

def update_data():
    """Update stock recommendations"""
    global recommendations, last_update_time, backtest_results
    
    print("Updating stock data...")
    stocks = get_stock_list()
    new_recommendations = []
    
    for symbol in stocks:
        recommendation = analyze_stock(symbol)
        if recommendation:
            new_recommendations.append(recommendation)
            
            # Run backtest if not already done
            if symbol not in backtest_results:
                backtest = run_backtest(symbol)
                if 'error' not in backtest:
                    backtest_results[symbol] = backtest
    
    # Sort recommendations: Strong Buy, Buy, Hold, Sell, Strong Sell
    def rec_order(rec):
        order = {"Strong Buy": 0, "Buy": 1, "Hold": 2, "Sell": 3, "Strong Sell": 4}
        return order.get(rec['recommendation'], 5)
    
    new_recommendations.sort(key=rec_order)
    
    # Send Telegram notifications for Strong Buy and Strong Sell
    if recommendations:  # Skip on first run to avoid massive notifications
        for rec in new_recommendations:
            if rec['recommendation'] in ['Strong Buy', 'Strong Sell']:
                old_rec = next((r for r in recommendations if r['symbol'] == rec['symbol']), None)
                if not old_rec or old_rec['recommendation'] != rec['recommendation']:
                    message = f"<b>{rec['recommendation']}: {rec['company_name']} ({rec['symbol']})</b>\n"
                    message += f"Current Price: ₹{rec['current_price']}\n"
                    message += f"Target Price: ₹{rec['target_price']}\n"
                    message += f"RSI: {rec['rsi']}\n"
                    message += f"MACD: {rec['macd']}\n"
                    if rec['adx']:
                        message += f"ADX: {rec['adx']}\n"
                    message += f"\nView more at: https://robot-pdwz.onrender.com/"
                    
                    send_telegram_message(message)
                    send_to_channel(message)
    
    recommendations = new_recommendations
    last_update_time = datetime.now()
    print(f"Data updated at {last_update_time}")

def background_updater():
    """Background thread to update data periodically"""
    while True:
        update_data()
        time.sleep(UPDATE_INTERVAL)

@app.route('/')
def index():
    """Render the main page with stock recommendations"""
    return render_template('index.html', 
                          recommendations=recommendations, 
                          last_update=last_update_time)

@app.route('/backtest/<symbol>')
def backtest_view(symbol):
    """Render backtest results for a specific stock"""
    if symbol in backtest_results:
        result = backtest_results[symbol]
    else:
        result = run_backtest(symbol)
        if 'error' not in result:
            backtest_results[symbol] = result
    
    return render_template('backtest.html', result=result)

@app.route('/api/recommendations')
def api_recommendations():
    """API endpoint for recommendations"""
    return {"recommendations": recommendations, "last_update": last_update_time.isoformat() if last_update_time else None}

if __name__ == '__main__':
    # Start the background updater thread
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
