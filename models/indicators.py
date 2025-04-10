"""
Technical indicators calculation
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_sma(series, window):
    """Calculate Simple Moving Average"""
    return series.rolling(window=window).mean()

def calculate_ema(series, span):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Handle potential division by zero
    avg_loss = avg_loss.replace(0, np.finfo(float).eps)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast_span=12, slow_span=26, signal_span=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(series, fast_span)
    ema_slow = calculate_ema(series, slow_span)
    macd = ema_fast - ema_slow
    signal = calculate_ema(macd, signal_span)
    return macd, signal

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(series, window)
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_all_indicators(df):
    """Calculate all technical indicators for a DataFrame"""
    if df.empty or 'Close' not in df.columns:
        logger.warning("Empty DataFrame or missing 'Close' column")
        return pd.DataFrame()
    
    try:
        # Create a copy to avoid modifying the original
        result = df.copy()
        close = result['Close']
        
        # Simple Moving Averages
        result['SMA20'] = calculate_sma(close, 20)
        result['SMA50'] = calculate_sma(close, 50)
        result['SMA200'] = calculate_sma(close, 200)
        
        # Exponential Moving Averages
        result['EMA12'] = calculate_ema(close, 12)
        result['EMA26'] = calculate_ema(close, 26)
        
        # MACD
        result['MACD'], result['Signal_Line'] = calculate_macd(close)
        result['MACD_Histogram'] = result['MACD'] - result['Signal_Line']
        
        # RSI
        result['RSI'] = calculate_rsi(close)
        
        # Bollinger Bands
        result['BB_Upper'], result['BB_Middle'], result['BB_Lower'] = calculate_bollinger_bands(close)
        
        # Additional indicators
        # ADX - Average Directional Index (simplified version)
        high = df['High'] if 'High' in df.columns else close
        low = df['Low'] if 'Low' in df.columns else close
        
        # True Range
        tr1 = high - low
        if 'Close' in df.columns and len(df) > 1:
            prev_close = df['Close'].shift(1)
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result['ATR14'] = true_range.rolling(window=14).mean()
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        return pd.DataFrame()

def generate_recommendations(symbol, df_with_indicators):
    """Generate trading recommendations based on indicators"""
    if df_with_indicators.empty or len(df_with_indicators) < 50:
        logger.warning(f"Insufficient data for {symbol} to generate recommendations")
        return None
    
    try:
        # Get the latest values
        latest = df_with_indicators.iloc[-1]
        prev = df_with_indicators.iloc[-2]
        
        close_price = latest['Close']
        signal = "HOLD"  # Default
        target_price = None
        stop_loss = None
        strength = 0  # Signal strength (0-5)
        reasons = []
        
        # 1. MACD Crossover (strong signal)
        macd_current = latest['MACD']
        signal_current = latest['Signal_Line']
        macd_prev = prev['MACD']
        signal_prev = prev['Signal_Line']
        
        if macd_prev < signal_prev and macd_current > signal_current:
            signal = "BUY"
            reasons.append("MACD bullish crossover")
            strength += 2
        elif macd_prev > signal_prev and macd_current < signal_current:
            signal = "SELL"
            reasons.append("MACD bearish crossover")
            strength += 2
            
        # 2. RSI Conditions
        rsi = latest['RSI']
        if rsi < 30:
            if signal != "BUY":  # Don't override MACD signal
                signal = "BUY"
                reasons.append("RSI oversold")
                strength += 1
            else:
                reasons.append("RSI confirms oversold")
                strength += 1
        elif rsi > 70:
            if signal != "SELL":  # Don't override MACD signal
                signal = "SELL"
                reasons.append("RSI overbought")
                strength += 1
            else:
                reasons.append("RSI confirms overbought")
                strength += 1
        
        # 3. Moving Average Crossover
        if latest['SMA20'] > latest['SMA50'] and prev['SMA20'] <= prev['SMA50']:
            if signal != "SELL":  # Don't contradict previous signals
                signal = "BUY"
                reasons.append("SMA20 crossed above SMA50")
                strength += 1
            else:
                strength -= 1  # Contradicting signal reduces strength
        elif latest['SMA20'] < latest['SMA50'] and prev['SMA20'] >= prev['SMA50']:
            if signal != "BUY":  # Don't contradict previous signals
                signal = "SELL"
                reasons.append("SMA20 crossed below SMA50")
                strength += 1
            else:
                strength -= 1  # Contradicting signal reduces strength
        
        # 4. Bollinger Bands
        if 'BB_Upper' in latest and 'BB_Lower' in latest:
            if close_price > latest['BB_Upper']:
                if signal != "BUY":  # Potential reversal or continuation of uptrend
                    reasons.append("Price above upper Bollinger Band")
                    strength += 1 if signal == "SELL" else 0
                else:
                    strength -= 1  # Contradicting signal
            elif close_price < latest['BB_Lower']:
                if signal != "SELL":  # Potential reversal or continuation of downtrend
                    reasons.append("Price below lower Bollinger Band")
                    strength += 1 if signal == "BUY" else 0
                else:
                    strength -= 1  # Contradicting signal
        
        # Set target and stop loss based on ATR if available
        if 'ATR14' in latest and not pd.isna(latest['ATR14']):
            atr = latest['ATR14']
            if signal == "BUY":
                target_price = close_price + (atr * 2)  # 2x ATR for target
                stop_loss = close_price - (atr * 1)  # 1x ATR for stop loss
            elif signal == "SELL":
                target_price = close_price - (atr * 2)  # 2x ATR for target
                stop_loss = close_price + (atr * 1)  # 1x ATR for stop loss
        else:
            # Fallback if no ATR available
            if signal == "BUY":
                target_price = close_price * 1.05  # 5% target
                stop_loss = close_price * 0.97  # 3% stop loss
            elif signal == "SELL":
                target_price = close_price * 0.95  # 5% target
                stop_loss = close_price * 1.03  # 3% stop loss
        
        # If signals are contradictory or weak, default to HOLD
        if strength < 2:
            if len(reasons) == 0 or (signal != "HOLD" and strength <= 0):
                signal = "HOLD"
                reasons = ["No clear signals"]
                target_price = None
                stop_loss = None
        
        return {
            'symbol': symbol,
            'signal': signal,
            'price': close_price,
            'target': target_price,
            'stop_loss': stop_loss,
            'strength': strength,
            'reason': "; ".join(reasons),
            'timestamp': pd.Timestamp.now(tz='UTC')
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendations for {symbol}: {e}", exc_info=True)
        return None
