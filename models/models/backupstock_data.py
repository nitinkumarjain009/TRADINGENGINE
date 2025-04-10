"""
Stock data fetching and processing
"""
import pandas as pd
import yfinance as yf
import logging
import time
import csv
import os
from config import DATA_FETCH_PERIOD, DEFAULT_STOCKS, STOCK_LIST_CSV

logger = logging.getLogger(__name__)

class StockDataManager:
    def __init__(self):
        self.session = None
        self.symbols = self._load_stock_symbols()
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize a persistent session for API calls"""
        self.session = yf.Tickers("")
    
    def _load_stock_symbols(self):
        """Load stock symbols from CSV file"""
        try:
            if os.path.exists(STOCK_LIST_CSV):
                symbols = []
                with open(STOCK_LIST_CSV, 'r') as file:
                    csv_reader = csv.reader(file)
                    next(csv_reader, None)  # Skip header
                    for row in csv_reader:
                        if row and len(row) > 0:
                            symbol = row[0].strip()
                            if symbol:
                                symbols.append(symbol)
                
                if symbols:
                    logger.info(f"Loaded {len(symbols)} symbols from CSV")
                    return symbols
                else:
                    logger.warning(f"No valid symbols found in {STOCK_LIST_CSV}")
            else:
                logger.warning(f"Stock list CSV file not found: {STOCK_LIST_CSV}")
        except Exception as e:
            logger.error(f"Error loading stock symbols from CSV: {e}", exc_info=True)
        
        logger.info(f"Using default stock list with {len(DEFAULT_STOCKS)} symbols")
        return DEFAULT_STOCKS
    
    def fetch_stock_data(self, symbol, period=DATA_FETCH_PERIOD):
        """Fetch stock data for a single symbol"""
        if not symbol or not isinstance(symbol, str):
            logger.warning(f"Invalid symbol: {symbol}")
            return pd.DataFrame()
        
        try:
            logger.info(f"Fetching {period} data for {symbol}")
            start_time = time.time()
            
            data = yf.download(
                symbol, 
                period=period, 
                auto_adjust=True, 
                progress=False, 
                threads=True
            )
            
            end_time = time.time()
            logger.info(f"Fetched data for {symbol} in {end_time - start_time:.2f} seconds")
            
            if data.empty:
                logger.warning(f"No data returned for symbol: {symbol}")
                return pd.DataFrame()
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    
    def fetch_batch_data(self, symbols=None, period=DATA_FETCH_PERIOD):
        """Fetch data for multiple symbols in a batch"""
        if symbols is None:
            symbols = self.symbols
        
        if not symbols:
            logger.warning("No symbols provided for batch download")
            return {}
        
        try:
            logger.info(f"Batch downloading data for {len(symbols)} symbols")
            start_time = time.time()
            
            # Split into batches of 10 symbols to avoid API limitations
            results = {}
            batch_size = 10
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
                
                # Download data for the batch
                batch_data = yf.download(
                    batch, 
                    period=period, 
                    auto_adjust=True, 
                    progress=False, 
                    group_by='ticker'
                )
                
                # Split the data by symbol
                for symbol in batch:
                    # Handle both single symbol and multiple symbol return formats
                    if len(batch) == 1:
                        # If only one symbol, it doesn't use MultiIndex
                        results[symbol] = batch_data
                    else:
                        # For multiple symbols, we get a MultiIndex DataFrame
                        if symbol in batch_data.columns.levels[0]:
                            symbol_data = batch_data[symbol].copy()
                            results[symbol] = symbol_data
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            end_time = time.time()
            logger.info(f"Batch download completed in {end_time - start_time:.2f} seconds")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during batch download: {e}", exc_info=True)
            
            # Fall back to individual downloads if batch fails
            logger.info("Falling back to individual downloads")
            results = {}
            
            for symbol in symbols:
                try:
                    data = self.fetch_stock_data(symbol, period)
                    if not data.empty:
                        results[symbol] = data
                except Exception as e:
                    logger.error(f"Error fetching individual data for {symbol}: {e}")
            
            return results
    
    def extract_close_data(self, stock_data):
        """Extract Close price data from potentially complex DataFrame structure"""
        if stock_data.empty:
            return pd.DataFrame()
        
        try:
            # Check if we have MultiIndex columns (which happens with some yfinance returns)
            if isinstance(stock_data.columns, pd.MultiIndex):
                logger.debug(f"MultiIndex detected. Levels: {stock_data.columns.levels}")
                
                # Get Close data which is usually in the first level of MultiIndex
                if 'Close' in stock_data.columns.get_level_values(0):
                    close_data = stock_data['Close']
                    
                    # Handle both Series and DataFrame cases
                    if isinstance(close_data, pd.Series):
                        df = close_data.to_frame(name='Close')
                    else:
                        df = close_data.copy()
                        df.columns = ['Close']
                    
                    return df
                else:
                    logger.warning("'Close' not found in column levels")
                    return pd.DataFrame()
            
            # Standard DataFrame with direct columns
            elif 'Close' in stock_data.columns:
                df = stock_data[['Close']].copy()
                
                if 'Open' in stock_data.columns:
                    df['Open'] = stock_data['Open']
                if 'High' in stock_data.columns:
                    df['High'] = stock_data['High']
                if 'Low' in stock_data.columns:
                    df['Low'] = stock_data['Low']
                if 'Volume' in stock_data.columns:
                    df['Volume'] = stock_data['Volume']
                
                return df
            else:
                logger.warning("No 'Close' column found in DataFrame")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error extracting close data: {e}", exc_info=True)
            return pd.DataFrame()
