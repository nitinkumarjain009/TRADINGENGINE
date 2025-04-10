"""
Paper trading functionality
"""
import pandas as pd
import logging
import json
import os
from datetime import datetime
from config import INITIAL_CASH

logger = logging.getLogger(__name__)

class PaperTradingPortfolio:
    def __init__(self, initial_cash=INITIAL_CASH, portfolio_file='portfolio.json'):
        self.portfolio_file = portfolio_file
        self.portfolio = self.load_portfolio()
        if not self.portfolio:
            self.portfolio = {
                'cash': initial_cash,
                'holdings': {},
                'trades': [],
                'performance': []
            }
            self.save_portfolio()
    
    def load_portfolio(self):
        """Load portfolio from file"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as file:
                    portfolio = json.load(file)
                logger.info("Portfolio loaded successfully")
                return portfolio
            else:
                logger.info("No existing portfolio file found, creating new portfolio")
                return None
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}", exc_info=True)
            return None
    
    def save_portfolio(self):
        """Save portfolio to file"""
        try:
            with open(self.portfolio_file, 'w') as file:
                json.dump(self.portfolio, file, indent=2, default=str)
            logger.info("Portfolio saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}", exc_info=True)
            return False
    
    def execute_trade(self, symbol, action, price, shares=None, amount=None):
        """Execute a paper trade"""
        if price <= 0:
            logger.warning(f"Invalid price: {price}")
            return False
        
        timestamp = datetime.now().isoformat()
        action = action.upper()
        
        try:
            # Buy logic
            if action == 'BUY':
                # Determine shares to buy
                if shares is None and amount is not None:
                    shares = int(amount / price)
                elif shares is None:
                    shares = int(self.portfolio['cash'] * 0.1 / price)  # Default to 10% of cash
                
                # Calculate total cost
                cost = shares * price
                
                # Check if we have enough cash
                if cost > self.portfolio['cash']:
                    logger.warning(f"Insufficient funds to buy {shares} shares of {symbol} at {price}")
                    return False
                
                # Execute the buy
                self.portfolio['cash'] -= cost
                
                if symbol in self.portfolio['holdings']:
                    # Update existing position
                    current_shares = self.portfolio['holdings'][symbol]['shares']
                    current_cost = self.portfolio['holdings'][symbol]['avg_price'] * current_shares
                    new_total_shares = current_shares + shares
                    new_avg_price = (current_cost + cost) / new_total_shares
                    
                    self.portfolio['holdings'][symbol] = {
                        'shares': new_total_shares,
                        'avg_price': new_avg_price
                    }
                else:
                    # Create new position
                    self.portfolio['holdings'][symbol] = {
                        'shares': shares,
                        'avg_price': price
                    }
                
                # Record the trade
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': cost,
                    'balance': self.portfolio['cash']
                }
            
            # Sell logic
            elif action == 'SELL':
                # Check if we own the stock
                if symbol not in self.portfolio['holdings'] or self.portfolio['holdings'][symbol]['shares'] <= 0:
                    logger.warning(f"No shares of {symbol} to sell")
                    return False
                
                available_shares = self.portfolio['holdings'][symbol]['shares']
                
                # Determine shares to sell
                if shares is None:
                    shares = available_shares  # Default to selling all shares
                elif shares > available_shares:
                    logger.warning(f"Trying to sell {shares} but only have {available_shares} shares of {symbol}")
                    shares = available_shares
                
                # Calculate proceeds
                proceeds = shares * price
                
                # Execute the sell
                self.portfolio['cash'] += proceeds
                self.portfolio['holdings'][symbol]['shares'] -= shares
                
                # Remove the holding if no shares left
                if self.portfolio['holdings'][symbol]['shares'] <= 0:
                    del self.portfolio['holdings'][symbol]
                
                # Record the trade
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': proceeds,
                    'balance': self.portfolio['cash']
                }
            
            else:
                logger.warning(f"Invalid trade action: {action}")
                return False
            
            # Add the trade to history
            self.portfolio['trades'].append(trade)
            
            # Save the portfolio
            self.save_portfolio()
            
            logger.info(f"Trade executed: {action} {shares} shares of {symbol} at {price}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return False
    
    def update_portfolio_value(self, current_prices):
        """Update the portfolio value based on current prices"""
        if not current_prices:
            logger.warning("No current prices provided to update portfolio value")
            return False
        
        try:
            total_value = self.portfolio['cash']
            holdings_value = 0
            
            for symbol, holding in list(self.portfolio['holdings'].items()):
                if symbol in current_prices and holding['shares'] > 0:
                    current_price = current_prices[symbol]
                    value = holding['shares'] * current_price
                    holdings_value += value
                    
                    # Update holding with current price and value
                    self.portfolio['holdings'][symbol]['current_price'] = current_price
                    self.portfolio['holdings'][symbol]['current_value'] = value
                    
                    # Calculate P&L
                    cost_basis = holding['shares'] * holding['avg_price']
                    self.portfolio['holdings'][symbol]['pnl'] = value - cost_basis
                    self.portfolio['holdings'][symbol]['pnl_percent'] = ((value / cost_basis) - 1) * 100 if cost_basis > 0 else 0
            
            # Calculate total portfolio value
            total_value += holdings_value
            
            # Record performance point
            timestamp = datetime.now().isoformat()
            performance_point = {
                'timestamp': timestamp,
                'cash': self.portfolio['cash'],
                'holdings_value': holdings_value,
                'total_value': total_value
            }
            
            # Limit performance history to last 100 points
            self.portfolio['performance'].append(performance_point)
            if len(self.portfolio['performance']) > 100:
                self.portfolio['performance'] = self.portfolio['performance'][-100:]
            
            # Save the updated portfolio
            self.save_portfolio()
            
            logger.info(f"Portfolio value updated: Total: {total_value}, Holdings: {holdings_value}, Cash: {self.portfolio['cash']}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}", exc_info=True)
            return False
    
    def process_recommendations(self, recommendations, current_prices):
        """Process trading recommendations and execute trades"""
        if not recommendations or not current_prices:
            logger.warning("No recommendations or prices to process")
            return [], []
        
        executed_trades = []
        skipped_recommendations = []
        
        try:
            for rec in recommendations:
                symbol = rec['symbol']
                signal = rec['signal']
                price = current_prices.get(symbol, rec.get('price'))
                
                if not price:
                    logger.warning(f"No price available for {symbol}, skipping recommendation")
                    skipped_recommendations.append({
                        'symbol': symbol,
                        'signal': signal,
                        'reason': 'No price available'
                    })
                    continue
                
                # Execute the trade based on the signal
                if signal == 'BUY':
                    # Buy with 10% of available cash or 10 shares, whichever is less
                    max_shares = int(self.portfolio['cash'] * 0.1 / price)
                    shares_to_buy = min(10, max_shares)
                    
                    if shares_to_buy <= 0:
                        logger.warning(f"Not enough cash to buy {symbol}")
                        skipped_recommendations.append({
                            'symbol': symbol,
                            'signal': signal,
                            'reason': 'Insufficient funds'
                        })
                        continue
                    
                    if self.execute_trade(symbol, 'BUY', price, shares=shares_to_buy):
                        executed_trades.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': price,
                            'value': shares_to_buy * price,
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        skipped_recommendations.append({
                            'symbol': symbol,
                            'signal': signal,
                            'reason': 'Trade execution failed'
                        })
                
                elif signal == 'SELL':
                    # Check if we own the stock
                    if symbol in self.portfolio['holdings'] and self.portfolio['holdings'][symbol]['shares'] > 0:
                        shares_to_sell = self.portfolio['holdings'][symbol]['shares']
                        
                        if self.execute_trade(symbol, 'SELL', price, shares=shares_to_sell):
                            executed_trades.append({
                                'symbol': symbol,
                                'action': 'SELL',
                                'shares': shares_to_sell,
                                'price': price,
                                'value': shares_to_sell * price,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            skipped_recommendations.append({
                                'symbol': symbol,
                                'signal': signal,
                                'reason': 'Trade execution failed'
                            })
                    else:
                        skipped_recommendations.append({
                            'symbol': symbol,
                            'signal': signal,
                            'reason': 'Stock not in portfolio'
                        })
            
            # Update portfolio value with latest prices
            self.update_portfolio_value(current_prices)
            
            return executed_trades, skipped_recommendations
        
        except Exception as e:
            logger.error(f"Error processing recommendations: {e}", exc_info=True)
            return executed_trades, skipped_recommendations
