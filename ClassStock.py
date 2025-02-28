# ClassStock.py

import yfinance as yf  # Fetching stock and market data
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
import logging  # Logging for debugging and warnings
from datetime import datetime, timedelta  # Date manipulations

class Stock:
    """
    Class representing an individual stock in the portfolio.
    """
    def __init__(self, symbol, invested, value, risk_free_rate):
        self.symbol = symbol
        self.RF = risk_free_rate
        self.value = value              # Valor monetário atribuído (quantity * price)
        self.investment = invested      # Quanto foi investido originalmente
        self.data = None                # Historical data of the stock
        self.stock = None               # yfinance Ticker object
        self.simulated_prices = None    # Simulated future prices
        self.adjusted_beta = None       # Adjusted beta for stress testing
        self.adjusted_volatility = None # Adjusted volatility for stress testing
        self.adjusted_return = None     # Adjusted expected return for stress testing

        # Se tiveres um 'quantity' explicitamente:
        self.quantity = 0.0             # Calculado a partir de value / last_price (no main)

    def getTicker(self):
        if self.stock is None:
            self.stock = yf.Ticker(self.symbol)
        return self.stock

    def getInvestment(self):
        return self.investment

    def getAmount(self):
        return self.value

    def get_data(self, start_date=None, end_date=None):
        if self.data is None:
            if end_date is None:
                end_date = datetime.today()
            if start_date is None:
                start_date = end_date - timedelta(days=365)  # Last 365 days
            try:
                # Download historical data with daily intervals
                self.data = yf.download(self.symbol, start=start_date, end=end_date,
                                        interval='1d', progress=False)
                if self.data.empty:
                    logging.warning(f"No data retrieved for {self.symbol}.")
                    self.data = pd.DataFrame()
                else:
                    # Flatten columns if multi-level
                    if isinstance(self.data.columns, pd.MultiIndex):
                        self.data.columns = self.data.columns.get_level_values(0)
            except Exception as e:
                logging.error(f"Error fetching data for {self.symbol}: {e}")
                self.data = pd.DataFrame()
        return self.data

    def getInfo(self):
        try:
            return self.getTicker().info
        except Exception as e:
            logging.error(f"Error fetching info for {self.symbol}: {e}")
            return {}

    def getBeta(self):
        if self.adjusted_beta is not None:
            return self.adjusted_beta
        beta = self.getInfo().get('beta', np.nan)
        return beta

    def setAdjustedBeta(self, beta):
        self.adjusted_beta = beta

    def getClose(self):
        if self.data is None:
            self.get_data()
        if 'Adj Close' in self.data.columns:
            adj_close = self.data['Adj Close']
            if isinstance(adj_close, pd.DataFrame):
                adj_close = adj_close.iloc[:, 0]
            return adj_close
        else:
            logging.warning(f"Adjusted Close data not available for {self.symbol}.")
            return pd.Series(dtype=float)

    def getReturns(self):
        close_prices = self.getClose()
        if close_prices.empty:
            return pd.Series(dtype=float)
        return close_prices.pct_change()

    def getAdjustedReturns(self):
        """
        Returns adjusted returns based on modified expected returns and volatility (if set).
        Else, normal daily returns.
        """
        returns = self.getReturns().dropna()
        if returns.empty:
            return pd.Series(dtype=float)

        if (self.adjusted_return is not None) and (self.adjusted_volatility is not None):
            # Generate adjusted returns ~ N(mu/252, sigma/sqrt(252)) 
            adjusted = np.random.normal(
                loc=self.adjusted_return / 252,
                scale=self.adjusted_volatility / np.sqrt(252),
                size=len(returns)
            )
            return pd.Series(adjusted, index=returns.index)
        else:
            return returns

    def setAdjustedReturn(self, expected_return):
        self.adjusted_return = expected_return

    def setAdjustedVolatility(self, volatility):
        self.adjusted_volatility = volatility

    def getTotalReturn(self):
        close_prices = self.getClose()
        if len(close_prices) < 2:
            return np.nan
        total_return = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        return total_return

    def annualToDaily(self):
        return (1 + self.RF) ** (1 / 252) - 1

    def getExcessReturns(self):
        returns = self.getAdjustedReturns().dropna()
        if returns.empty:
            return pd.Series(dtype=float)
        excess_returns = returns - self.annualToDaily()
        return excess_returns

    def getSharpeRatio(self):
        er = self.getExcessReturns().dropna()
        std_dev = er.std()
        if std_dev == 0 or np.isnan(std_dev):
            return np.nan
        return er.mean() / std_dev * np.sqrt(252)

    def getSortinoRatio(self):
        er = self.getExcessReturns().dropna()
        downside = er[er < 0]
        ds_std = downside.std()
        if ds_std == 0 or np.isnan(ds_std):
            return np.nan
        return er.mean() / ds_std * np.sqrt(252)

    def monteCarloSimulation(self, num_simulations=1000, num_days=252):
        """
        Monte Carlo simulation using Geometric Brownian Motion for this stock.
        """
        returns = self.getAdjustedReturns()
        if returns.empty:
            logging.warning(f"No returns data for Monte Carlo simulation of {self.symbol}.")
            return pd.DataFrame()

        last_price = self.getClose().iloc[-1]
        mu = returns.mean()
        sigma = returns.std()
        dt = 1 / 252

        sim_array = np.zeros((num_days, num_simulations))

        for i in range(num_simulations):
            np.random.seed(i)
            eps = np.random.normal(0, 1, num_days)
            path = last_price * np.exp(
                np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * eps)
            )
            sim_array[:, i] = path

        last_date = self.getClose().index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=num_days)

        sim_df = pd.DataFrame(sim_array, index=future_dates,
                              columns=[f"Simulation_{i+1}" for i in range(num_simulations)])
        self.simulated_prices = sim_df
        return sim_df
