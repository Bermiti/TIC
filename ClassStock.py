import numpy as np
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

# git test

class Stock:
    """
    Represents an individual stock in the portfolio.
    """

    def __init__(self, symbol, invested, value, risk_free_rate):
        self.symbol = symbol
        self.RF = risk_free_rate
        self.value = value            # current monetary value = quantity * price
        self.investment = invested    # amount initially invested
        self.data = None
        self.stock = None
        self.simulated_prices = None
        self.adjusted_beta = None
        self.adjusted_volatility = None
        self.adjusted_return = None

        # Estimated number of shares based on "value / last price"
        self.quantity = 0.0

    def getTicker(self):
        """
        Returns the yfinance Ticker object, or creates it if it doesn't exist yet.
        """
        if self.stock is None:
            self.stock = yf.Ticker(self.symbol)
        return self.stock

    def getInvestment(self):
        return self.investment

    def getAmount(self):
        return self.value

    def get_data(self, start_date=None, end_date=None):
        """
        Downloads the price data via yfinance and stores it in self.data.
        """
        if self.data is None:
            if end_date is None:
                end_date = datetime.today()
            if start_date is None:
                start_date = end_date - timedelta(days=365)
            try:
                df = yf.download(
                    self.symbol,
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    progress=False,
                    auto_adjust=False  # Keeps the 'Adj Close' column
                )
                if df.empty:
                    logging.warning(f"No data returned for {self.symbol}.")
                    self.data = pd.DataFrame()
                else:
                    # In case there is a MultiIndex in columns, normalize it
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    self.data = df
            except Exception as e:
                logging.error(f"Error fetching data for {self.symbol}: {e}")
                self.data = pd.DataFrame()
        return self.data

    def getInfo(self):
        """
        Returns the info dictionary from the ticker in Yahoo Finance.
        """
        try:
            return self.getTicker().info
        except Exception as e:
            logging.error(f"Error fetching info for {self.symbol}: {e}")
            return {}

    def getBeta(self):
        """
        If an adjusted beta has been set, returns it. Otherwise,
        attempts to fetch it from the .info property in yfinance.
        """
        if self.adjusted_beta is not None:
            return self.adjusted_beta
        beta = self.getInfo().get('beta', np.nan)
        return beta

    def setAdjustedBeta(self, beta):
        self.adjusted_beta = beta

    def getClose(self):
        """
        Returns a Series of 'Adj Close', if available.
        """
        if self.data is None:
            self.get_data()
        if 'Adj Close' in self.data.columns:
            adj_close = self.data['Adj Close']
            if isinstance(adj_close, pd.DataFrame):
                # If for some reason it comes as multi-column, grab the first
                adj_close = adj_close.iloc[:, 0]
            return adj_close
        else:
            logging.warning(f"No 'Adj Close' for {self.symbol}.")
            return pd.Series(dtype=float)

    def getReturns(self):
        """
        Returns the daily returns (pct_change) based on 'Adj Close'.
        """
        c = self.getClose()
        if c.empty or len(c) < 2:
            return pd.Series(dtype=float)
        return c.pct_change()

    def setAdjustedReturn(self, expected_return):
        """
        Manually sets an (annual) expected return for custom simulations.
        """
        self.adjusted_return = expected_return

    def setAdjustedVolatility(self, volatility):
        """
        Manually sets an (annual) volatility for custom simulations.
        """
        self.adjusted_volatility = volatility

    def getAdjustedReturns(self):
        """
        If (adjusted_return, adjusted_volatility) are defined, it generates synthetic returns.
        Otherwise, uses the historical returns.
        """
        r = self.getReturns().dropna()
        if r.empty:
            return pd.Series(dtype=float)

        if (self.adjusted_return is not None) and (self.adjusted_volatility is not None):
            rng = np.random.default_rng(42)
            # convert annual to daily
            daily_mean = self.adjusted_return / 252
            daily_std = self.adjusted_volatility / np.sqrt(252)
            synthetic = rng.normal(loc=daily_mean, scale=daily_std, size=len(r))
            return pd.Series(synthetic, index=r.index)
        else:
            return r

    def getTotalReturn(self):
        """
        (LastPrice - FirstPrice) / FirstPrice over the loaded period.
        """
        c = self.getClose()
        if len(c) < 2:
            return np.nan
        return (c.iloc[-1] - c.iloc[0]) / c.iloc[0]

    def annualToDaily(self):
        """
        Converts annual interest rate to daily (approx for 252 trading days).
        """
        return (1 + self.RF) ** (1/252) - 1

    def getExcessReturns(self):
        """
        Daily return - (daily risk-free rate).
        """
        r = self.getAdjustedReturns().dropna()
        if r.empty:
            return pd.Series(dtype=float)
        return r - self.annualToDaily()

    def getSharpeRatio(self):
        """
        Annualized Sharpe Ratio for this stock, using adjusted returns.
        """
        er = self.getExcessReturns().dropna()
        std_dev = er.std()
        if std_dev == 0 or np.isnan(std_dev):
            return np.nan
        return er.mean() / std_dev * np.sqrt(252)

    def getSortinoRatio(self):
        """
        Annualized Sortino Ratio for this stock, using adjusted returns.
        """
        er = self.getExcessReturns().dropna()
        negative_part = er[er < 0]
        if negative_part.empty or negative_part.std() == 0:
            return np.nan
        return er.mean() / negative_part.std() * np.sqrt(252)

    def monteCarloSimulation(self, num_simulations=1000, num_days=252):
        """
        Price simulation (GBM) for the individual stock.
        """
        r = self.getAdjustedReturns()
        if r.empty:
            logging.warning(f"No returns for MC: {self.symbol}")
            return pd.DataFrame()

        c = self.getClose()
        if c.empty:
            logging.warning(f"No prices for MC: {self.symbol}")
            return pd.DataFrame()

        last_price = c.iloc[-1]
        mu = r.mean()
        sigma = r.std()
        dt = 1 / 252

        sim_array = np.zeros((num_days, num_simulations))
        for i in range(num_simulations):
            np.random.seed(i)
            eps = np.random.normal(0, 1, num_days)
            path = last_price * np.exp(
                np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps)
            )
            sim_array[:, i] = path

        last_date = c.index[-1] if len(c) else datetime.today()
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=num_days)

        sim_df = pd.DataFrame(
            sim_array,
            index=future_dates,
            columns=[f"Simulation_{i+1}" for i in range(num_simulations)]
        )
        self.simulated_prices = sim_df
        return sim_df
