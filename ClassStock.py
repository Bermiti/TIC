#ClassStock.py
import yfinance as yf
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class Stock:
    """
    Represents a single stock in the portfolio.
    """

    def __init__(self, symbol, risk_free_rate=0.0, investment=0.0, quantity=0.0):
        self.symbol = symbol
        self.RF = risk_free_rate
        # 'investment' can represent the total cost of all buy transactions
        self.investment = investment
        # 'value' will be derived from quantity * last market price
        self.value = 0.0
        # 'quantity' is the net shares held (buys - sells)
        self.quantity = quantity

        # Data (DataFrame from yfinance)
        self.data = None
        # Ticker object from yfinance
        self.stock = None

        # Adjusted overrides (optional for custom simulations)
        self.adjusted_return = None
        self.adjusted_volatility = None
        self.adjusted_beta = None

        # For storing any Monte Carlo results if needed
        self.simulated_prices = None

    def getTicker(self):
        if self.stock is None:
            self.stock = yf.Ticker(self.symbol)
        return self.stock

    def get_data(self, start_date=None, end_date=None):
        """
        Downloads price data via yfinance, stores in self.data.
        By default, if no dates are given, it fetches ~1 year of data.
        """
        if self.data is None:
            if end_date is None:
                # Provide a more realistic fallback if your system date is in the future:
                end_date = pd.Timestamp.today().floor('D')
                # e.g., end_date = min(pd.Timestamp.today(), pd.Timestamp("2024-12-31"))

            if start_date is None:
                start_date = end_date - timedelta(days=365)

            try:
                df = yf.download(
                    self.symbol,
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    progress=False,
                    auto_adjust=False
                )
                if df.empty:
                    logging.warning(f"No data returned for {self.symbol}.")
                    self.data = pd.DataFrame()
                else:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    self.data = df
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
        """
        If an adjusted beta has been manually set, return that.
        Otherwise attempt to fetch from yfinance 'info'.
        """
        if self.adjusted_beta is not None:
            return self.adjusted_beta
        beta = self.getInfo().get('beta', np.nan)
        return beta

    def setAdjustedBeta(self, beta):
        self.adjusted_beta = beta

    def getClose(self):
        """
        Returns the 'Adj Close' column as a Series.
        """
        if self.data is None:
            self.get_data()
        if 'Adj Close' in self.data.columns:
            adj_close = self.data['Adj Close']
            if isinstance(adj_close, pd.DataFrame):
                # If for some reason it comes as a multi-column DataFrame,
                # flatten to a Series
                adj_close = adj_close.iloc[:, 0]
            return adj_close
        else:
            logging.warning(f"No 'Adj Close' for {self.symbol}.")
            return pd.Series(dtype=float)

    def getPriceOnDate(self, date):
        """
        Fetches the 'Adj Close' price for this stock on the specific 'date'.
        If the exact date is not in the index, we might do a forward/back fill.
        """
        c = self.getClose()
        if c.empty:
            return np.nan

        # Ensure the index is a DatetimeIndex
        if not isinstance(c.index, pd.DatetimeIndex):
            c.index = pd.to_datetime(c.index)

        if date in c.index:
            return float(c.loc[date])
        else:
            # fallback: find the closest previous valid date
            idx = c.index.asof(date)
            if pd.isna(idx):
                # no earlier date => fallback to earliest price
                idx = c.index[0]
            return float(c.loc[idx])

    def getReturns(self):
        """
        Daily returns (percentage change) based on 'Adj Close'.
        """
        c = self.getClose()
        if c.empty or len(c) < 2:
            return pd.Series(dtype=float)
        return c.pct_change()

    def setAdjustedReturn(self, expected_return):
        self.adjusted_return = expected_return

    def setAdjustedVolatility(self, volatility):
        self.adjusted_volatility = volatility

    def getAdjustedReturns(self):
        """
        If (adjusted_return, adjusted_volatility) are set, generate synthetic returns;
        otherwise use historical returns.
        """
        r = self.getReturns().dropna()
        if r.empty:
            return pd.Series(dtype=float)

        if (self.adjusted_return is not None) and (self.adjusted_volatility is not None):
            rng = np.random.default_rng(42)
            daily_mean = self.adjusted_return / 252
            daily_std = self.adjusted_volatility / np.sqrt(252)
            synthetic = rng.normal(loc=daily_mean, scale=daily_std, size=len(r))
            return pd.Series(synthetic, index=r.index)
        else:
            return r

    def getTotalReturn(self):
        c = self.getClose()
        if len(c) < 2:
            return np.nan
        return (c.iloc[-1] - c.iloc[0]) / c.iloc[0]

    def annualToDaily(self):
        return (1 + self.RF)**(1/252) - 1

    def getExcessReturns(self):
        r = self.getAdjustedReturns().dropna()
        if r.empty:
            return pd.Series(dtype=float)
        return r - self.annualToDaily()

    def getSharpeRatio(self):
        er = self.getExcessReturns().dropna()
        std_dev = er.std()
        if std_dev == 0 or np.isnan(std_dev):
            return np.nan
        return er.mean() / std_dev * np.sqrt(252)

    def getSortinoRatio(self):
        er = self.getExcessReturns().dropna()
        neg = er[er < 0]
        if neg.empty or neg.std() == 0:
            return np.nan
        return er.mean() / neg.std() * np.sqrt(252)

    def monteCarloSimulation(self, num_simulations=1000, num_days=252):
        """
        GBM-based price simulation for the stock using its historical mean and std dev of returns.
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
