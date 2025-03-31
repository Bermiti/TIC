# ClassPortfolio.py

import numpy as np
import pandas as pd
import logging
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from ClassStock import Stock

class Portfolio:
    """
    Manages a collection of Stock objects and cash balance, 
    computes risk metrics, and runs Monte Carlo simulations.
    """

    def __init__(self, stocks, risk_free_rate, initial_cash):
        """
        :param stocks: dict of symbol -> Stock object
        :param risk_free_rate: float, annual risk-free rate
        :param initial_cash: float, total starting cash
        """
        self.stocks = stocks
        self.RF = risk_free_rate
        self.cash_balance = initial_cash  # uninvested cash

        # Attempt to compute weights based on each Stock's current value.
        self.weights = self.getWeights()

        # Caches
        self.portfolio_returns = None
        self.portfolio_excess_returns = None
        self.market_close = None
        self.simulated_portfolio_values = None
        self.simulated_metrics = None

        # Store dynamic TS here if you use getDynamicTimeSeries()
        self._dynamic_ts = pd.DataFrame()

    def getWeights(self):
        """
        Returns the fraction of each stock's value in the total (stock) portion
        of the portfolio, ignoring uninvested cash.
        """
        total_val = sum(st.value for st in self.stocks.values())
        if total_val == 0:
            logging.warning("Portfolio has total (stock) value == 0; cannot compute weights.")
            return {}
        w = {}
        for sym, st in self.stocks.items():
            w[sym] = st.value / total_val
        return w

    def updateWeights(self):
        """
        Recomputes portfolio weights (call after buy/sell).
        """
        self.weights = self.getWeights()

    def update_stock(self, symbol, quantity, price, increase=True):
        """
        Adjusts the stock's position (buy or sell), updates cash_balance, 
        and recalculates the stock 'value' based on the last known or fallback price.
        """
        if symbol not in self.stocks and increase is False:
            raise ValueError(f"Cannot sell {symbol} that is not in the portfolio.")

        if symbol not in self.stocks:
            self.stocks[symbol] = Stock(symbol, risk_free_rate=self.RF, 
                                        investment=0.0, quantity=0.0)

        st = self.stocks[symbol]
        trade_cost = quantity * price

        if increase:
            # Buy
            if self.cash_balance < trade_cost:
                raise ValueError("Insufficient cash to buy.")
            self.cash_balance -= trade_cost
            st.investment += trade_cost
            st.quantity += quantity
        else:
            # Sell
            if st.quantity < quantity:
                raise ValueError("Not enough quantity to sell.")
            self.cash_balance += trade_cost
            st.quantity -= quantity

        # Update 'value' based on the last or fallback price
        c = st.getClose()
        if not c.empty:
            st.value = st.quantity * c.iloc[-1]
        else:
            st.value = st.quantity * price

        self.updateWeights()

    def has_stock(self, symbol, quantity):
        """
        Checks if we hold 'quantity' or more shares of 'symbol'.
        """
        if symbol not in self.stocks:
            return False
        return self.stocks[symbol].quantity >= quantity

    def getAdjClosePrices(self):
        """
        Returns a DataFrame: columns = stock symbols; rows = daily 'Adj Close' prices.
        """
        df = pd.DataFrame()
        for st in self.stocks.values():
            c = st.getClose()
            if not c.empty:
                df[st.symbol] = c
        return df

    def getPortfolioAdjustedClosePrices(self):
        """
        Weighted sum of 'Adj Close' if we had a constant buy-and-hold 
        portfolio (for reference).
        """
        adj_close = self.getAdjClosePrices()
        if adj_close.empty:
            return pd.Series(dtype=float)
        w_ = pd.Series(self.weights)
        return adj_close.mul(w_, axis=1).sum(axis=1)

    def getReturns(self):
        """
        Daily returns for each symbol (pct_change), plus 'Portfolio' column.
        """
        prices = self.getAdjClosePrices()
        if prices.empty:
            return pd.DataFrame()

        rets = prices.pct_change()
        rets["Portfolio"] = self.getPortfolioReturns()
        return rets

    def getAdjustedReturns(self):
        """
        Uses each stock's getAdjustedReturns() (either historical or synthetic)
        then constructs a weighted portfolio return series (ignoring uninvested cash).
        """
        df = pd.DataFrame()
        for st in self.stocks.values():
            r = st.getAdjustedReturns().dropna()
            if not r.empty:
                df[st.symbol] = r

        if df.empty:
            return pd.Series(dtype=float)

        # Only keep weights for symbols that appear in df
        valid_w = {
            sym: self.weights[sym] 
            for sym in df.columns if sym in self.weights
        }
        if not valid_w:
            return pd.Series(dtype=float)

        w_s = pd.Series(valid_w)
        portfolio_r = df.dot(w_s)
        return portfolio_r

    def getPortfolioReturns(self):
        """
        Cached daily returns of the stock portion (weighted sum).
        """
        if self.portfolio_returns is not None:
            return self.portfolio_returns
        series = self.getAdjustedReturns().dropna()
        self.portfolio_returns = series
        return series

    def getPortfolioExcessReturns(self):
        """
        Portfolio returns minus the daily risk-free rate.
        """
        if self.portfolio_excess_returns is not None:
            return self.portfolio_excess_returns
        pr = self.getPortfolioReturns().dropna()
        if pr.empty:
            self.portfolio_excess_returns = pd.Series(dtype=float)
            return self.portfolio_excess_returns

        daily_rf = (1 + self.RF)**(1/252) - 1
        self.portfolio_excess_returns = pr - daily_rf
        return self.portfolio_excess_returns

    def getPortfolioSharpe(self):
        """
        Annualized Sharpe ratio (mean of daily excess returns / std dev).
        """
        er = self.getPortfolioExcessReturns().dropna()
        if er.empty:
            return np.nan
        std_dev = er.std()
        if std_dev == 0 or np.isnan(std_dev):
            return np.nan
        return (er.mean() / std_dev) * np.sqrt(252)

    def getPortfolioSortino(self):
        """
        Annualized Sortino ratio = mean of daily excess returns / std of negative returns.
        """
        er = self.getPortfolioExcessReturns().dropna()
        if er.empty:
            return np.nan
        neg = er[er < 0]
        if neg.empty or neg.std() == 0:
            return np.nan
        return (er.mean() / neg.std()) * np.sqrt(252)

    def getPortfolioStdDev(self):
        """
        Annualized standard deviation of the stock portion's daily returns.
        """
        r = self.getPortfolioReturns().dropna()
        if r.empty:
            return np.nan
        return r.std() * np.sqrt(252)

    def getMaxDrawdown(self):
        """
        Max drawdown in the stock portion's daily returns (cumulative).
        """
        r = self.getPortfolioReturns().dropna()
        if r.empty:
            return np.nan
        cum = (1 + r).cumprod()
        peak = cum.expanding().max()
        dd = (cum / peak) - 1
        return dd.min()

    def getVaR(self, cl=0.05):
        """
        Value-at-Risk at the specified confidence level (e.g. 0.05=5%).
        """
        r = self.getPortfolioReturns().dropna()
        if r.empty:
            return np.nan
        return np.percentile(r, cl * 100)

    def getCVaR(self, cl=0.05):
        """
        Conditional VaR (aka Expected Shortfall) at e.g. 5% tail.
        """
        r = self.getPortfolioReturns().dropna()
        if r.empty:
            return np.nan
        var_ = self.getVaR(cl)
        return r[r <= var_].mean()

    def getPortfolioCovarianceMatrix(self):
        """
        Covariance matrix of daily returns for each stock in the portfolio.
        """
        df = pd.DataFrame()
        for st in self.stocks.values():
            ret = st.getAdjustedReturns().dropna()
            if not ret.empty:
                df[st.symbol] = ret
        if df.empty:
            return pd.DataFrame()
        df = df.dropna(how='all')
        return df.cov()

    def getCorrelationMatrix(self):
        """
        Correlation matrix of daily returns for each stock in the portfolio.
        """
        df = pd.DataFrame()
        for st in self.stocks.values():
            ret = st.getAdjustedReturns().dropna()
            if not ret.empty:
                df[st.symbol] = ret
        if df.empty:
            return pd.DataFrame()
        df = df.dropna(how='all')
        return df.corr()

    def getMarginalContributionToRisk(self):
        """
        MCTR for each stock: partial derivative of portfolio volatility 
        w.r.t. that stock's weight.
        """
        cov = self.getPortfolioCovarianceMatrix()
        if cov.empty:
            return pd.Series(dtype=float)
        w = np.array([self.weights.get(c, 0) for c in cov.columns])
        port_vol_daily = self.getPortfolioStdDev() / np.sqrt(252)
        if port_vol_daily == 0 or np.isnan(port_vol_daily):
            return pd.Series(dtype=float)
        mctr_ = cov.dot(w) / port_vol_daily
        return pd.Series(mctr_, index=cov.columns)

    def getComponentContributionToRisk(self):
        """
        CCTR: each stock's weight * that stock's MCTR, i.e. fraction of total risk.
        """
        mctr = self.getMarginalContributionToRisk()
        w_s = pd.Series(self.weights)
        return w_s[mctr.index] * mctr

    def simulatePortfolioWithoutAsset(self, symbol):
        """
        Hypothetical scenario removing 'symbol' from the portfolio 
        to see how volatility changes.
        """
        if symbol not in self.stocks:
            return np.nan
        new_stocks = {s: obj for s, obj in self.stocks.items() if s != symbol}
        if not new_stocks:
            return np.nan
        # Build a test portfolio ignoring that asset:
        from ClassPortfolio import Portfolio
        test_p = Portfolio(new_stocks, self.RF, self.cash_balance)
        return test_p.getPortfolioStdDev()

    def getAssetImpactOnVolatility(self, symbol):
        """
        Return how removing 'symbol' changes the portfolio's std dev.
        """
        orig = self.getPortfolioStdDev()
        new_ = self.simulatePortfolioWithoutAsset(symbol)
        if np.isnan(new_):
            return np.nan
        return orig - new_

    def getMarketData(self, symbol='^GSPC'):
        """
        Download market index data (e.g. S&P500) to estimate portfolio Beta.
        """
        if getattr(self, 'market_close', None) is not None and not self.market_close.empty:
            return self.market_close
        end_ = pd.Timestamp.today().floor('D')
        start_ = end_ - timedelta(days=365)
        try:
            df = yf.download(symbol, start=start_, end=end_, interval='1d',
                             progress=False, auto_adjust=False)
            if df.empty:
                self.market_close = pd.Series(dtype=float)
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                self.market_close = df['Adj Close']
        except Exception as e:
            logging.error(f"Error fetching market data {symbol}: {e}")
            self.market_close = pd.Series(dtype=float)
        return self.market_close

    def getMarketReturns(self):
        """
        Daily returns of the market index (e.g. S&P500).
        """
        m = self.getMarketData()
        if m.empty:
            return pd.Series(dtype=float)
        return m.pct_change()

    def getPortfolioBeta(self):
        """
        Beta vs. the market, via covariance/variance ratio of daily returns.
        """
        pr = self.getPortfolioReturns().dropna()
        mr_full = self.getMarketReturns()
        if pr.empty or mr_full.empty:
            return np.nan
        mr = mr_full.reindex(pr.index)
        df = pd.DataFrame({"p": pr, "m": mr}).dropna()
        if df.empty:
            return np.nan
        c = np.cov(df["p"], df["m"])
        pm = c[0,1]
        mm = c[1,1]
        if mm == 0:
            return np.nan
        return pm / mm

    def getTreynorRatio(self):
        """
        (Annualized portfolio return - RF) / Beta
        """
        b = self.getPortfolioBeta()
        if b == 0 or np.isnan(b):
            return np.nan
        ann_ret = self.getPortfolioReturns().mean() * 252
        return (ann_ret - self.RF) / b

    def getDetailedStockData(self):
        """
        Returns a DataFrame with stock-level stats (Beta, Sharpe, etc.),
        plus 'danger' flags if certain thresholds are exceeded.
        """
        rows = []
        mctr = self.getMarginalContributionToRisk()
        cctr = self.getComponentContributionToRisk()

        for st in tqdm(self.stocks.values(), desc="Processing stocks"):
            sym = st.symbol
            beta = st.getBeta()
            sharpe_ = st.getSharpeRatio()
            sortino_ = st.getSortinoRatio()
            impact = self.getAssetImpactOnVolatility(sym)
            mc = mctr.get(sym, np.nan)
            cc = cctr.get(sym, np.nan)

            c = st.getClose()
            if c.empty:
                cur_price = np.nan
                hi52 = np.nan
                lo52 = np.nan
            else:
                cur_price = c.iloc[-1]
                hi52 = c.max()
                lo52 = c.min()

            total_ret = st.getTotalReturn()

            # Danger flags
            beta_danger = (abs(beta) > 1.5) if not np.isnan(beta) else False
            sharpe_danger = (sharpe_ < 1) if not np.isnan(sharpe_) else False
            sortino_danger = (sortino_ < 1) if not np.isnan(sortino_) else False

            rows.append({
                "Stock": sym,
                "Weight %": round(self.weights.get(sym,0)*100,2),
                "Value $": round(st.value,2),
                "Current Price": round(cur_price,2) if not np.isnan(cur_price) else "N/A",
                "52-Week High": round(hi52,2) if not np.isnan(hi52) else "N/A",
                "52-Week Low": round(lo52,2) if not np.isnan(lo52) else "N/A",
                "Total Return %": round(total_ret*100,2) if not np.isnan(total_ret) else "N/A",
                "Beta": round(beta,4) if not np.isnan(beta) else "N/A",
                "Beta Danger": "Yes" if beta_danger else "No",
                "Sharpe": round(sharpe_,4) if not np.isnan(sharpe_) else "N/A",
                "Sharpe Danger": "Yes" if sharpe_danger else "No",
                "Sortino": round(sortino_,4) if not np.isnan(sortino_) else "N/A",
                "Sortino Danger": "Yes" if sortino_danger else "No",
                "Volatility Impact": round(impact,4) if not np.isnan(impact) else "N/A",
                "Marginal CTR": round(mc,6) if not np.isnan(mc) else "N/A",
                "Component CTR": round(cc,6) if not np.isnan(cc) else "N/A"
            })

        return pd.DataFrame(rows)

    def Summary(self):
        """
        Dictionary summarizing portfolio-level stats + 'danger' flags.

        Now includes self.cash_balance in the final "Portfolio Value."
        """
        # 1) Sum all stocks
        stock_value = sum(st.value for st in self.stocks.values())
        # 2) Add uninvested cash
        total_value = stock_value + self.cash_balance

        stdv = self.getPortfolioStdDev()
        sh = self.getPortfolioSharpe()
        so = self.getPortfolioSortino()
        be = self.getPortfolioBeta()
        tr = self.getTreynorRatio()
        dd = self.getMaxDrawdown()
        var95 = self.getVaR(0.05)
        cvar95 = self.getCVaR(0.05)

        sharpe_danger = (sh < 1) if not np.isnan(sh) else False
        sortino_danger = (so < 1) if not np.isnan(so) else False
        beta_danger = (abs(be) > 1.5) if not np.isnan(be) else False
        dd_danger = (dd < -0.2) if not np.isnan(dd) else False
        var_danger = (var95 < -0.05) if not np.isnan(var95) else False
        cvar_danger = (cvar95 < -0.05) if not np.isnan(cvar95) else False

        return {
            "Portfolio Value": round(total_value, 2),  # <== includes cash now
            "Portfolio StdDev": round(stdv, 4) if not np.isnan(stdv) else "N/A",
            "Portfolio Sharpe": round(sh, 4) if not np.isnan(sh) else "N/A",
            "Sharpe Danger": "Yes" if sharpe_danger else "No",
            "Portfolio Sortino": round(so, 4) if not np.isnan(so) else "N/A",
            "Sortino Danger": "Yes" if sortino_danger else "No",
            "Portfolio Beta": round(be, 4) if not np.isnan(be) else "N/A",
            "Beta Danger": "Yes" if beta_danger else "No",
            "Portfolio Treynor": round(tr, 4) if not np.isnan(tr) else "N/A",
            "Max Drawdown": round(dd, 4) if not np.isnan(dd) else "N/A",
            "Max Drawdown Danger": "Yes" if dd_danger else "No",
            "VaR 95%": round(var95, 4) if not np.isnan(var95) else "N/A",
            "VaR Danger": "Yes" if var_danger else "No",
            "CVaR 95%": round(cvar95, 4) if not np.isnan(cvar95) else "N/A",
            "CVaR Danger": "Yes" if cvar_danger else "No"
        }

    def getDynamicTimeSeries(self, transaction_log=None, by_stock=False):
        """
        Rebuilds the portfolio's daily value from the transaction history,
        applying transactions as-of their date. This lets weekend/holiday
        buys/sells 'take effect' on the next trading day.
        """
        if transaction_log is None or transaction_log.empty:
            logging.warning("No transaction log for dynamic series.")
            return pd.DataFrame()

        prices_df = self.getAdjClosePrices()
        if prices_df.empty:
            logging.warning("No prices to generate dynamic time series.")
            return pd.DataFrame()

        # Sort the transaction log by date
        transaction_log = transaction_log.sort_values("Date").reset_index(drop=True)

        # Create a boolean column to mark which transactions have been used
        transaction_log["Used"] = False

        all_dates = prices_df.index.sort_values().unique()

        # Start all holdings at zero, plus the initial cash
        current_holdings = {sym: 0 for sym in self.stocks.keys()}
        current_cash = self.cash_balance

        portfolio_values = []

        for day in all_dates:
            # Grab all not-yet-used transactions with Date <= this day
            mask = (transaction_log["Date"] <= day) & (transaction_log["Used"] == False)
            day_txs = transaction_log.loc[mask]

            # Apply each transaction
            for idx, tx in day_txs.iterrows():
                ttype = tx["Type"].lower()
                tsym = tx["Symbol"]
                qty = tx["Quantity"]
                px = tx["Price"]

                if ttype == "buy":
                    cost = qty * px
                    if current_cash < cost:
                        logging.error("Not enough cash to buy.")
                        continue
                    current_cash -= cost
                    current_holdings[tsym] = current_holdings.get(tsym, 0) + qty

                elif ttype == "sell":
                    if current_holdings.get(tsym, 0) < qty:
                        logging.error("Not enough shares to sell.")
                        continue
                    revenue = qty * px
                    current_cash += revenue
                    current_holdings[tsym] -= qty
                    if current_holdings[tsym] <= 0:
                        current_holdings[tsym] = 0

                # Mark transaction as used
                transaction_log.at[idx, "Used"] = True

            # Now compute portfolio value for this day
            row_dict = {"Date": day, "Cash": current_cash}
            day_val = current_cash

            if by_stock:
                # If user wants each stock's daily value in a separate column
                for sym_ in self.stocks.keys():
                    if sym_ in prices_df.columns:
                        px_today = prices_df.loc[day, sym_]
                        shares_ = current_holdings[sym_]
                        if shares_ > 0:
                            val_ = shares_ * px_today
                            day_val += val_
                            row_dict[sym_] = val_
                        else:
                            row_dict[sym_] = np.nan
            else:
                # Only track total (stocks + cash)
                for sym_ in current_holdings:
                    sh_ = current_holdings[sym_]
                    if sh_ > 0 and sym_ in prices_df.columns:
                        px_today = prices_df.loc[day, sym_]
                        day_val += sh_ * px_today

            row_dict["PortfolioValue"] = day_val
            portfolio_values.append(row_dict)

        dyn_df = pd.DataFrame(portfolio_values).sort_values("Date")
        dyn_df.reset_index(drop=True, inplace=True)

        # Optionally store it if you want to reference it in Summary
        self._dynamic_ts = dyn_df.copy()

        return dyn_df

    def monteCarloSimulation(self, num_simulations=1000, num_days=252):
        """
        GBM-based Monte Carlo simulation on the stock portion's daily returns.
        (Ignoring uninvested cash for the return process.)
        """
        portfolio_returns = self.getPortfolioReturns().dropna()
        if portfolio_returns.empty:
            logging.warning("No portfolio returns for Monte Carlo.")
            return pd.DataFrame()

        mean_ret = portfolio_returns.mean()
        std_dev = portfolio_returns.std()
        drift = mean_ret - 0.5*(std_dev**2)
        dt = 1 / 252

        # Just use the sum of current stock values + cash as the start
        last_port_val = sum(st.value for st in self.stocks.values()) + self.cash_balance

        sim_array = np.zeros((num_days, num_simulations))
        for i in range(num_simulations):
            np.random.seed(i)
            shocks = np.random.normal(0, std_dev, num_days)
            daily_ret = np.exp(drift*dt + shocks)
            path = last_port_val * np.cumprod(daily_ret)
            sim_array[:, i] = path

        if len(portfolio_returns.index) == 0:
            last_date = datetime.today()
        else:
            last_date = portfolio_returns.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=num_days)

        sim_df = pd.DataFrame(
            sim_array, 
            index=future_dates,
            columns=[f"Simulation_{i+1}" for i in range(num_simulations)]
        )
        self.simulated_portfolio_values = sim_df

        # Basic metrics per simulation
        ret_ = sim_df.pct_change().dropna(how='all')
        sharpe_ = ret_.mean(axis=0) / ret_.std(axis=0) * np.sqrt(252)
        neg_only = ret_[ret_ < 0]
        sortino_ = ret_.mean(axis=0) / neg_only.std(axis=0) * np.sqrt(252)
        mdd_ = sim_df.apply(self.calculate_max_drawdown, axis=0)
        var_ = ret_.quantile(0.05, axis=0)
        cvar_ = ret_.apply(lambda x: x[x <= x.quantile(0.05)].mean(), axis=0)

        self.simulated_metrics = pd.DataFrame({
            "Simulation": sim_df.columns,
            "Sharpe Ratio": sharpe_.values,
            "Sortino Ratio": sortino_.values,
            "Max Drawdown": mdd_.values,
            "VaR 95%": var_.values,
            "CVaR 95%": cvar_.values
        })
        return sim_df

    @staticmethod
    def calculate_max_drawdown(series):
        """
        Max drawdown for a price series: 
        the lowest (cumprod / peak) - 1 across the entire timeframe.
        """
        cumret = (1 + series.pct_change()).cumprod()
        peak = cumret.expanding().max()
        dd = (cumret / peak) - 1
        return dd.min()

    def getSimulatedMetrics(self):
        """
        Returns the DataFrame with Monte Carlo simulation metrics if available.
        """
        if self.simulated_metrics is None:
            return pd.DataFrame()
        return self.simulated_metrics
