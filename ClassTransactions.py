#ClassTransactions.py
import pandas as pd
import logging
from datetime import datetime
from ClassPortfolio import Portfolio

class Transactions:
    """
    Manages buy/sell transactions for a Portfolio object.
    We have [Date, Symbol, Type, Quantity] in the Excel input,
    so we must dynamically fetch the price from yfinance data.
    """

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.transaction_log = pd.DataFrame(columns=['Date','Symbol','Type','Quantity','Price','Cash Balance'])

    def get_price_for_transaction(self, symbol, date_):
        """
        Fetches the price for a given symbol on a specific date
        by using the stock's getPriceOnDate() method (which uses 'Adj Close').
        """
        if symbol not in self.portfolio.stocks:
            from ClassStock import Stock
            self.portfolio.stocks[symbol] = Stock(symbol, risk_free_rate=self.portfolio.RF)
            self.portfolio.stocks[symbol].get_data()
        else:
            self.portfolio.stocks[symbol].get_data()

        px = self.portfolio.stocks[symbol].getPriceOnDate(date_)
        if pd.isna(px):
            logging.warning(f"No price found for symbol={symbol} on {date_}, defaulting to 0.0")
            px = 0.0
        return px

    def record_transaction(self, date, symbol, transaction_type, quantity):
        if transaction_type not in ['buy','sell']:
            raise ValueError("Transaction type must be 'buy' or 'sell'.")

        # Convert date to datetime if needed
        if not isinstance(date, datetime):
            date = pd.to_datetime(date)

        # Dynamically get the price for this date
        price = self.get_price_for_transaction(symbol, date)

        # Proceed with cost calculation & update the portfolio
        total_cost = quantity * price
        if transaction_type == 'buy':
            if self.portfolio.cash_balance < total_cost:
                raise ValueError("Insufficient cash for purchase.")
            self.portfolio.cash_balance -= total_cost
            self.portfolio.update_stock(symbol, quantity, price, increase=True)
        else:  # 'sell'
            if not self.portfolio.has_stock(symbol, quantity):
                raise ValueError("Not enough shares to sell.")
            self.portfolio.cash_balance += total_cost
            self.portfolio.update_stock(symbol, quantity, price, increase=False)

        # Record in transaction log
        tx_row = {
            'Date': date,
            'Symbol': symbol,
            'Type': transaction_type,
            'Quantity': quantity,
            'Price': price,
            'Cash Balance': self.portfolio.cash_balance
        }
        if self.transaction_log.empty:
            self.transaction_log = pd.DataFrame([tx_row])
        else:
            self.transaction_log = pd.concat([self.transaction_log, pd.DataFrame([tx_row])],
                                             ignore_index=True)

        logging.info(f"Transaction recorded: {tx_row}")

    def import_transactions_from_excel(self, file_path, sheet_name='Transactions'):
        """
        Reads an Excel sheet with columns [Date, Symbol, Type, Quantity].
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            logging.error(f"Error reading transactions from {file_path}, sheet={sheet_name}: {e}")
            raise

        required = ['Date','Symbol','Type','Quantity']
        if not all(c in df.columns for c in required):
            raise ValueError(f"Sheet must contain columns: {required}")

        # Sort by Date in case it is not sorted
        df.sort_values("Date", inplace=True)

        for _, row in df.iterrows():
            try:
                self.record_transaction(
                    date=row['Date'],
                    symbol=row['Symbol'],
                    transaction_type=row['Type'].lower(),
                    quantity=row['Quantity']
                )
            except Exception as e:
                logging.error(f"Error processing transaction row={row.to_dict()}, e={e}")

    def get_transaction_log(self):
        return self.transaction_log
