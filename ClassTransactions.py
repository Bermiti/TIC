import pandas as pd
import logging
from datetime import datetime

from ClassPortfolio import Portfolio

class Transactions:
    """
    Manages buy/sell transactions of the portfolio.
    """

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.transaction_log = pd.DataFrame(columns=['Date','Symbol','Type','Quantity','Price','Cash Balance'])
       

    def record_transaction(self, date, symbol, transaction_type, quantity, price):
        if transaction_type not in ['buy','sell']:
            raise ValueError("Transaction type must be 'buy' or 'sell'.")

        if not isinstance(date, datetime):
            date = pd.to_datetime(date)  # convert string or other format to Timestamp


        total_cost = quantity * price

        if transaction_type == 'buy':
            if self.portfolio.cash_balance < total_cost:
                raise ValueError("Insufficient cash for purchase.")
            self.portfolio.cash_balance -= total_cost
            self.portfolio.update_stock(symbol, quantity, price, increase=True)
        else:  # sell
            if not self.portfolio.has_stock(symbol, quantity):
                raise ValueError("Not enough shares to sell.")
            self.portfolio.cash_balance += total_cost
            self.portfolio.update_stock(symbol, quantity, price, increase=False)

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
        Reads an Excel sheet with columns [Date, Symbol, Type, Price, Quantity]
        and processes them, updating the portfolio accordingly.
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            logging.error(f"Error reading transactions from {file_path}, sheet={sheet_name}: {e}")
            raise
            

        required = ['Date','Symbol','Type','Price','Quantity']
        if not all(c in df.columns for c in required):
            raise ValueError(f"Transactions sheet must contain columns: {required}")

        for _, row in df.iterrows():
            try:
                self.record_transaction(
                    date=str(row['Date']),
                    symbol=row['Symbol'],
                    transaction_type=row['Type'].lower(),
                    quantity=row['Quantity'],
                    price=row['Price']
                )
            except Exception as e:
                logging.error(f"Error processing transaction row: {row.to_dict()}, e={e}")

    def get_transaction_log(self):
        # Return a copy or direct reference if you prefer
        return self.transaction_log
