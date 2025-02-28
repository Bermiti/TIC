# ClassTransactions.py

import pandas as pd
import logging
from datetime import datetime

from ClassPortfolio import Portfolio

class Transactions:
    """
    Class to handle portfolio transactions (buy/sell).
    """
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.transaction_log = pd.DataFrame(columns=['Date','Symbol','Type','Quantity','Price','Cash Balance'])

    def record_transaction(self, date, symbol, transaction_type, quantity, price):
        if transaction_type not in ['buy','sell']:
            raise ValueError("Transaction type must be 'buy' or 'sell'.")

        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        total_cost = quantity * price

        if transaction_type == 'buy':
            if self.portfolio.cash_balance < total_cost:
                raise ValueError("Insufficient cash.")
            self.portfolio.cash_balance -= total_cost
            self.portfolio.update_stock(symbol, quantity, price, increase=True)

        else:  # sell
            if not self.portfolio.has_stock(symbol, quantity):
                raise ValueError("Not enough shares.")
            self.portfolio.cash_balance += total_cost
            self.portfolio.update_stock(symbol, quantity, price, increase=False)

        # Log
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
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            logging.error(f"Error reading transactions from {file_path}, sheet={sheet_name}: {e}")
            raise

        required = ['Date','Symbol','Type','Price','Quantity']
        if not all(c in df.columns for c in required):
            raise ValueError(f"Transaction sheet must have columns: {required}")

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
                logging.error(f"Error processing row: {row.to_dict()}, e={e}")

    def get_transaction_log(self):
        return self.transaction_log
