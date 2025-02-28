# main.py 

import sys
import logging
import pandas as pd
import csv
from openpyxl import Workbook
from openpyxl.styles import numbers
import openpyxl

from ClassStock import Stock
from ClassPortfolio import Portfolio
from ClassTransactions import Transactions

logging.basicConfig(
    filename='portfolio_analysis.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def main():
    """
    Exemplo de script principal:
      1) Lê Stocks e Transações de um Excel
      2) Cria o Portfolio
      3) Calcula métricas e simulações
      4) Exporta resultados
    """
    # Caminhos OneDrive (exemplo) - ajuste para seu próprio caminho
    excel_file_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Input\input.xlsx"
    output_file_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\PortfolioOutput.xlsx"
    simulations_csv_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\MonteCarloSimulations.csv"
    time_series_csv_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\TimeSeriesData.csv"
    stock_returns_csv_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\StockReturns.csv"

    risk_free_rate = 0.048
    initial_cash = 1_000_000.0
    num_simulations = 1000
    num_days = 252

    # 1) Carrega Stocks
    try:
        df_stocks = pd.read_excel(excel_file_path, sheet_name="Stocks")
    except Exception as e:
        logging.error(f"Could not read Stocks sheet: {e}")
        sys.exit(1)

    if not all(x in df_stocks.columns for x in ["Symbol","Invested","Value"]):
        logging.error("Missing columns in Stocks sheet.")
        sys.exit(1)

    stocks = {}
    for _, row in df_stocks.iterrows():
        sym = row["Symbol"]
        invested = float(str(row["Invested"]).replace(',','.'))
        val = float(str(row["Value"]).replace(',','.'))
        st = Stock(sym, invested=invested, value=val, risk_free_rate=risk_free_rate)
        # Optionally guess quantity:
        c = st.getClose()
        if not c.empty:
            last_price = c.iloc[-1]
            if last_price>0:
                st.quantity = val/last_price
        stocks[sym] = st

    # 2) Cria Portfolio e lê transações
    portfolio = Portfolio(stocks=stocks, risk_free_rate=risk_free_rate, initial_cash=initial_cash)
    transactions = Transactions(portfolio)
    try:
        transactions.import_transactions_from_excel(excel_file_path, sheet_name="Transactions")
    except Exception as e:
        logging.error(f"Transaction import error: {e}")

    # 3) Gather outputs
    try:
        detailed_df = portfolio.getDetailedStockData()
        summary_dict = portfolio.Summary()
        corr_matrix = portfolio.getCorrelationMatrix()  # <- Agora existe
        if not corr_matrix.empty:
            corr_long = corr_matrix.stack().reset_index()
            corr_long.columns=["Stock1","Stock2","Correlation"]
        else:
            corr_long = pd.DataFrame()
    except Exception as e:
        logging.error(f"Error computing data: {e}")
        detailed_df = pd.DataFrame()
        summary_dict = {}
        corr_long = pd.DataFrame()

    # Monte Carlo (portfólio + stocks unificado)
    try:
        mc_df = portfolio.monteCarloSimulation(num_simulations, num_days)
        mc_metrics = portfolio.getSimulatedMetrics()

        # Simular cada stock
        stock_simulations = {}
        for symbol, st in portfolio.stocks.items():
            sim_df = st.monteCarloSimulation(num_simulations, num_days)
            stock_simulations[symbol] = sim_df

        # Juntar num DataFrame “longo”
        sim_records = []
        def add_sim_records(sim_df, name):
            for i in range(1, num_simulations+1):
                col = f"Simulation_{i}"
                tmp = sim_df[[col]].copy()
                tmp = tmp.reset_index().rename(columns={'index':'Date', col:'Value'})
                tmp["Stock"] = name
                tmp["Simulation"] = i
                sim_records.append(tmp)

        # Adiciona do portfolio
        add_sim_records(mc_df, "Portfolio")

        # Cada stock
        for sym, sdf in stock_simulations.items():
            add_sim_records(sdf, sym)

        mc_long_df = pd.concat(sim_records, ignore_index=True)

    except Exception as e:
        logging.error(f"MC simulation error: {e}")
        mc_df = pd.DataFrame()
        mc_metrics = pd.DataFrame()
        mc_long_df = pd.DataFrame()

    # 5) Time series data
    try:
        adj_close = portfolio.getAdjClosePrices()
        if not adj_close.empty:
            portfolio_series = portfolio.getPortfolioAdjustedClosePrices()
            adj_close["Portfolio"] = portfolio_series
            adj_close.reset_index(inplace=True)
            ts_data = pd.melt(adj_close, id_vars=["Date"], var_name="Stock", value_name="AdjustedClosePrice")
        else:
            ts_data = pd.DataFrame()
        stock_returns = portfolio.getReturns()  # agora já inclui col “Portfolio”
    except Exception as e:
        logging.error(f"Error computing time series or returns: {e}")
        ts_data = pd.DataFrame()
        stock_returns = pd.DataFrame()

    # (Exemplo) Se quiseres time series dinâmica, descomente:
    # dyn_df = portfolio.getDynamicTimeSeries(transactions.get_transaction_log())

    # 6) Write Excel
    try:
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            if not detailed_df.empty:
                detailed_df.to_excel(writer, sheet_name="DetailedStockData", index=False)
            if summary_dict:
                df_sum = pd.DataFrame([summary_dict])
                df_sum.to_excel(writer, sheet_name="PortfolioSummary", index=False)
            if not mc_metrics.empty:
                mc_metrics.to_excel(writer, sheet_name="SimulatedMetrics", index=False)
            if not corr_long.empty:
                corr_long.to_excel(writer, sheet_name="CorrelationMatrix", index=False)

            workbook = writer.book
            for shtname in writer.sheets:
                sht = writer.sheets[shtname]
                for row in sht.iter_rows(min_row=2, max_row=sht.max_row):
                    for cell in row:
                        if cell.value is not None and isinstance(cell.value,(int,float)):
                            cell.number_format = '#,##0.00'
        print(f"Excel output saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Error writing Excel: {e}")

    # 7) Write CSVs
    if not mc_long_df.empty:
        try:
            mc_long_df.to_csv(simulations_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
            print(f"Monte Carlo simulations written to {simulations_csv_path}")
        except Exception as e:
            logging.error(f"Error writing simulations CSV: {e}")

    if not ts_data.empty:
        try:
            ts_data.to_csv(time_series_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
            print(f"Time series data written to {time_series_csv_path}")
        except Exception as e:
            logging.error(f"Error writing time series CSV: {e}")

    if not stock_returns.empty:
        try:
            stock_returns.reset_index(inplace=True)
            ret_long = pd.melt(stock_returns, id_vars=['Date'], var_name='Stock', value_name='DailyReturn')
            ret_long.dropna(inplace=True)
            ret_long.to_csv(stock_returns_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
            print(f"Stock returns data written to {stock_returns_csv_path}")
        except Exception as e:
            logging.error(f"Error writing stock returns CSV: {e}")

if __name__ == "__main__":
    main()
