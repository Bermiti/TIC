import sys
import logging
import pandas as pd
import csv
from openpyxl import Workbook
from openpyxl.styles import numbers
import openpyxl
import numpy as np
from datetime import datetime

from ClassStock import Stock
from ClassPortfolio import Portfolio
from ClassTransactions import Transactions
from utils import calculate_simulated_metrics

logging.basicConfig(
    filename='portfolioanalysis.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def main():
    """
    Main script that:
      1) Reads the Stocks and Transactions worksheets from an Excel input file.
      2) Creates the Portfolio.
      3) Computes metrics and runs Monte Carlo simulations for:
         - Portfolio
         - Each Stock
         - Benchmark (^GSPC)
      4) Exports results (Excel and CSVs), including a CSV with returns (all stocks + Portfolio + Benchmark).
    """

    # Input and output paths
    excel_file_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Input\input.xlsx"
    output_file_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\PortfolioOutput.xlsx"

    simulations_csv_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\MonteCarloSimulations.csv"
    mc_metrics_csv_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\MonteCarloMetrics.csv"
    stock_returns_csv_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\StockReturns.csv"
    time_series_csv_path = r"C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Output\TimeSeriesData.csv"

    risk_free_rate = 0.048
    initial_cash = 1_000_000.0
    num_simulations = 1000
    num_days = 252

    # 1) Load Stocks
    try:
        df_stocks = pd.read_excel(excel_file_path, sheet_name="Stocks")
    except Exception as e:
        logging.error(f"Could not read the Stocks worksheet: {e}")
        sys.exit(1)

    required_cols = ["Symbol","Invested","Value"]
    if not all(x in df_stocks.columns for x in required_cols):
        logging.error(f"Missing columns in Stocks worksheet. Required: {required_cols}")
        sys.exit(1)

    stocks = {}
    for _, row in df_stocks.iterrows():
        sym = row["Symbol"]
        invested = float(str(row["Invested"]).replace(',','.'))
        val = float(str(row["Value"]).replace(',','.'))
        st = Stock(sym, invested=invested, value=val, risk_free_rate=risk_free_rate)
        # Estimate number of shares
        c = st.getClose()
        if not c.empty:
            last_price = c.iloc[-1]
            if last_price > 0:
                st.quantity = val / last_price
        stocks[sym] = st

    # Create Benchmark
    benchmark_symbol = "^GSPC"
    benchmark_stock = Stock(benchmark_symbol, 0.0, 0.0, risk_free_rate)

    # 2) Create Portfolio and load transactions
    portfolio = Portfolio(stocks=stocks, risk_free_rate=risk_free_rate, initial_cash=initial_cash)
    transactions = Transactions(portfolio)
    try:
        transactions.import_transactions_from_excel(excel_file_path, sheet_name="Transactions")
    except Exception as e:
        logging.error(f"Error importing transactions: {e}")

    # 3) Collect portfolio data
    try:
        detailed_df = portfolio.getDetailedStockData()
        summary_dict = portfolio.Summary()
        corr_matrix = portfolio.getCorrelationMatrix()
        if not corr_matrix.empty:
            corr_long = corr_matrix.stack().reset_index()
            corr_long.columns = ["Stock1","Stock2","Correlation"]
        else:
            corr_long = pd.DataFrame()
    except Exception as e:
        logging.error(f"Error computing portfolio data: {e}")
        detailed_df = pd.DataFrame()
        summary_dict = {}
        corr_long = pd.DataFrame()

    # 4) Monte Carlo simulations
    # 4.1) Portfolio
    try:
        mc_df_portfolio = portfolio.monteCarloSimulation(num_simulations, num_days)
        mc_metrics_portfolio = portfolio.getSimulatedMetrics()
        mc_metrics_portfolio["Asset"] = "Portfolio"
    except Exception as e:
        logging.error(f"Error in Portfolio MC: {e}")
        mc_df_portfolio = pd.DataFrame()
        mc_metrics_portfolio = pd.DataFrame()

    # 4.2) Each stock
    stock_simulations = {}
    stock_metrics_all = []
    for symbol, st in portfolio.stocks.items():
        try:
            sim_df = st.monteCarloSimulation(num_simulations, num_days)
            stock_simulations[symbol] = sim_df
            df_metrics = calculate_simulated_metrics(sim_df)
            df_metrics["Asset"] = symbol
            stock_metrics_all.append(df_metrics)
        except Exception as e:
            logging.error(f"Error in MC for stock={symbol}: {e}")

    # 4.3) Benchmark
    try:
        bench_sim_df = benchmark_stock.monteCarloSimulation(num_simulations, num_days)
        bench_metrics_df = calculate_simulated_metrics(bench_sim_df)
        bench_metrics_df["Asset"] = "Benchmark"
    except Exception as e:
        logging.error(f"Error in Benchmark MC: {e}")
        bench_sim_df = pd.DataFrame()
        bench_metrics_df = pd.DataFrame()

    # Combine simulations into a single "long" DataFrame
    sim_records = []

    def add_sim_records(sim_df, asset_name):
        for i in range(1, num_simulations+1):
            col = f"Simulation_{i}"
            if col not in sim_df.columns:
                continue
            tmp = sim_df[[col]].copy()
            tmp = tmp.reset_index().rename(columns={'index': 'Date', col: 'Value'})
            tmp["Asset"] = asset_name
            tmp["Simulation"] = i
            sim_records.append(tmp)

    add_sim_records(mc_df_portfolio, "Portfolio")
    for sym, sdf in stock_simulations.items():
        add_sim_records(sdf, sym)
    add_sim_records(bench_sim_df, "Benchmark")

    if sim_records:
        mc_long_df = pd.concat(sim_records, ignore_index=True)
    else:
        mc_long_df = pd.DataFrame()

    # Combine metrics as well
    all_sim_metrics = []
    if not mc_metrics_portfolio.empty:
        all_sim_metrics.append(mc_metrics_portfolio[[
            "Simulation","Sharpe Ratio","Sortino Ratio","Max Drawdown","VaR 95%","CVaR 95%","Asset"
        ]])
    if stock_metrics_all:
        df_stock_metrics = pd.concat(stock_metrics_all, ignore_index=True)
        all_sim_metrics.append(df_stock_metrics)
    if not bench_metrics_df.empty:
        all_sim_metrics.append(bench_metrics_df)

    if all_sim_metrics:
        mc_metrics_full = pd.concat(all_sim_metrics, ignore_index=True)
    else:
        mc_metrics_full = pd.DataFrame()

    # 5) Time series (Portfolio + Benchmark)
    try:
        dyn_df = portfolio.getDynamicTimeSeries(transactions.get_transaction_log(), by_stock=True)

        if dyn_df.empty:
            logging.warning("Dynamic time series is empty, fallback to buy-and-hold.")
            fallback_data = portfolio.getAdjClosePrices()
            if fallback_data.empty:
                # no data
                ts_data = pd.DataFrame()
                ret_long = pd.DataFrame()
            else:
                bench_close = benchmark_stock.getClose()
                if not bench_close.empty:
                    fallback_data["Benchmark"] = bench_close.reindex(fallback_data.index).fillna(method='ffill')
                fallback_data["Portfolio"] = portfolio.getPortfolioAdjustedClosePrices()
                fallback_data.reset_index(inplace=True)
                fallback_data.rename(columns={"index":"Date"}, inplace=True)

                # ts_data - in long format
                ts_data = pd.melt(
                    fallback_data,
                    id_vars=["Date"],
                    var_name="Asset",
                    value_name="Value"
                )

                # daily returns
                pivot_ts = fallback_data.set_index("Date")
                daily_returns = pivot_ts.pct_change().dropna(how='all').reset_index()
                ret_long = pd.melt(
                    daily_returns,
                    id_vars=["Date"],
                    var_name="Asset",
                    value_name="DailyReturn"
                )
                ret_long.dropna(subset=["DailyReturn"], inplace=True)

        else:
            # we have dates and portfolio values
            bench_close = benchmark_stock.getClose()
            bench_close = bench_close.reindex(dyn_df["Date"]).fillna(method='ffill')
            bench_close.name = "Benchmark"

            ts_records = []
            for idx, row in dyn_df.iterrows():
                date_ = row["Date"]

                # per stock (if by_stock=True)
                for col in dyn_df.columns:
                    if col in ["Date","Cash","PortfolioValue"]:
                        continue
                    ts_records.append({"Date": date_, "Asset": col, "Value": row[col]})

                # total value of the Portfolio
                ts_records.append({"Date": date_, "Asset": "Portfolio", "Value": row["PortfolioValue"]})

                # Benchmark
                val_bench = bench_close.loc[date_] if date_ in bench_close.index else None
                ts_records.append({"Date": date_, "Asset": "Benchmark", "Value": val_bench})

            ts_data = pd.DataFrame(ts_records)
            if ts_data.empty:
                ret_long = pd.DataFrame()
            else:
                piv = ts_data.pivot(index="Date", columns="Asset", values="Value")
                piv.sort_index(inplace=True)
                daily_returns = piv.pct_change().dropna(how='all').reset_index()
                ret_long = pd.melt(daily_returns, id_vars=["Date"], var_name="Asset", value_name="DailyReturn")
                ret_long.dropna(subset=["DailyReturn"], inplace=True)

    except Exception as e:
        logging.error(f"Error creating time series or returns: {e}")
        ts_data = pd.DataFrame()
        ret_long = pd.DataFrame()

    # 6) Export main data to Excel
    try:
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            # DetailedStockData
            if not detailed_df.empty:
                detailed_df.to_excel(writer, sheet_name="DetailedStockData", index=False)

            # PortfolioSummary
            if summary_dict:
                pd.DataFrame([summary_dict]).to_excel(writer, sheet_name="PortfolioSummary", index=False)

            # CorrelationMatrix
            if not corr_long.empty:
                corr_long.to_excel(writer, sheet_name="CorrelationMatrix", index=False)

            # Numeric formatting (optional)
            workbook = writer.book
            for shtname in writer.sheets:
                sht = writer.sheets[shtname]
                for row in sht.iter_rows(min_row=2, max_row=sht.max_row):
                    for cell in row:
                        if cell.value is not None and isinstance(cell.value, (int,float)):
                            cell.number_format = '#,##0.00'

        print(f"Output Excel saved at {output_file_path}")
    except Exception as e:
        logging.error(f"Error writing Excel: {e}")

    # 7) Export CSVs
    # Monte Carlo Simulations (all simulations in long format)
    try:
        mc_long_df.to_csv(simulations_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
        print(f"MC simulations saved at {simulations_csv_path}")
    except Exception as e:
        logging.error(f"Error writing simulations CSV: {e}")

    # Monte Carlo Metrics
    try:
        mc_metrics_full.to_csv(mc_metrics_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
        print(f"MC metrics saved at {mc_metrics_csv_path}")
    except Exception as e:
        logging.error(f"Error writing MC metrics CSV: {e}")

    # StockReturns (daily returns for each asset + Portfolio + Benchmark)
    try:
        ret_long.to_csv(stock_returns_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
        print(f"Daily returns (StockReturns) saved at {stock_returns_csv_path}")
    except Exception as e:
        logging.error(f"Error writing returns CSV: {e}")

    # TimeSeriesData (daily values in long format)
    try:
        ts_data.to_csv(time_series_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
        print(f"Time series saved at {time_series_csv_path}")
    except Exception as e:
        logging.error(f"Error writing time series CSV: {e}")


if __name__ == "__main__":
    main()
