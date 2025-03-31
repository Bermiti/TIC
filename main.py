# main.py

import sys
import logging
import pandas as pd
import csv
import numpy as np
from datetime import datetime
from ClassStock import Stock
from ClassPortfolio import Portfolio
from ClassTransactions import Transactions
from utils import calculate_simulated_metrics

# NOTE: filemode='a' appends logs each run, 'w' overwrites
logging.basicConfig(
    filename='portfolioanalysis.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

import os
from pathlib import Path

def main():
    """
    Main script that:
      1) Reads ONLY the Transactions sheet with [Date, Symbol, Type, Quantity]
      2) Dynamically finds the price from the Stock data
      3) Builds a portfolio from those transactions
      4) Computes metrics, runs Monte Carlo, and exports results
    """

    # Detect the Operating System
    is_windows = os.name == "nt"

    # Get the script's directory dynamically
    script_dir = Path(__file__).resolve().parent

    # Set base directory (TICRiskAnalysis)
    base_dir = script_dir.parent

    # Define Input and Output paths dynamically
    excel_file_path = base_dir / "Input" / "input.xlsx"
    output_file_path = base_dir / "Output" / "PortfolioOutput.xlsx"

    simulations_csv_path = base_dir / "Output" / "MonteCarloSimulations.csv"
    mc_metrics_csv_path = base_dir / "Output" / "MonteCarloMetrics.csv"
    stock_returns_csv_path = base_dir / "Output" / "StockReturns.csv"
    time_series_csv_path = base_dir / "Output" / "TimeSeriesData.csv"

    excel_file_path = str(excel_file_path)
    output_file_path = str(output_file_path)
    simulations_csv_path = str(simulations_csv_path)
    mc_metrics_csv_path = str(mc_metrics_csv_path)
    stock_returns_csv_path = str(stock_returns_csv_path)
    time_series_csv_path = str(time_series_csv_path)

    # Print debug information
    print(f"Running on: {'Windows' if is_windows else 'macOS/Linux'}")

    # Parameters
    risk_free_rate = 0.048
    initial_cash = 1_000_000.0
    num_simulations = 1000
    num_days = 252

    # Try reading the Transactions sheet
    try:
        df_tx = pd.read_excel(excel_file_path, sheet_name="Transactions")
    except Exception as e:
        logging.error(f"Could not read Transactions sheet: {e}")
        sys.exit(1)

    required_cols = ["Date","Symbol","Type","Quantity"]
    if not all(x in df_tx.columns for x in required_cols):
        logging.error(f"Missing columns in Transactions. Required: {required_cols}")
        sys.exit(1)

    # Sort transactions by date
    df_tx.sort_values("Date", inplace=True)

    # Identify earliest transaction date
    earliest_tx_date = df_tx["Date"].min()

    # We'll define a realistic end_date so we don't ask for future data from Yahoo
    # For example, use today's date if it's truly valid, or set it to something known
    today_system = pd.Timestamp.today().floor('D')
    # If your system date is far in the future, you might want a fallback:
    #   real_end_date = min(today_system, pd.Timestamp("2024-12-31"))
    real_end_date = today_system

    # Build stock objects for unique symbols, using earliest date as start_date
    symbols = df_tx["Symbol"].unique().tolist()
    stocks = {}
    for sym in symbols:
        st = Stock(symbol=sym, risk_free_rate=risk_free_rate, investment=0.0, quantity=0.0)
        st.get_data(start_date=earliest_tx_date, end_date=real_end_date)
        stocks[sym] = st

    # Create the portfolio and load transactions
    portfolio = Portfolio(stocks=stocks, risk_free_rate=risk_free_rate, initial_cash=initial_cash)
    transactions = Transactions(portfolio)
    try:
        transactions.import_transactions_from_excel(excel_file_path, sheet_name="Transactions")
    except Exception as e:
        logging.error(f"Error importing transactions: {e}")

    # Create a benchmark (optional)
    benchmark_symbol = "^GSPC"
    benchmark_stock = Stock(benchmark_symbol, risk_free_rate=risk_free_rate)
    benchmark_stock.get_data(start_date=earliest_tx_date, end_date=real_end_date)

    # Collect portfolio data
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

    # Monte Carlo Simulations
    # 1) Portfolio
    try:
        mc_df_portfolio = portfolio.monteCarloSimulation(num_simulations, num_days)
        mc_metrics_portfolio = portfolio.getSimulatedMetrics()
        if not mc_metrics_portfolio.empty:
            mc_metrics_portfolio["Asset"] = "Portfolio"
    except Exception as e:
        logging.error(f"Error in Portfolio MC: {e}")
        mc_df_portfolio = pd.DataFrame()
        mc_metrics_portfolio = pd.DataFrame()

    # 2) Each Stock
    stock_simulations = {}
    stock_metrics_all = []
    for symbol, st in portfolio.stocks.items():
        try:
            sim_df = st.monteCarloSimulation(num_simulations, num_days)
            if not sim_df.empty:
                stock_simulations[symbol] = sim_df
                df_metrics = calculate_simulated_metrics(sim_df)
                df_metrics["Asset"] = symbol
                stock_metrics_all.append(df_metrics)
        except Exception as e:
            logging.error(f"Error in MC for {symbol}: {e}")

    # 3) Benchmark
    try:
        bench_sim_df = benchmark_stock.monteCarloSimulation(num_simulations, num_days)
        if not bench_sim_df.empty:
            bench_metrics_df = calculate_simulated_metrics(bench_sim_df)
            bench_metrics_df["Asset"] = "Benchmark"
        else:
            bench_metrics_df = pd.DataFrame()
    except Exception as e:
        logging.error(f"Error in Benchmark MC: {e}")
        bench_sim_df = pd.DataFrame()
        bench_metrics_df = pd.DataFrame()

    # Combine all simulations in a "long" format
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

    # Combine metrics
    all_sim_metrics = []
    if not mc_metrics_portfolio.empty:
        all_sim_metrics.append(mc_metrics_portfolio[[
            "Simulation","Sharpe Ratio","Sortino Ratio","Max Drawdown","VaR 95%","CVaR 95%","Asset"
        ]])
    if stock_metrics_all:
        df_stock_metrics = pd.concat(stock_metrics_all, ignore_index=True)
        all_sim_metrics.append(df_stock_metrics)
    if not bench_sim_df.empty and not bench_metrics_df.empty:
        all_sim_metrics.append(bench_metrics_df)

    if all_sim_metrics:
        mc_metrics_full = pd.concat(all_sim_metrics, ignore_index=True)
    else:
        mc_metrics_full = pd.DataFrame()

    # Build time series for Portfolio + Benchmark
    try:
        dyn_df = portfolio.getDynamicTimeSeries(transactions.get_transaction_log(), by_stock=True)
        if dyn_df.empty:
            # fallback to simple buy-and-hold approach
            fallback_data = portfolio.getAdjClosePrices()
            if fallback_data.empty:
                ts_data = pd.DataFrame()
                ret_long = pd.DataFrame()
            else:
                # add benchmark if available
                bench_close = benchmark_stock.getClose()
                if not bench_close.empty:
                    fallback_data["Benchmark"] = bench_close.reindex(fallback_data.index).fillna(method='ffill')
                # add portfolio as a weighted sum (static)
                fallback_data["Portfolio"] = portfolio.getPortfolioAdjustedClosePrices()
                fallback_data.reset_index(inplace=True)
                fallback_data.rename(columns={"index":"Date"}, inplace=True)

                ts_data = pd.melt(
                    fallback_data,
                    id_vars=["Date"],
                    var_name="Asset",
                    value_name="Value"
                )
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
            # We have a daily series from transaction-based approach
            bench_close = benchmark_stock.getClose()
            bench_close = bench_close.reindex(dyn_df["Date"]).fillna(method='ffill')
            bench_close.name = "Benchmark"

            ts_records = []
            for idx, row in dyn_df.iterrows():
                date_ = row["Date"]
                # for each stock column (if by_stock=True, we have them)
                for col in dyn_df.columns:
                    if col in ["Date","Cash","PortfolioValue"]:
                        continue
                    ts_records.append({"Date": date_, "Asset": col, "Value": row[col]})

                # total portfolio
                ts_records.append({"Date": date_, "Asset": "Portfolio", "Value": row["PortfolioValue"]})

                # benchmark
                val_bench = bench_close.loc[date_] if date_ in bench_close.index else None
                ts_records.append({"Date": date_, "Asset": "Benchmark", "Value": val_bench})

            ts_data = pd.DataFrame(ts_records)
            if ts_data.empty:
                ret_long = pd.DataFrame()
            else:
                piv = ts_data.pivot(index="Date", columns="Asset", values="Value")
                piv.sort_index(inplace=True)
                daily_returns = piv.pct_change().dropna(how='all').reset_index()
                ret_long = pd.melt(
                    daily_returns,
                    id_vars=["Date"],
                    var_name="Asset",
                    value_name="DailyReturn"
                )
                ret_long.dropna(subset=["DailyReturn"], inplace=True)

    except Exception as e:
        logging.error(f"Error creating time series or returns: {e}")
        ts_data = pd.DataFrame()
        ret_long = pd.DataFrame()

    # Export to Excel, ensuring at least 1 sheet is visible
    try:
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            wrote_sheet = False

            if not detailed_df.empty:
                detailed_df.to_excel(writer, sheet_name="DetailedStockData", index=False)
                wrote_sheet = True

            if summary_dict:
                pd.DataFrame([summary_dict]).to_excel(writer, sheet_name="PortfolioSummary", index=False)
                wrote_sheet = True

            if not corr_long.empty:
                corr_long.to_excel(writer, sheet_name="CorrelationMatrix", index=False)
                wrote_sheet = True

            # If we never wrote anything, create a dummy sheet so the file is valid.
            if not wrote_sheet:
                dummy_df = pd.DataFrame({"Info": ["No portfolio data to show"]})
                dummy_df.to_excel(writer, sheet_name="EmptyReport", index=False)

            # Optionally format numeric cells
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

    # Export CSVs
    try:
        mc_long_df.to_csv(simulations_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
        print(f"MC simulations saved at {simulations_csv_path}")
    except Exception as e:
        logging.error(f"Error writing simulations CSV: {e}")

    try:
        mc_metrics_full.to_csv(mc_metrics_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
        print(f"MC metrics saved at {mc_metrics_csv_path}")
    except Exception as e:
        logging.error(f"Error writing MC metrics CSV: {e}")

    try:
        ret_long.to_csv(stock_returns_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
        print(f"Daily returns saved at {stock_returns_csv_path}")
    except Exception as e:
        logging.error(f"Error writing returns CSV: {e}")

    try:
        ts_data.to_csv(time_series_csv_path, sep=';', decimal=',', index=False, quoting=csv.QUOTE_ALL)
        print(f"Time series saved at {time_series_csv_path}")
    except Exception as e:
        logging.error(f"Error writing time series CSV: {e}")


if __name__ == "__main__":
    main()