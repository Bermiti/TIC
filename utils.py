import pandas as pd
import numpy as np

def calculate_simulated_metrics(sim_df):
    """
    Utility function that, given a DataFrame of simulations (num_days x num_simulations),
    returns a DataFrame with Sharpe, Sortino, Max Drawdown, VaR, and CVaR metrics 
    for each column (simulation).
    """
    ret_ = sim_df.pct_change().dropna(how='all')  # daily returns
    sharpe_ = {}
    sortino_ = {}
    mdd_ = {}
    var_ = {}
    cvar_ = {}

    for col in sim_df.columns:
        series = sim_df[col].pct_change().dropna()
        if series.empty:
            sharpe_[col] = None
            sortino_[col] = None
            mdd_[col] = None
            var_[col] = None
            cvar_[col] = None
            continue

        mean_ = series.mean()
        std_ = series.std()

        # Sharpe
        if std_ == 0:
            sharpe_[col] = None
        else:
            sharpe_[col] = (mean_ / std_) * np.sqrt(252)

        # Sortino
        negative = series[series < 0]
        if negative.empty or negative.std() == 0:
            sortino_[col] = None
        else:
            sortino_[col] = (mean_ / negative.std()) * np.sqrt(252)

        # Max Drawdown
        cum = (1 + series).cumprod()
        peak = cum.expanding().max()
        dd = (cum / peak) - 1
        mdd_[col] = dd.min()

        # VaR and CVaR
        q95 = series.quantile(0.05)
        var_[col] = q95
        in_tail = series[series <= q95]
        cvar_[col] = in_tail.mean() if not in_tail.empty else None

    df_out = pd.DataFrame({
        "Simulation": sim_df.columns,
        "Sharpe Ratio": [sharpe_[c] for c in sim_df.columns],
        "Sortino Ratio": [sortino_[c] for c in sim_df.columns],
        "Max Drawdown": [mdd_[c] for c in sim_df.columns],
        "VaR 95%": [var_[c] for c in sim_df.columns],
        "CVaR 95%": [cvar_[c] for c in sim_df.columns]
    })
    return df_out
