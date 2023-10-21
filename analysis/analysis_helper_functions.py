from math import sqrt
import pandas as pd
import numpy as np
# Function to calculate portfolio returns
def calculate_portfolio_returns(portfolios, returns_df):
    merged_df = portfolios.merge(returns_df, on=['date', 'TICKER'])
    portfolio_returns = merged_df['RET'].groupby(merged_df['date']).sum()
    
    return portfolio_returns


# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, rf=0.02):
    risk_free_rate = 0.02
    daily_risk_free_return = risk_free_rate/252

    # Calculate the excess returns by subtracting the daily returns by daily risk-free return
    excess_daily_returns = returns - daily_risk_free_return

    # Calculate the sharpe ratio using the given formula
    sharpe_ratio = (excess_daily_returns.mean() /
                    excess_daily_returns.std()) * np.sqrt(252)
    return sharpe_ratio
