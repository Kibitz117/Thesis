from math import sqrt
import pandas as pd
# Function to calculate portfolio returns
def calculate_portfolio_returns(portfolios, returns_df):
    merged_df = portfolios.merge(returns_df, on=['date', 'TICKER'])
    
    # Debugging print to check the merged DataFrame
    print(merged_df.head())
    
    portfolio_returns = merged_df['RET'].groupby(merged_df['date']).sum()
    
    # Debugging print to check the portfolio returns
    print(portfolio_returns.head())
    
    return portfolio_returns


# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, rf=0.0004):
    expected_return = returns.mean()
    std_dev = returns.std()
    sharpe_ratio = (expected_return - rf) / std_dev if std_dev != 0 else 0  # Added a condition to avoid division by zero
    return sharpe_ratio

def calculate_annualized_sharpe_ratio(sharpe_ratio, periods_per_year=252):
    annualized_sharpe_ratio = sharpe_ratio * sqrt(periods_per_year)
    return annualized_sharpe_ratio