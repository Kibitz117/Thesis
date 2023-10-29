from math import sqrt
import pandas as pd
import numpy as np
# Function to calculate portfolio returns
def calculate_portfolio_returns(portfolios, returns_df, position_type='long'):
    # Merging the portfolios DataFrame with the returns DataFrame on 'date' and 'TICKER'
    merged_df = portfolios.merge(returns_df, on=['date', 'TICKER'])
    
    # Calculating the return based on the position type
    if position_type == 'short':
        merged_df['weighted_RET'] = -merged_df['RET']
    else:  # Default to long position if anything other than 'short' is specified
        merged_df['weighted_RET'] = merged_df['RET']
    
    # Grouping by date and summing the weighted returns to get the portfolio return for each date
    portfolio_returns = merged_df.groupby('date')['weighted_RET'].sum()
    
    return portfolio_returns


# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, rf=0.004):
    risk_free_rate = 0.02
    daily_risk_free_return = risk_free_rate/252

    # Calculate the excess returns by subtracting the daily returns by daily risk-free return
    excess_daily_returns = returns - daily_risk_free_return

    # Calculate the sharpe ratio using the given formula
    sharpe_ratio = (excess_daily_returns.mean() /
                    excess_daily_returns.std()) * np.sqrt(252)
    return sharpe_ratio
