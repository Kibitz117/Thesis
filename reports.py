import pandas_market_calendars as mcal
import pandas as pd
import quantstats as qs

# Load your data
combined_portfolios_performance = pd.read_csv('combined_portfolios.csv')
combined_k_portfolios_performance = pd.read_csv('combined_k_portfolios.csv')

# Set up the DataFrame
combined_portfolios_performance.columns = ['date', 'returns']
combined_portfolios_performance['date'] = pd.to_datetime(combined_portfolios_performance['date'])
combined_portfolios_performance.set_index('date', inplace=True)

combined_k_portfolios_performance.columns = ['date', 'returns']
combined_k_portfolios_performance['date'] = pd.to_datetime(combined_k_portfolios_performance['date'])
combined_k_portfolios_performance.set_index('date', inplace=True)

# Get NYSE trading days
nyse = mcal.get_calendar('NYSE')
trading_days = nyse.schedule(start_date='2000-01-01', end_date='2016-12-31')
trading_day_list = mcal.date_range(trading_days, frequency='1D')

# Normalize and localize to remove timezone
trading_day_list = trading_day_list.normalize().tz_localize(None)

# Reindex the data to match trading days, filling non-trading days with NaN
combined_portfolios_performance = combined_portfolios_performance.reindex(trading_day_list)
combined_k_portfolios_performance = combined_k_portfolios_performance.reindex(trading_day_list)

# Generate the reports
qs.reports.html(combined_portfolios_performance['returns'],combined_k_portfolios_performance['returns'], output='comparison.html', title='Weighted Score vs K Portfolio')
