import torch
import pandas as pd
from datetime import datetime
import sys
from tqdm import tqdm
import numpy as np
sys.path.append('utils')
sys.path.append('models')
sys.path.append('data_func')
from model_factory import ModelFactory
from data_helper_functions import create_study_periods,create_tensors
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
#We aim to long/short k stocks in a given day, but if there are not k stocks that have the highest probability of being long/short we can do <k
def generate_predictions(model, test_data, device, k=5):
    model.eval()  # Set the model to evaluation mode
    predictions_by_date = {}

    with torch.no_grad():
        for ticker, date, data, _ in test_data:
            data_tensor = data.to(device)
            outputs = model(data_tensor.unsqueeze(0))  # Get model outputs
            probabilities = torch.softmax(outputs, dim=1).squeeze()  # Probabilities for each class

            # Assuming class 0 is for underperformance and class 1 is for outperformance
            prob_underperform = probabilities[0].item()
            prob_outperform = probabilities[1].item()

            # Organize probabilities by date and then by ticker
            if date not in predictions_by_date:
                predictions_by_date[date] = []
            predictions_by_date[date].append((ticker, prob_outperform, prob_underperform))

    # Rank stocks for each date for long and short positions
    ranked_predictions_by_date = {}
    for date, ticker_probs in predictions_by_date.items():
        # Stocks are eligible for long if prob_outperform > prob_underperform
        eligible_long = [(ticker, prob_outperform) for ticker, prob_outperform, prob_underperform in ticker_probs if prob_outperform > prob_underperform]
        eligible_long.sort(key=lambda x: x[1], reverse=True)  # Sort by probability of outperforming
        long_tickers = eligible_long[:k]  # Select top k stocks

        # Stocks are eligible for short if prob_underperform > prob_outperform
        eligible_short = [(ticker, prob_underperform) for ticker, prob_outperform, prob_underperform in ticker_probs if prob_underperform > prob_outperform]
        eligible_short.sort(key=lambda x: x[1], reverse=True)  # Sort by probability of underperforming
        short_tickers = eligible_short[:k]  # Select top k stocks

        ranked_predictions_by_date[date] = {
            'long': long_tickers,
            'short': short_tickers
        }

    return ranked_predictions_by_date




def generate_predictions_with_abstention(model, test_data, device, k=10, reward=2.0):
    model.eval()
    predictions_by_date = {}
    abstentions_by_date = {}

    with torch.no_grad():
        for ticker, date, data, _ in test_data:
            data_tensor = data.to(device)
            outputs = model(data_tensor.unsqueeze(0))  # Get model outputs
            outputs = F.softmax(outputs, dim=1)
            class_outputs, abstention_output = outputs[:, :-1], outputs[:, -1]
            
            # Check if the model abstains
            gain = torch.max(class_outputs).item()  # Gain of the best class
            is_abstained = abstention_output.item() > gain / reward

            if is_abstained:
                # Record abstention
                if date not in abstentions_by_date:
                    abstentions_by_date[date] = []
                abstentions_by_date[date].append(ticker)
            else:
                # Record prediction
                prob_outperform = class_outputs[0, 1].item()  # Probability of outperforming
                if date not in predictions_by_date:
                    predictions_by_date[date] = {}
                predictions_by_date[date][ticker] = prob_outperform

    # Ranking and selecting top and bottom k stocks
    ranked_portfolios = rank_stocks_by_probability(predictions_by_date, k)
    return ranked_portfolios, abstentions_by_date




import numpy as np
import pandas as pd

def analyze_portfolio_performance(historical_data, ranked_portfolios):
    portfolio_daily_returns = []
    
    # Normalize dates in historical_data to ensure they are date-only with no time
    historical_data['date'] = pd.to_datetime(historical_data['date']).dt.date

    for date, positions in ranked_portfolios.items():
        daily_return = 0
        date = pd.to_datetime(date).date() # Convert Timestamp to date

        # Calculate the daily return for long positions
        for ticker in positions['long']:
            ticker_data = historical_data[(historical_data['date'] == date) & (historical_data['TICKER'] == ticker[0])]
            if not ticker_data.empty:
                daily_return += ticker_data['RET'].iloc[0] - ticker_data['RF'].iloc[0]

        # Calculate the daily return for short positions
        for ticker in positions['short']:
            ticker_data = historical_data[(historical_data['date'] == date) & (historical_data['TICKER'] == ticker[0])]
            if not ticker_data.empty:
                daily_return -= ticker_data['RET'].iloc[0] - ticker_data['RF'].iloc[0]

        portfolio_daily_returns.append(daily_return)

    # Calculating metrics over time
    portfolio_returns = np.array(portfolio_daily_returns)
    avg_return = np.mean(portfolio_returns) if portfolio_returns.size > 0 else 0
    volatility = np.std(portfolio_returns) if portfolio_returns.size > 0 else 0
    sharpe_ratio = avg_return / volatility if volatility != 0 else 0
    max_drawdown = np.min(portfolio_returns) if portfolio_returns.size > 0 else 0

    return {
        "average_return": avg_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }




