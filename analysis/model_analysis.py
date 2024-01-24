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

def generate_predictions(model, test_data, device, k=10):
    model.eval()
    predictions_by_date = {}

    with torch.no_grad():
        for ticker, date, data, _ in test_data:
            data_tensor = data.to(device)
            logit = model(data_tensor.unsqueeze(0))  # Get logit
            probability = torch.sigmoid(logit).item()  # Convert logit to probability

            # Organize probabilities by date and then by ticker
            if date not in predictions_by_date:
                predictions_by_date[date] = {}
            predictions_by_date[date][ticker] = probability

    # Ranking and selecting top and bottom k stocks
    ranked_portfolios = rank_stocks_by_probability(predictions_by_date, k)
    return ranked_portfolios

def rank_stocks_by_probability(predictions_by_date, k):
    ranked_portfolios = {}
    for date, probs in predictions_by_date.items():
        sorted_tickers = sorted(probs, key=probs.get, reverse=True)
        long_positions = sorted_tickers[:k]
        short_positions = sorted_tickers[-k:]

        ranked_portfolios[date] = {'long': long_positions, 'short': short_positions}
    return ranked_portfolios
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




def analyze_portfolio_performance(historical_data, ranked_portfolios):
    portfolio_daily_returns = []
    historical_data['date'] = pd.to_datetime(historical_data['date'])

    for date, positions in ranked_portfolios.items():
        daily_return = 0

        # Calculate the daily return for long positions
        for ticker in positions['long']:
            if (ticker in historical_data['TICKER'].values) and (date in historical_data['date'].values):
                daily_return += historical_data[(historical_data['date'] == date) & (historical_data['TICKER'] == ticker)]['RET'].iloc[0]

        # Calculate the daily return for short positions
        for ticker in positions['short']:
            if (ticker in historical_data['TICKER'].values) and (date in historical_data['date'].values):
                daily_return -= historical_data[(historical_data['date'] == date) & (historical_data['TICKER'] == ticker)]['RET'].iloc[0]

        portfolio_daily_returns.append(daily_return)

    # Calculating metrics over time
    portfolio_returns = np.array(portfolio_daily_returns)
    avg_return = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    sharpe_ratio = avg_return / volatility if volatility != 0 else 0
    max_drawdown = np.min(portfolio_returns)  # Simplistic approach

    return {
        "average_return": avg_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }



