import torch
import pandas as pd
from datetime import datetime
import sys
from tqdm import tqdm
import json
import numpy as np
sys.path.append('utils')
sys.path.append('models')
sys.path.append('data_func')
from model_factory import ModelFactory
from data_helper_functions import create_study_periods_parallel,create_tensors
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F











import numpy as np
import torch
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

def generate_portfolio_with_simplified_metrics(test_data, model, device, k, num_classes):
    model.eval()
    all_predictions = []
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for ticker, date, data, label in tqdm(test_data):
            data_tensor = data.float().to(device)
            outputs = model(data_tensor.unsqueeze(0))
            outputs = torch.softmax(outputs, dim=1)
            class_outputs, reservation = outputs[:, :-1], outputs[:, -1]

            predicted_label = torch.argmax(class_outputs, dim=1).item()
            correct_prediction = (predicted_label == label.item())

            # Determine if stock is long or short based on class probabilities
            is_long = class_outputs[0, 0] < class_outputs[0, 1]

            all_predictions.append({
                "date": pd.to_datetime(date).date(),
                "ticker": ticker,
                "label": label.item(),
                "predicted_label": predicted_label,
                "reservation": reservation.item(),
                "is_long": is_long,
            })

            confusion_matrix[label, predicted_label] += 1

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df = predictions_df.sort_values(by='date')

    # Analyze misclassification rates and abstention scores
    misclassification_rates, correlation_coefficient = analyze_metrics(confusion_matrix, predictions_df)

    mean_accuracy = predictions_df['correct_prediction'].mean()
    daily_portfolios, mean_long_accuracy, mean_short_accuracy = calculate_daily_portfolios(predictions_df, k)

    # Calculate overall mean accuracy considering both long and short
    #Mean top k flop k accuracy
    mean_abstention_accuracy = (mean_long_accuracy + mean_short_accuracy) / 2

    return daily_portfolios, mean_accuracy, misclassification_rates, mean_abstention_accuracy, correlation_coefficient

def analyze_metrics(confusion_matrix, predictions_df):
    num_classes = confusion_matrix.size(0)
    misclassification_rates = {}
    for i in range(num_classes):
        total_predictions = confusion_matrix[i, :].sum().item()
        correct_predictions = confusion_matrix[i, i].item()
        misclassification_rates[i] = 1 - correct_predictions / total_predictions if total_predictions > 0 else 0
    
    predictions_df['incorrect_prediction'] = (~predictions_df['correct_prediction']).astype(int)
    correlation_coefficient, _ = pearsonr(predictions_df['reservation'], predictions_df['incorrect_prediction'])
    # mean_abstention_correct = predictions_df[predictions_df['correct_prediction'] == True]['reservation'].mean()
    # mean_abstention_incorrect = predictions_df[predictions_df['correct_prediction'] == False]['reservation'].mean()

    return misclassification_rates, correlation_coefficient

def calculate_daily_portfolios(predictions_df, k):
    daily_portfolios = {}
    long_accuracies = []
    short_accuracies = []

    for date, group in tqdm(predictions_df.groupby('date')):
        # Separate groups for long and short stocks
        group['is_long'] = group['is_long'].apply(lambda x: x.item())
        long_stocks = group[group['is_long']]
        short_stocks = group[~group['is_long']]

        # Sort and select top k for long and short based on weighted score
        top_long_stocks = long_stocks.sort_values(by='weighted_score', ascending=False).head(k)
        top_short_stocks = short_stocks.sort_values(by='weighted_score', ascending=False).head(k)

        # Store daily portfolio
        daily_portfolios[date] = {
            "long": top_long_stocks.to_dict('records'),
            "short": top_short_stocks.to_dict('records')
        }

        # Calculate accuracy for top k long and short stocks
        long_accuracy = top_long_stocks['correct_prediction'].mean()
        short_accuracy = top_short_stocks['correct_prediction'].mean()

        long_accuracies.append(long_accuracy if not np.isnan(long_accuracy) else 0)
        short_accuracies.append(short_accuracy if not np.isnan(short_accuracy) else 0)

    mean_long_accuracy = np.mean(long_accuracies)
    mean_short_accuracy = np.mean(short_accuracies)

    return daily_portfolios, mean_long_accuracy, mean_short_accuracy


import pandas as pd
import numpy as np
  # Assuming this is already imported

def analyze_portfolio_performance(df, daily_portfolios, strategy='equal_weight'):
    # Convert 'date' to datetime and set it with 'TICKER' as a multi-index
    df=df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['date', 'TICKER'], inplace=True)
    
    # Initialize a DataFrame to hold portfolio daily returns
    portfolio_daily_returns = pd.Series(dtype='float')
    
    # Iterate through daily_portfolios, which is a dict with dates as keys
    for date, portfolio in tqdm(daily_portfolios.items()):
        date = pd.to_datetime(date)  # Convert date to pandas datetime, matching df's index
        daily_return = 0
        
        for side in ['long', 'short']:
            # Extract tickers from the portfolio for the given side
            positions = [p['ticker'] for p in portfolio[side]]
            
            if positions:  # Ensure there are positions to process
                # Default weights to equal weight if no specific strategy is defined
                weights = np.array([1 / len(positions)] * len(positions))
                
                # Retrieve returns for the positions, if they exist
                try:
                    positions_returns = df.loc[(slice(date, date), positions), 'RET']
                    adjusted_returns = positions_returns.groupby(level='TICKER').mean() * weights
                    # Sum adjusted returns; invert if short
                    daily_return += adjusted_returns.sum() if side == 'long' else -adjusted_returns.sum()
                except KeyError:
                    # Handle the case where no matching data is found
                    pass
        
        # Record the daily return
        if daily_return != 0:
            portfolio_daily_returns[date] = daily_return
    
    # Ensure the index is in datetime format for compatibility with QuantStats
    portfolio_daily_returns.index = pd.to_datetime(portfolio_daily_returns.index)
    
    # Generate and display the performance report using QuantStats
    
    return portfolio_daily_returns






