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
            date=pd.to_datetime(date).date()
            date = date.isoformat()
            data_tensor = data.float().to(device)
            outputs = model(data_tensor.unsqueeze(0))
            outputs = torch.softmax(outputs, dim=1)
            class_outputs, reservation = outputs[:, :-1], outputs[:, -1]

            predicted_label = torch.argmax(class_outputs, dim=1).item()
            correct_prediction = (predicted_label == label.item())

            # Determine if stock is long or short based on class probabilities
            is_long = class_outputs[0, 0] < class_outputs[0, 1]
            if is_long:
                score=class_outputs[0,1]
            else:
                score=class_outputs[0,0]
            
            all_predictions.append({
                "date": date,
                "TICKER": ticker,
                "label": label.item(),
                "correct_prediction":correct_prediction,
                "predicted_label": predicted_label,
                "reservation": reservation.item(),
                "score":score.item()
            })

            confusion_matrix[label, predicted_label] += 1

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df = predictions_df.sort_values(by='date')
    date_name=predictions_df['date'].min()
    predictions_df.to_csv(f'period_{date_name}_predictions')

    # Analyze misclassification rates and abstention scores
    misclassification_rates, correlation_coefficient = analyze_metrics(confusion_matrix, predictions_df)

    mean_accuracy = predictions_df['correct_prediction'].mean()
    k_values=[10]
    daily_confidence_portfolios,daily_k_portfolio,portfolio_accuracies  = calculate_daily_portfolios(predictions_df, k_values)

    # Calculate overall mean accuracy considering both long and short
    #Mean top k flop k accuracy
    # mean_abstention_accuracy = (mean_long_accuracy + mean_short_accuracy) / 2

    return daily_confidence_portfolios[10],daily_k_portfolio[10],portfolio_accuracies, mean_accuracy, misclassification_rates, correlation_coefficient

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



def calculate_daily_portfolios(predictions_df, k_values, weights=None):
    # Initialize containers for confidence and k portfolios
    daily_confidence_portfolios = {k: {} for k in k_values}
    daily_k_portfolios = {k: {} for k in k_values}
    portfolio_accuracies = {
        'high_confidence_long_accuracy': {k: [] for k in k_values},
        'high_confidence_short_accuracy': {k: [] for k in k_values},
        'top_k_long_accuracy': {k: [] for k in k_values},
        'flop_k_short_accuracy': {k: [] for k in k_values},
        'high_confidence_long_misclassification': {k: [] for k in k_values},
        'high_confidence_short_misclassification': {k: [] for k in k_values},
        'top_k_long_misclassification': {k: [] for k in k_values},
        'flop_k_short_misclassification': {k: [] for k in k_values},
        'correlation_scores_reservation': [],
        'correlation_scores_score': []
    }

    # Calculate weights
    # reservation_weight = weights[0]
    # score_weight = weights[1]
    #Weights from training reservation .5 score .5
    reservation_weight = 0.2 #0.42261430123201404
    score_weight = 0.8 #0.4370459627865435

    for date, group in tqdm(predictions_df.groupby('date')):
        # Calculate and append correlations
        correlation_reservation = group['reservation'].corr(group['correct_prediction'])
        correlation_score = group['score'].corr(group['correct_prediction'])

        portfolio_accuracies['correlation_scores_reservation'].append(correlation_reservation)

        portfolio_accuracies['correlation_scores_score'].append(correlation_score)

        group['reservation_z'] = (group['reservation'] - group['reservation'].mean()) / group['reservation'].std()
        group['score_z'] = (group['score'] - group['score'].mean()) / group['score'].std()

        # Create a combined score for the group
        group['combined_score'] = (score_weight * group['score_z']) - (reservation_weight * group['reservation_z'])
        portfolio_accuracies['correlation_scores_reservation'].append(correlation_reservation)
        # Sort by combined score for high confidence portfolios
        group_sorted = group.sort_values(by='combined_score', ascending=False)
        # group_sorted=group_sorted[group_sorted['reservation_z']>abs(.8)]

        # Loop through k values
        for k in k_values:
            # high_confidence_long_stocks=group_sorted[group_sorted['reservation_z']>1].sort_values(by='score_z', ascending=False)
            # high_confidence_short_stocks=group_sorted[group_sorted['reservation_z']<-1].sort_values(by='score_z', ascending=False)
            top_k_long_stocks = group_sorted[group_sorted['predicted_label'] == 1].head(k)
            top_k_short_stocks = group_sorted[group_sorted['predicted_label'] == 0].head(k)

            # Append accuracy and misclassification rate for high confidence long and short portfolios
            portfolio_accuracies['high_confidence_long_accuracy'][k].append(top_k_long_stocks['correct_prediction'].mean() if not top_k_long_stocks.empty else np.nan)
            portfolio_accuracies['high_confidence_short_accuracy'][k].append(top_k_short_stocks['correct_prediction'].mean() if not top_k_short_stocks.empty else np.nan)
            portfolio_accuracies['high_confidence_long_misclassification'][k].append((1 - (top_k_long_stocks['correct_prediction'].mean())) if not top_k_long_stocks.empty else np.nan)
            portfolio_accuracies['high_confidence_short_misclassification'][k].append((1 - (top_k_short_stocks['correct_prediction'].mean())) if not top_k_short_stocks.empty else np.nan)

            # Store high confidence portfolios for each k
            daily_confidence_portfolios[k][date] = {
                "long": top_k_long_stocks,
                "short": top_k_short_stocks
            }

            # Store top k long and short portfolios sorted by score
            top_k_long_stocks_by_score = group[group['predicted_label'] == 1].sort_values(by='score', ascending=False).head(k)
            top_k_short_stocks_by_score = group[group['predicted_label'] == 0].sort_values(by='score', ascending=False).head(k)
            
            daily_k_portfolios[k][date] = {
                "long": top_k_long_stocks_by_score,
                "short": top_k_short_stocks_by_score
            }

            # Append accuracy and misclassification rate for top k long and short portfolios
            portfolio_accuracies['top_k_long_accuracy'][k].append(top_k_long_stocks_by_score['correct_prediction'].mean() if not top_k_long_stocks_by_score.empty else np.nan)
            portfolio_accuracies['flop_k_short_accuracy'][k].append(top_k_short_stocks_by_score['correct_prediction'].mean() if not top_k_short_stocks_by_score.empty else np.nan)
            portfolio_accuracies['top_k_long_misclassification'][k].append((1 - (top_k_long_stocks_by_score['correct_prediction'].mean())) if not top_k_long_stocks_by_score.empty else np.nan)
            portfolio_accuracies['flop_k_short_misclassification'][k].append((1 - (top_k_short_stocks_by_score['correct_prediction'].mean())) if not top_k_short_stocks_by_score.empty else np.nan)

    # Average the accuracy metrics and correlations across all dates
    for k in k_values:
        portfolio_accuracies['high_confidence_long_accuracy'][k] = np.nanmean(portfolio_accuracies['high_confidence_long_accuracy'][k])
        portfolio_accuracies['high_confidence_short_accuracy'][k] = np.nanmean(portfolio_accuracies['high_confidence_short_accuracy'][k])
        portfolio_accuracies['high_confidence_long_misclassification'][k] = np.nanmean(portfolio_accuracies['high_confidence_long_misclassification'][k])
        portfolio_accuracies['high_confidence_short_misclassification'][k] = np.nanmean(portfolio_accuracies['high_confidence_short_misclassification'][k])
        portfolio_accuracies['top_k_long_accuracy'][k] = np.nanmean(portfolio_accuracies['top_k_long_accuracy'][k])
        portfolio_accuracies['flop_k_short_accuracy'][k] = np.nanmean(portfolio_accuracies['flop_k_short_accuracy'][k])
        portfolio_accuracies['top_k_long_misclassification'][k] = np.nanmean(portfolio_accuracies['top_k_long_misclassification'][k])
        portfolio_accuracies['flop_k_short_misclassification'][k] = np.nanmean(portfolio_accuracies['flop_k_short_misclassification'][k])

    portfolio_accuracies['overall_correlation_reservation'] = np.nanmean(portfolio_accuracies['correlation_scores_reservation'])
    portfolio_accuracies['overall_correlation_score'] = np.nanmean(portfolio_accuracies['correlation_scores_score'])

    return daily_confidence_portfolios, daily_k_portfolios, portfolio_accuracies




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
            positions = portfolio[side]['TICKER'].tolist() 
            
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






