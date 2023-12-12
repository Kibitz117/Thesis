import torch
import pandas as pd
from datetime import datetime
import sys
from tqdm import tqdm
sys.path.append('utils')
sys.path.append('models')
sys.path.append('data_func')
from model_factory import ModelFactory
from data_helper_functions import create_study_periods,create_tensors
from torch.utils.data import DataLoader, TensorDataset
def evaluate_model_performance(model_state_path, train_test_splits, device, model_name, target_type, model_config):
    factory = ModelFactory()
    model, _ = factory.create(model_name, target_type, 'bce', model_config=model_config)  # Loss is not used in evaluation
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.to(device)
    model.eval()

    accuracy_meter = AverageMeter()
    # Additional metrics can be initialized here if needed
    i=0
    for split in tqdm(train_test_splits):
        train_data, train_labels, test_data, test_labels = split

        # Evaluating on training data
        train_accuracy = compute_accuracy(model, train_data, train_labels, device)
        accuracy_meter.update(train_accuracy, train_data.size(0))

        # Evaluating on test data
        test_accuracy = compute_accuracy(model, test_data, test_labels, device)
        accuracy_meter.update(test_accuracy, test_data.size(0))

        # Add additional metrics calculations here if needed
        print(f'Accuracy for Period{i}: {test_accuracy}')
        i+=1

    return {
        "Average Accuracy": accuracy_meter.avg,
        # Add additional metrics here
    }

def compute_accuracy(model, data, labels, device):
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            predictions = torch.sigmoid(outputs).round()
            
            total_correct += (predictions.view(-1) == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = (total_correct / total_samples) * 100  # Convert to percentage
    return accuracy

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_portfolios(model_state_path, study_periods, model_name, target_type, model_config, device, sequence_length=240):
    # Load the model
    factory = ModelFactory()
    model, _ = factory.create(model_name, target_type, 'bce', model_config=model_config)
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.to(device)
    model.eval()

    portfolio_returns = []

    for in_sample_df, out_of_sample_df in study_periods:
        _, _, test_data, test_labels, _ = process_window(in_sample_df, out_of_sample_df, sequence_length)

        # Ensure data fits in memory, else consider batching
        test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs, _ = model(test_data_tensor)
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Assign predictions to out_of_sample_df
        out_of_sample_df['model_prediction'] = predictions
        out_of_sample_df['rank'] = out_of_sample_df.groupby('date')['model_prediction'].rank(ascending=False)

        # Go long on top 10 and short on bottom 10 each day
        long_stocks = out_of_sample_df[out_of_sample_df['rank'] <= 10]
        short_stocks = out_of_sample_df[out_of_sample_df['rank'] > len(out_of_sample_df) - 10]

        # Calculate portfolio returns
        daily_returns = (long_stocks.groupby('date')['RET'].mean() - short_stocks.groupby('date')['RET'].mean()).reset_index()
        daily_returns.columns = ['date', 'portfolio_return']
        portfolio_returns.append(daily_returns)

    # Combine returns from all periods
    combined_returns = pd.concat(portfolio_returns)

    return combined_returns




class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# Load data
target = 'cross_sectional_median'
df = pd.read_csv('data/crsp_ff_adjusted.csv')
df['date'] = pd.to_datetime(df['date'])
df.dropna(subset=['RET'], inplace=True)
df = df.drop(columns='Unnamed: 0')
#subset df to 2014-2015
# df = df[df['date'] >= datetime(2014, 1, 1)]

# Create tensors
study_periods = create_study_periods(df, n_periods=23, window_size=240, trade_size=250, train_size=750, forward_roll=250, 
                                        start_date=datetime(1990, 1, 1), end_date=datetime(2015, 12, 31), target_type=target)
train_test_splits, task_types = create_tensors(study_periods)

# # Example Usage
# model_state_path = 'model_state_dict.pth'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Specify the model configuration
# model_name = 'transformer'  # Replace with your model's name if different
# target_type = 'classification'  # or 'regression', based on your model's task
# model_config = {
#     'd_model': 16,  # Update these parameters based on your model's configuration
#     'num_heads': 4,
#     'd_ff': 32,
#     'num_encoder_layers': 1,
#     'dropout': 0.1,
# }

# # Ensure train_test_splits is defined and loaded as per your dataset
# # train_test_splits = [...]

# performance_stats = evaluate_model_performance(model_state_path, train_test_splits, device, model_name, target_type, model_config)
# print(performance_stats)

# Example Usage
model_state_path = 'model_state_dict.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'transformer'
target_type = 'classification'
model_config = {
    'd_model': 16,
    'num_heads': 4,
    'd_ff': 32,
    'num_encoder_layers': 1,
    'dropout': 0.1,
}

# Assuming study_periods is already defined as per your dataset
portfolio_returns = create_portfolios(model_state_path, study_periods, model_name, target_type, model_config, device)
print(portfolio_returns)