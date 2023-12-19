import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
import math
from datetime import datetime
#import relu
from torch.nn import functional as F
sys.path.append('models')
sys.path.append('utils')
sys.path.append('data_func')
# from sharpe_loss import SharpeLoss
from data_helper_functions import create_study_periods,create_tensors
from transformer_model import TimeSeriesTransformer,ScaledMultiHeadAttention

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)  # for reproducibility

def generate_individual_stock_data(num_days=1000, pattern='sine', start_price=100):
    if pattern == 'random':
        price_changes = np.random.randn(num_days) * 0.5 + 0.1  # daily returns
        prices = np.cumsum(price_changes) + start_price  # price series
    elif pattern == 'sine':
        x = np.linspace(0, 20, num_days)
        prices = np.sin(x) * 10 + start_price
    elif pattern == 'exponential':
        x = np.linspace(0, 5, num_days)
        prices = np.exp(x) + start_price

    return prices

def process_stock_data(prices, sequence_length=250):
    log_returns = np.log(prices[1:] / prices[:-1])
    data = []
    labels = []
    for i in range(len(log_returns) - sequence_length):
        data.append(log_returns[i:i + sequence_length])
        labels.append(log_returns[i + sequence_length])
    return np.array(data), np.array(labels)

def standardize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std

def apply_standardization_to_data(data, mean, std):
    return (data - mean) / std

# Set parameters
num_days = 1000
sequence_length = 250
train_period = 750

# Patterns
patterns = ['random', 'sine', 'exponential']

# Generate data for each pattern
stocks_data = []
for pattern in patterns:
    prices = generate_individual_stock_data(num_days=num_days, pattern=pattern, start_price=100)
    stock_data, stock_labels = process_stock_data(prices, sequence_length=sequence_length)
    stocks_data.append((stock_data[:train_period], stock_labels[:train_period],
                        stock_data[train_period-sequence_length:],
                        stock_labels[train_period-sequence_length:]))


train_data_flat = np.concatenate([data[0] for data in stocks_data]).flatten()
train_data_standardized, mean, std = standardize_data(train_data_flat)

# Apply standardization to each split using the calculated mean and std
stocks_data_standardized = []
for stock_data, stock_labels, test_data, test_labels in stocks_data:
    stock_data = apply_standardization_to_data(stock_data, mean, std)
    test_data = apply_standardization_to_data(test_data, mean, std)
    stocks_data_standardized.append((stock_data, stock_labels, test_data, test_labels))

# Prepare the final datasets with standardized data
train_data_combined = np.concatenate([data[0] for data in stocks_data_standardized], axis=0)
train_labels_combined = np.concatenate([data[1] for data in stocks_data_standardized], axis=0)
test_data_combined = np.concatenate([data[2] for data in stocks_data_standardized], axis=0)
test_labels_combined = np.concatenate([data[3] for data in stocks_data_standardized], axis=0)

# Convert to tensors
train_data_tensor = torch.tensor(train_data_combined, dtype=torch.float32).unsqueeze(-1)
train_labels_tensor = torch.tensor(train_labels_combined, dtype=torch.float32)
test_data_tensor = torch.tensor(test_data_combined, dtype=torch.float32).unsqueeze(-1)
test_labels_tensor = torch.tensor(test_labels_combined, dtype=torch.float32)

train_test_splits = [(train_data_tensor, train_labels_tensor, test_data_tensor, test_labels_tensor)]

task_types=['regression']

# Check if CUDA, MPS, or CPU should be used
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("mps")

print("Using device:", device)
best_model_path = "best_model.pth" 
model = TimeSeriesTransformer(d_model=32, num_heads=4, d_ff=128, num_encoder_layers=1, 
                               dropout=.1,task_type=task_types[0]).to(device)
# model = TimeSeriesTransformerModel(config)

# Loss depends on target, MAE for returns, Cross Entropy for above/below cross-sectional median. Also have selective loss in utils
if task_types[0] == 'classification':
    criterion = nn.NLLLoss()
else:
    criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
patience = 5
best_loss = np.inf
counter = 0
batch_size=64
for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0

    for train_data, train_labels, val_data, val_labels in tqdm(train_test_splits):
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        #reshape to batch,sequence,d_model
        
        # Access the batch size from the train_loader
        batch_size = train_loader.batch_size
        sequence_length = train_loader.dataset.tensors[0].size(1)

        train_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            mask = TimeSeriesTransformer.create_lookahead_mask(data.size(1)).to(data.device)
            outputs,_ = model(data,mask)
            if task_types[0] == 'classification':
                labels = labels.long()  # Adjusted here to use the look-ahead mask
            loss = criterion(outputs, labels)  # Adjust based on your specific use case
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        total_train_loss += train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                mask = TimeSeriesTransformer.create_lookahead_mask(data.size(1)).to(data.device)
                outputs,_ = model(data,mask)
                if task_types[0] == 'classification':
                    labels = labels.long() # Adjusted here to use the look-ahead mask
                loss = criterion(outputs, labels)  # Adjust based on your specific use case
                val_loss += loss.item() * data.size(0)

        total_val_loss += val_loss / len(val_loader.dataset)

    average_train_loss = total_train_loss / len(train_test_splits)
    average_val_loss = total_val_loss / len(train_test_splits)
    
    print(f'Epoch {epoch+1}/{n_epochs}, '
          f'Average Train Loss: {average_train_loss:.4f}, '
          f'Average Validation Loss: {average_val_loss:.4f}')

    if average_val_loss < best_loss:
        best_loss = average_val_loss
        torch.save(model.state_dict(), best_model_path)
        counter = 0
    else:
        counter += 1

    if counter == patience:
        print('Early stopping!')
        break

best_model_state = torch.load(best_model_path, map_location=device)
model.load_state_dict(best_model_state)
