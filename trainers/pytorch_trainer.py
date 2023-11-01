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
sys.path.append('../models')
sys.path.append('../data_func')
from data_helper_functions import create_study_periods,create_tensors
from transformer_model import TimeSeriesTransformer

model='transformer'
target='cross_sectional_median'
#load data
df=pd.read_csv('../data/crsp_ff_adjusted.csv')
#drop unamed 0
df['date'] = pd.to_datetime(df['date'])
df.dropna(subset=['RET'],inplace=True)
df=df.drop(columns='Unnamed: 0')
#create tensors
study_periods=create_study_periods(df,n_periods=23,window_size=240,trade_size=250,train_size=750,forward_roll=250,start_date=datetime(1990,1,1),end_date=datetime(2015,12,31),target_type=target)
train_test_splits,task_types=create_tensors(study_periods)

#train model
# Check if CUDA, MPS, or CPU should be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
best_model_path = "best_model.pth" 
model = TimeSeriesTransformer(d_model=64, num_heads=8, d_ff=256, num_encoder_layers=2, 
                               dropout=.1, max_len=240,task_type='classification').to(device)

# Loss depends on target, MAE for returns, Cross Entropy for above/below cross-sectional median. Also have selective loss in utils
if task_types[0] == 'classification':
        criterion = nn.NLLLoss()
else:
    criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 1
patience = 5
best_loss = np.inf
counter = 0

for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0

    for train_data, train_labels, val_data, val_labels in tqdm(train_test_splits):

        # Generate look-ahead masks
        train_mask = ScaledMultiHeadAttention.create_look_ahead_mask(train_data.size(1)).to(device)
        val_mask = ScaledMultiHeadAttention.create_look_ahead_mask(val_data.size(1)).to(device)


        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        train_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data, src_mask=train_mask).squeeze()
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

                outputs = model(data, src_mask=train_mask).squeeze() 
                if task_types[0] == 'classification':
                    labels = labels.long() # Adjusted here to use the look-ahead mask
                loss = criterion(outputs.squeeze(), labels)  # Adjust based on your specific use case
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

#create portfolios
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(d_model=64, num_heads=8, d_ff=256, num_encoder_layers=2, 
                               dropout=.1, max_len=240,task_type='classification')
model.load_state_dict(torch.load('best_model.pth',map_location=torch.device('cpu')) )
model.eval()

in_sample_long_portfolios = pd.DataFrame()
out_of_sample_long_portfolios = pd.DataFrame()

in_sample_short_portfolios = pd.DataFrame()
out_of_sample_short_portfolios = pd.DataFrame()

k = 10  # Number of top assets to select in portfolios

for train_data, train_labels, val_data, val_labels in tqdm(train_test_splits):
    # Here, train_data, val_data are your training and validation data respectively
    
    train_mask = ScaledMultiHeadAttention.create_look_ahead_mask(train_data.size(1))
    val_mask = ScaledMultiHeadAttention.create_look_ahead_mask(val_data.size(1))

    with torch.no_grad():
        train_predictions = model(train_data.to(device), src_mask=train_mask.to(device))
        val_predictions = model(val_data.to(device), src_mask=val_mask.to(device))

        train_probs = torch.softmax(train_predictions, dim=1)[:, 1].cpu().numpy()
        val_probs = torch.softmax(val_predictions, dim=1)[:, 1].cpu().numpy()

    # Assuming you have a dataframe or similar structure to hold the date and TICKER information
    train_df['predicted_prob'] = train_probs
    val_df['predicted_prob'] = val_probs

    # In-Sample Portfolio Construction
    for date in train_df['date'].unique():
        date_data = train_df[train_df['date'] == date].sort_values(by='predicted_prob', ascending=False)
        
        long_tickers = date_data.head(k)
        short_tickers = date_data.tail(k)
        
        in_sample_long_portfolios = pd.concat([in_sample_long_portfolios, long_tickers])
        in_sample_short_portfolios = pd.concat([in_sample_short_portfolios, short_tickers])

    # Out-of-Sample Portfolio Construction
    for date in val_df['date'].unique():
        date_data = val_df[val_df['date'] == date].sort_values(by='predicted_prob', ascending=False)
        
        long_tickers = date_data.head(k)
        short_tickers = date_data.tail(k)
        
        out_of_sample_long_portfolios = pd.concat([out_of_sample_long_portfolios, long_tickers])
        out_of_sample_short_portfolios = pd.concat([out_of_sample_short_portfolios, short_tickers])

#export portfolios
in_sample_long_portfolios.to_csv('../data/transformer_results/in_sample_long_portfolios.csv')
in_sample_short_portfolios.to_csv('../data/transformer_results/in_sample_short_portfolios.csv')
out_of_sample_long_portfolios.to_csv('../data/transformer_results/out_of_sample_long_portfolios.csv')
out_of_sample_short_portfolios.to_csv('../data/transformer_results/out_of_sample_short_portfolios.csv')
