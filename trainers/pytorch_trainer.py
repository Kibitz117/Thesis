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
sys.path.append('../models')
sys.path.append('../data_func')
from data_helper_functions import create_study_periods,create_tensors
from model_factory import create

import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def train_and_save_model(model_name, task_type, loss_name, train_test_splits, device, model_save_path="best_model.pth"):
    factory = ModelFactory()
    model, criterion = factory.create(model_name, task_type, loss_name)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 1000
    patience = 5
    best_loss = np.inf
    counter = 0

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        total_val_loss = 0.0

        for train_data, train_labels, val_data, val_labels in tqdm(train_test_splits):
            train_dataset = TensorDataset(train_data, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            val_dataset = TensorDataset(val_data, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            # Train step
            train_loss = 0.0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
            total_train_loss += train_loss / len(train_loader.dataset)

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * data.size(0)
            total_val_loss += val_loss / len(val_loader.dataset)

        average_train_loss = total_train_loss / len(train_test_splits)
        average_val_loss = total_val_loss / len(train_test_splits)
        
        print(f'Epoch {epoch+1}/{n_epochs}, '
              f'Average Train Loss: {average_train_loss:.4f}, '
              f'Average Validation Loss: {average_val_loss:.4f}')

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1

        if counter == patience:
            print('Early stopping!')
            break

    best_model_state = torch.load(model_save_path, map_location=device)
    model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, data_splits, k, device):
    in_sample_long_portfolios = pd.DataFrame()
    out_of_sample_long_portfolios = pd.DataFrame()
    in_sample_short_portfolios = pd.DataFrame()
    out_of_sample_short_portfolios = pd.DataFrame()
    
    for train_data, train_labels, val_data, val_labels in tqdm(data_splits):
        train_probs, val_probs, train_df, val_df = model.get_predictions(train_data, train_labels, val_data, val_labels, device)
        in_sample_long, in_sample_short = model.create_portfolios(train_df, train_probs, k)
        out_of_sample_long, out_of_sample_short = model.create_portfolios(val_df, val_probs, k)
        
        in_sample_long_portfolios = pd.concat([in_sample_long_portfolios, in_sample_long])
        in_sample_short_portfolios = pd.concat([in_sample_short_portfolios, in_sample_short])
        out_of_sample_long_portfolios = pd.concat([out_of_sample_long_portfolios, out_of_sample_long])
        out_of_sample_short_portfolios = pd.concat([out_of_sample_short_portfolios, out_of_sample_short])
    
    return in_sample_long_portfolios, in_sample_short_portfolios, out_of_sample_long_portfolios, out_of_sample_short_portfolios

def main():
    model_name = 'transformer'
    target = 'cross_sectional_median'

    # Load data
    df = pd.read_csv('../data/crsp_ff_adjusted.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.dropna(subset=['RET'], inplace=True)
    df = df.drop(columns='Unnamed: 0')
    
    # Create tensors
    study_periods = create_study_periods(df, n_periods=23, window_size=240, trade_size=250, train_size=750, forward_roll=250, 
                                         start_date=datetime(1990, 1, 1), end_date=datetime(2015, 12, 31), target_type=target)
    train_test_splits, task_types = create_tensors(study_periods)

    # Check if CUDA, MPS, or CPU should be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if target=='cross_sectional_median':
        loss_func = 'nll'
    else:
        loss_func = 'mse'
    #TODO:Add support for selective loss and sharpe loss
    model = train_and_save_model(model_name, task_types[0],loss_func, train_test_splits, device)

    in_sample_long_portfolios, in_sample_short_portfolios, out_of_sample_long_portfolios, out_of_sample_short_portfolios = evaluate_model(model, train_test_splits, k=10, device=device)

    # Export portfolios
    in_sample_long_portfolios.to_csv(f'../data/{model_name}_results/in_sample_long_portfolios.csv')
    in_sample_short_portfolios.to_csv(f'../data/{model_name}_results/in_sample_short_portfolios.csv')
    out_of_sample_long_portfolios.to_csv(f'../data/{model_name}_results/out_of_sample_long_portfolios.csv')
    out_of_sample_short_portfolios.to_csv(f'../data/{model_name}_results/out_of_sample_short_portfolios.csv')

if __name__ == "__main__":
    main()
