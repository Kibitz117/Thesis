import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import math
import copy
import os
from datetime import datetime
sys.path.append('utils')
sys.path.append('models')
sys.path.append('data_func')
from data_helper_functions import create_study_periods_parallel,create_tensors
from model_factory import ModelFactory
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
from pandas import Timestamp
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def selective_train_and_save_model(model, criterion,optimizer, periods_folder, device, model_config={}, model_save_path="best_model.pth", pretrain_epochs=5, initial_reward=2.0, num_classes=3, n_periods=None):
    if optimizer==None:
        optimizer = optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0001)
    # optimizer = optim.RMSprop(model.parameters(), lr=0.009, alpha=0.99, eps=1e-08, weight_decay=0)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    n_epochs = 1000
    patience = 150
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    best_val_loss=float('inf')
    if device.type == 'cuda' or device.type == 'mps':
        num_workers = 8
        batch_size = 128
    else:
        num_workers = 0  # Adjust based on your environment
        batch_size = 16

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        total_samples = 0
      
    # Set n_periods to the number of matching files
        if n_periods==None:
            files = os.listdir(periods_folder)
            period_files = [file for file in files if file.startswith("period_")]
            n_periods = len(period_files)
        # for period_index in tqdm(range(n_periods)):
        for period_index in tqdm(range(0,10)):
            file_path = os.path.join(periods_folder, f'period_{period_index}.pt')
            period_split = torch.load(file_path, map_location='cpu')

            train_data, train_labels,_,_ = period_split['train']
            test_data, test_labels,_,_ = period_split['val']

            # Load data onto the CPU and move to GPU as needed
            train_dataset = TensorDataset(train_data.clone().detach().to(dtype=torch.float32), train_labels.clone().detach().to(dtype=torch.long))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

             # Sequentially select a portion of the validation data
            val_dataset = TensorDataset(test_data.clone().detach().to(dtype=torch.float32), test_labels.clone().detach().to(dtype=torch.long))

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(data)

                if epoch < pretrain_epochs:
                    loss = criterion(outputs, labels) 
                     # Use all outputs for loss computation before abstention
                else:
                    
                    # Adjust loss computation for abstention mechanism
                    outputs = F.softmax(outputs, dim=1)
                    class_outputs, abstention_output = outputs[:, :-1], outputs[:, -1]
                    gain = torch.gather(class_outputs, 1, labels.unsqueeze(1)).squeeze()
                    doubling_rate = (gain + abstention_output.div(initial_reward)).log()
                    loss = -doubling_rate.mean()
                    

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        scheduler.step()

        # Validation logic
        if epoch >= pretrain_epochs:
            # Replace validate_with_abstention with your actual validation function that supports abstention
            current_val_loss = validate_with_abstention(model, val_loader, criterion, device, initial_reward)
            # initial_reward = adjust_reward(current_val_loss, best_val_loss, initial_reward)
        
        else:
            # Replace validate_without_abstention with your actual validation function for the initial phase
            current_val_loss = validate_without_abstention(model, val_loader, criterion, device)

        # Update best validation loss and model state
        if current_val_loss < best_val_loss and epoch>=pretrain_epochs:
            best_val_loss = current_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        elif epoch>=pretrain_epochs:
            counter += 1

        if counter >= patience:
            print('Early stopping!')
            break

        # Logging for each epoch
        print(f'Epoch {epoch+1}: Train Loss: {total_train_loss / total_samples:.4f}, Current Val Loss: {current_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, Current Reward: {initial_reward:.2f}')

    # Save the best model state
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f'Best model saved to {model_save_path}')
    
    return model


# Helper functions for validation with and without abstention
def validate_with_abstention(model, val_loader, criterion, device, reward):
    model.eval()
    total_val_loss = 0.0
    total_val_correct = 0
    total_val_samples = 0
    total_val_abstained = 0

    with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)

                outputs = F.softmax(outputs, dim=1)
                class_outputs, abstention_output = outputs[:, :-1], outputs[:, -1]
                gain = torch.gather(class_outputs, dim=1, index=labels.unsqueeze(1)).squeeze()
                doubling_rate = (gain.add(abstention_output.div(reward))).log()
                loss = -doubling_rate.mean()

                # is_abstained = (abstention_output > gain / reward).float()
                # total_val_abstained += is_abstained.sum().item()
                
                # Adjust loss for non-abstained samples
                # non_abstained_mask = ~is_abstained.bool()
                # effective_loss = loss * non_abstained_mask
                total_val_loss += loss.item() * data.size(0) # Sum up effective loss

                # predictions = torch.argmax(class_outputs, dim=1)
                # correct = (predictions == labels) & non_abstained_mask
                # total_val_correct += correct.sum().item()

                total_val_samples += data.size(0)  # Total samples including abstained

    average_val_loss = total_val_loss / (total_val_samples ) if total_val_samples > 0 else 0
    # val_accuracy = total_val_correct / (total_val_samples - total_val_abstained) if total_val_samples > total_val_abstained else 0
    # val_abstain_rate=total_val_abstained/total_val_samples if total_val_samples > 0 else 0

    return average_val_loss


def adjust_reward(current_val_loss, best_val_loss, initial_reward, adjustment_factor=0.1):
    """
    Adjusts the reward for abstention based on the change in validation loss.
    
    :param current_val_loss: The current epoch's validation loss.
    :param best_val_loss: The best validation loss observed so far.
    :param initial_reward: The current reward for abstention.
    :param adjustment_factor: How much to adjust the reward.
    :return: The adjusted reward.
    """
    if current_val_loss < best_val_loss:
        # If the model is improving, increase the reward to encourage more confident predictions
        new_reward = initial_reward * (1 + adjustment_factor)
    else:
        # If the model's performance worsens, decrease the reward to encourage more abstention
        new_reward = initial_reward * (1 - adjustment_factor)
    
    return max(new_reward, 1.0)  # Ensure the reward does not go below 1

def validate_without_abstention(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    total_val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)

                # Use only the class output (excluding abstention output)
                class_outputs = outputs[:, :-1]
                loss = criterion(class_outputs, labels)

                total_val_loss += loss.item() * data.size(0)
                
                # predictions = torch.argmax(class_outputs, dim=1)
                # total_val_correct += (predictions == labels).sum().item()
                total_val_samples += labels.size(0)

    average_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
    # val_accuracy = total_val_correct / total_val_samples if total_val_samples > 0 else 0

    return average_val_loss











def load_data(path,  start):
    # Check the file extension to decide the reading method
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format")

    # Proceed with your data processing...
    df['date'] = pd.to_datetime(df['date'])
    # Filter based on the start date and features...
    df = df[df['date'] >= start]

    # Assuming start_date and end_date are defined somewhere...
    start_date = df['date'].min()
    end_date = df['date'].max()

    return df, start_date, end_date


# def main():
    # #Parameters
    # model_name = 'transformer'
    # #cross_sectional_median, raw_returns, buckets
    # target = 'buckets'
    # num_classes=3
    # data_type='RET'
    # #Fama french factors might be big deal for learning (are wavelets too?)
    # #See how to use RET, and fama french factors at once 
    # #extra features:'Mkt-RF','SMB','HML','RF'
    # features=['RET','RF']
    # selective=False
    # sequence_length=240
    # if target=='cross_sectional_median' or target=='direction' or target=='cross_sectional_mean' or target=='buckets':
    #     loss_func = 'ce'
    # else:
    #     loss_func = 'mse'
    # model_config={
    #     'd_model': 128,
    #     'num_heads': 8,
    #     'd_ff': 256,
    #     'num_encoder_layers': 2,
    #     'dropout': .1,

    # }
    #  # Check if CUDA, MPS, or CPU should be used
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # # Check for MPS (Apple Silicon)
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # # Fallback to CPU
    # else:
    #     device = torch.device("cpu")

    # print("Using device:", device)




    # # path='data/crsp_ff_adjusted.csv'
    # # path='data/merged_data.csv'
    # path='data/stock_data_with_factors.csv'
    # # path='data/modern_stock_data.csv'
    # # path='data/corrected_crsp_ff_adjusted.csv'
    # start=datetime(2012,1,1)
    # df,start_date,end_date=load_data(path,features,start)
    # df = df[df['TICKER'].isin(['AAPL','MSFT','AMZN','GOOG','IBM'])]
    # #Wavelet transform signifigantly improves learning
    # study_periods = create_study_periods(df, window_size=240, trade_size=250, train_size=750, forward_roll=250, 
    #                                      start_date=start_date, end_date=end_date, target_type=target,data_type=data_type,apply_wavelet_transform=True)
    # columns=study_periods[0][0].columns.to_list()
    # feature_columns = [col for col in columns if col not in ['date', 'TICKER', 'target']]
    # model_config['input_features']=len(feature_columns)
    # if len(study_periods)>10:
    #     n_jobs=10
    # else:
    #     n_jobs=len(study_periods)
    # train_test_splits, ticker_tensor_mapping,task_type = create_tensors(study_periods,n_jobs,sequence_length,feature_columns)


    

    # if selective==True:
    #     model = selective_train_and_save_model(model_name, task_type,loss_func, train_test_splits, device,model_config,num_classes=num_classes)
    #     #Test method
    # else:
    #     model=train_and_save_model(model_name, task_type,loss_func, train_test_splits, device,model_config,num_classes=num_classes)
    #     #Test method
    # #export model config
    # torch.save(model.state_dict(), 'model_state_dict.pth')


# if __name__ == "__main__":
#     main()
