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


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import os
import copy
def sort_by_number(item):
  # Extract the number part (assuming format 'top_XX%')
  return int(item.split('_')[1].strip('%'))


def selective_train_and_save_model(model, criterion, periods_folder, device, initial_reward=2.0, num_classes=3):
    optimizer = optim.Adam(model.parameters(), lr=0.01) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    best_val_loss = float('inf')
    best_model_state = None
    patience =15
    batch_size = 512 if device.type in ['cuda', 'mps'] else 16
    num_workers = 6 if device.type in ['cuda', 'mps'] else 0
    avg_score_weight=[]
    curriculum_folders = [f for f in os.listdir(periods_folder) if os.path.isdir(os.path.join(periods_folder, f))]
    curriculum_folders=sorted(curriculum_folders, key=sort_by_number)
    num_curriculums = len(curriculum_folders)
    final_curriculum_idx = num_curriculums - 1
    avg_res_weight=0
    # Iterate through each curriculum folder
    for curriculum_idx, curriculum_folder in enumerate((curriculum_folders)):
        if curriculum_idx not in {0}:
            continue
        print(f"\nStarting training on curriculum: {curriculum_folder}")
        if curriculum_idx > 0:
            print("Resetting learning rate.")
            for g in optimizer.param_groups:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
                optimizer.lr = .01 
                epoch=0
        
        current_folder_path = os.path.join(periods_folder, curriculum_folder)
        period_files = sorted([f for f in os.listdir(current_folder_path) if f.startswith("period_")])
        
        counter = 0 
        best_val_loss=np.inf # Reset early stopping counter for each curriculum
        
        for epoch in range(1000):  # Placeholder for max epochs, subject to early stopping
            model.train()
            total_train_loss = 0
            total_samples = 0
            
            for period_file in tqdm(period_files, desc="Training"):
                file_path = os.path.join(current_folder_path, period_file)
                period_data = torch.load(file_path, map_location='cpu')
                train_data, train_labels, val_data, val_labels = period_data['train'][0], period_data['train'][1], period_data['val'][0], period_data['val'][1]

                train_dataset = TensorDataset(train_data.to(dtype=torch.float32), train_labels.to(dtype=torch.long))
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

                val_dataset = TensorDataset(val_data.to(dtype=torch.float32), val_labels.to(dtype=torch.long))
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

                for data, labels in train_loader:
                    data, labels = data.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    
                    if  epoch>15 and curriculum_idx==final_curriculum_idx:
                        outputs = F.softmax(outputs, dim=1)
                        class_outputs, abstention_output = outputs[:, :-1], outputs[:, -1]
                        gain = torch.gather(class_outputs, 1, labels.unsqueeze(1)).squeeze()
                        doubling_rate = (gain + abstention_output.div(initial_reward)).log()
                        loss = -doubling_rate.mean()
                    else:
                        loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item() * data.size(0)
                    total_samples += data.size(0)
            #and counter>10
            if  epoch>20 and curriculum_idx==final_curriculum_idx:
                val_loss,reservation_weight,score_weight = validate_with_abstention(model, val_loader,criterion, device, initial_reward)
                avg_res_weight=reservation_weight
                avg_score_weight.append(score_weight)
            else:
                val_loss = validate_without_abstention(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                counter = 0  # Reset counter on improvement
            else:
                counter += 1  # Increment counter if no improvement
            
            print(f'Epoch: {epoch+1}/{1000}, Curriculum: {curriculum_folder}, Train Loss: {total_train_loss / total_samples:.4f}, Val Loss: {val_loss:.4f}')
            
            if counter >= patience and curriculum_idx == final_curriculum_idx:
                break 
            elif counter>=patience and curriculum_idx != final_curriculum_idx:
                print(f"\nNo improvement for {patience} epochs in curriculum: {curriculum_folder}. Moving to next curriculum.") # Breaks from the epoch loop, moving to the next curriculum
                break
    if best_model_state:
        torch.save(best_model_state, "best_model.pth")
        print("\nBest model saved.")
        return model,np.mean(avg_res_weight),np.mean(avg_score_weight)



# Helper functions for validation with and without abstention
def validate_with_abstention(model, val_loader, criterion, device, reward):
    model.eval()
    total_val_loss = 0.0
    total_val_correct = 0
    total_val_samples = 0
    total_val_abstained = 0
    reservation_values = []
    score_values = []

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)

            outputs = F.softmax(outputs, dim=1)
            class_outputs, abstention_output = outputs[:, :-1], outputs[:, -1]
            gain = torch.gather(class_outputs, dim=1, index=labels.unsqueeze(1)).squeeze()
            doubling_rate = (gain.add(abstention_output.div(reward))).log()
            loss = -doubling_rate.mean()

            # Calculate weights
            reservation_values.extend(abstention_output.cpu().numpy().tolist())
            score_values.extend(class_outputs.cpu().numpy().tolist())

            # Accumulate loss
            total_val_loss += loss.item() * data.size(0) 

            total_val_samples += data.size(0)  

        # Calculate the correlation between reservation/score values and targets
        reservation_correlation = np.corrcoef(reservation_values, val_loader.dataset.tensors[1].cpu().numpy())[0, 1]
        predicted_classes = np.max(score_values, axis=1)
        actual_labels = val_loader.dataset.tensors[1].cpu().numpy().flatten()
        score_correlation = np.corrcoef(predicted_classes, actual_labels)[0, 1]


        # Calculate weights based on correlation with correct predictions
        total_correlation = abs(reservation_correlation) + abs(score_correlation)
        reservation_weight = abs(reservation_correlation) / total_correlation
        score_weight = abs(score_correlation) / total_correlation

        average_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0

        return average_val_loss, reservation_weight, score_weight



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
