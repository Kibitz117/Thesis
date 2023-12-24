import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import math
import copy
from datetime import datetime
sys.path.append('utils')
sys.path.append('models')
sys.path.append('data_func')
from data_helper_functions import create_study_periods,create_tensors
from model_factory import ModelFactory
from transformer_model import ScaledMultiHeadAttention
from sharpe_loss import SharpeLoss
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F



def selective_train_and_save_model(model_name, task_type, loss_name, train_test_splits, device, model_config={}, model_save_path="best_model.pth", pretrain_epochs=10, initial_reward=6.0, min_coverage=0.3):
    factory = ModelFactory()
    model, criterion = factory.create(model_name, task_type, loss_name, selective=True, model_config=model_config)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 1000
    reward = initial_reward
    dynamic_threshold = 0.9  # Adjusted initial dynamic rejection threshold
    penalty_factor = 60.0  # Adjusted hyperparameter for coverage penalty
    accuracy_boost_factor = 3.0  # Adjusted hyperparameter for accuracy reward

    best_loss = float('inf')
    best_model_state = None

    if device.type == 'cuda' or device.type=='mps':
        num_workers = 4
        batch_size = 64
    else:
        num_workers = 1
        batch_size = 16

    for epoch in range(n_epochs):
        model.train()
        total_train_loss, total_samples = 0.0, 0
        # Initialize coverage and rejection-related variables
        total_coverage, total_non_rejected_correct, total_non_rejected_samples = 0.0, 0, 0

        for train_data, train_labels, _, _ in tqdm(train_test_splits):
            train_dataset = TensorDataset(train_data, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                labels = labels.view(-1, 1).float()
                optimizer.zero_grad()

                main_output, reservation = model(data)

                if epoch < pretrain_epochs:
                    # Pretraining phase without rejection option
                    loss = criterion(main_output, labels)
                else:
                    # Regular training with selective mechanism
                    probabilities = torch.sigmoid(main_output)
                    gain = torch.where(labels == 1, probabilities, 1 - probabilities)

                    doubling_rate = (gain + reservation.div(reward + 1e-8)).log()
                    base_loss = -doubling_rate.mean()

                    # Calculate coverage and accuracy for non-rejected samples
                    accepted = reservation < dynamic_threshold
                    coverage = accepted.float().mean()
                    predictions = torch.round(probabilities)
                    correct_predictions = (predictions == labels).float()
                    non_rejected_correct = (correct_predictions * accepted).sum()

                    # Update totals
                    total_coverage += coverage.item() * data.size(0)
                    total_non_rejected_correct += non_rejected_correct.item()
                    total_non_rejected_samples += accepted.sum().item()
                    train_loss = base_loss.item() * data.size(0)

                    # Penalty and rewards
                    coverage_penalty = max(0, min_coverage - coverage) * penalty_factor
                    accuracy_term = non_rejected_correct / (accepted.sum() + 1e-8)  # Avoid division by zero
                    accuracy_reward = accuracy_boost_factor * accuracy_term
                    loss = base_loss + coverage_penalty - accuracy_reward

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        # Update best model state and print metrics outside the inner loop
        average_loss = total_train_loss / total_samples
        if average_loss < best_loss:
            best_loss = average_loss
            best_model_state = copy.deepcopy(model.state_dict())

        average_coverage = total_coverage / total_samples if epoch >= pretrain_epochs else 0
        non_rejected_accuracy = total_non_rejected_correct / total_non_rejected_samples if total_non_rejected_samples > 0 else 0
        print(f'Epoch {epoch+1}/{n_epochs}, Average Train Loss: {average_loss:.4f}, '
              f'Coverage: {average_coverage:.2f}, Non-Rejected Accuracy: {non_rejected_accuracy:.4f}')

        # Adjust dynamic threshold and reward based on coverage outside the inner loop
        if epoch >= pretrain_epochs and average_coverage < min_coverage:
            dynamic_threshold *= 0.9  # Decrease threshold
            reward *= 0.9  # Decrease reward

    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        model.load_state_dict(best_model_state)

    return model







def train_and_save_model(model_name, task_type, loss_name, train_test_splits, device, model_config={}, model_save_path="best_model.pth"):
    factory = ModelFactory()
    model, criterion = factory.create(model_name, task_type, loss_name, model_config=model_config)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)
    n_epochs = 1000
    patience = 5
    best_val_loss = np.inf
    counter = 0
    max_norm=1

    if device.type == 'cuda' or device.type=='mps':
        num_workers = 8
        batch_size = 64
    else:
        num_workers = 1
        batch_size = 16

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        total_correct = 0
        total_samples = 0
        i=1
        for train_data, train_labels, val_data, val_labels in tqdm(train_test_splits):
            # print(f'Period {i}') #Debug second iteration of period 2
            # i+=1
            train_dataset = TensorDataset(train_data, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            train_loss = 0.0
            for data, labels in (train_loader):
                data, labels = data.to(device), labels.to(device)
                labels = labels.view(-1, 1).float()

                optimizer.zero_grad()
                mask=model.create_lookahead_mask(data.size(1)).to(device)#Mask of size sequence length
                outputs,_ = model(data,src_mask=mask)
                loss = criterion(outputs, labels) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                train_loss += loss.item() * data.size(0)

                preds = torch.sigmoid(outputs) >= 0.5
                preds = preds.view(-1,1).float()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            total_train_loss += train_loss / len(train_loader.dataset)

        average_train_loss = total_train_loss / len(train_test_splits)
        train_accuracy = total_correct / total_samples

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for data, labels in DataLoader(TensorDataset(val_data, val_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers):
                data, labels = data.to(device), labels.to(device)
                labels = labels.view(-1, 1).float()

                # mask=TimeSeriesTransformer.create_lookahead_mask(data.size(1))#Mask of size sequence length
                outputs, _ = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * data.size(0)

                preds = torch.sigmoid(outputs) >= 0.5
                preds = preds.view(-1,1).float()
                val_correct += (preds == labels).sum().item()

            average_val_loss = val_loss / len(val_data)
            val_accuracy = val_correct / len(val_data)

        print(f'Epoch {epoch+1}/{n_epochs}, Average Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Average Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Early Stopping Check
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
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


#Goal with this would then be to see what model predicts right, wrong, and rejects. Then merge industry, and scatter plot to see if industry and rejection correlate
def selective_test(model_name, task_type, loss_name, test_loader, device, model_config={}, model_save_path="best_model.pth", reward=1.0, coverage_thresholds=[0.1, 0.2, 0.5, 0.7, 0.9]):
    factory = ModelFactory()
    model, _ = factory.create(model_name, task_type, loss_name, selective=True, model_config=model_config)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Store predictions, labels, and reservation probabilities
    all_labels = []
    all_predictions = []
    all_reservations = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            outputs = F.softmax(outputs, dim=1)

            class_probabilities, reservation = outputs[:, :-1], outputs[:, -1]
            predictions = torch.argmax(class_probabilities, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_reservations.extend(reservation.cpu().numpy())

    # Convert lists to numpy arrays for easier manipulation
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_reservations = np.array(all_reservations)

    # Evaluate model's accuracy and coverage/selectiveness
    for threshold in coverage_thresholds:
        mask = all_reservations <= threshold
        covered_labels = all_labels[mask]
        covered_predictions = all_predictions[mask]

        if len(covered_labels) > 0:
            accuracy = np.mean(covered_labels == covered_predictions)
            coverage = len(covered_labels) / len(all_labels)
            print(f"Coverage: {coverage:.2f}, Accuracy at threshold {threshold}: {accuracy:.2f}")
        else:
            print(f"No predictions made at threshold {threshold}")

    return all_labels, all_predictions, all_reservations

def load_data(path,data_type,start=None):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    columns_to_keep = [data_type, 'TICKER','date']
    # List of columns to drop
    columns_to_drop = df.columns.difference(columns_to_keep)

    df = df.drop(columns=columns_to_drop)

    df=df.dropna(subset=[data_type])
    if start is not None:
        start_date=start
    else:
        start_date=df['date'].min()
    end_date=df['date'].max()

    return df,start_date,end_date

def main():
    #Parameters
    model_name = 'cnn_transformer'
    #cross_sectional_median, raw_returns, buckets
    target = 'cross_sectional_median'
    data_type='RET'
    selective=False
    sequence_length=240
    if target=='cross_sectional_median' or target=='direction':
        loss_func = 'bce'
    elif target=='buckets':
        loss_func='ce'
    else:
        loss_func = 'mse'
    model_config={
        'd_model': 64,
        'num_heads': 8,
        'd_ff': 256,
        'num_encoder_layers': 2,
        'dropout': .1,

    }
     # Check if CUDA, MPS, or CPU should be used
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    # Fallback to CPU
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    #Create code that intelligently drops unused columns aside from date and ticker, and any classes. num_classes is 1 by default, but ends up being whatever it is after wavelet transform or adding interest rate
    # Load data
    # df = pd.read_csv('data/crsp_ff_adjusted.csv')
    # df['date'] = pd.to_datetime(df['date'])
    # df.dropna(subset=['RET'], inplace=True)
    # df = df.drop(columns='Unnamed: 0')
    # #subset df to 2014-2015
    # df = df[df['date'] >= datetime(2013, 1, 1)]
    # # df = df[df['TICKER'].isin(['AAPL','MSFT','AMZN','GOOG','FB'])]
    # start_date=df['date'].min()
    # end_date=df['date'].max()
    # # df=pd.read_csv('data/vectorbt_data.csv')
    # # df['date'] = pd.to_datetime(df['date'])
    # # df.dropna(subset=['PRICE'],inplace=True)
    # # df = df.drop(columns='Unnamed: 0')
    

    # # Create tensors
    # study_periods = create_study_periods(df, window_size=240, trade_size=250, train_size=750, forward_roll=250, 
    #                                      start_date=start_date, end_date=end_date, target_type=target,data_type=data_type,apply_wavelet_transform=False)
    # train_test_splits, task_types = create_tensors(study_periods,n_jobs=10)

 #WAVELETS
    # df = pd.read_csv('data/crsp_ff_adjusted.csv')
    # df['date'] = pd.to_datetime(df['date'])
    # df.dropna(subset=['RET'], inplace=True)
    # df = df.drop(columns='Unnamed: 0')
    # #subset df to 2014-2015
    # df = df[df['date'] >= datetime(2013, 1, 1)]
    # # df = df[df['TICKER'].isin(['AAPL','MSFT','AMZN','GOOG','FB'])]
    # start_date=df['date'].min()
    # end_date=df['date'].max()

    path='data/crsp_ff_adjusted.csv'
    # path='data/corrected_crsp_ff_adjusted.csv'
    start=datetime(2012,1,1)
    df,start_date,end_date=load_data(path,'RET',start)
    # df = df[df['TICKER'].isin(['AAPL','MSFT','AMZN','GOOG','IBM'])]

    study_periods = create_study_periods(df, window_size=240, trade_size=250, train_size=750, forward_roll=250, 
                                         start_date=start_date, end_date=end_date, target_type=target,data_type=data_type,apply_wavelet_transform=False)
    columns=study_periods[0][0].columns.to_list()
    feature_columns = [col for col in columns if col not in ['date', 'TICKER', 'target']]
    model_config['input_features']=len(feature_columns)
    if len(study_periods)>10:
        n_jobs=10
    else:
        n_jobs=len(study_periods)
    train_test_splits, task_types = create_tensors(study_periods,n_jobs,sequence_length,feature_columns)

    

    if selective==True:
        model = selective_train_and_save_model(model_name, task_types[0],loss_func, train_test_splits, device,model_config)
        #Test method
    else:
        model=train_and_save_model(model_name, task_types[0],loss_func, train_test_splits, device,model_config)
        #Test method
    #export model config
    torch.save(model.state_dict(), 'model_state_dict.pth')


if __name__ == "__main__":
    main()
