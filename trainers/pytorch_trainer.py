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
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from pandas import Timestamp


def selective_train_and_save_model(model, criterion, train_test_splits, device, model_config={}, model_save_path="best_model.pth", pretrain_epochs=10, initial_reward=2.0, num_classes=3):

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    n_epochs = 1000
    reward = initial_reward
    patience = pretrain_epochs + 5
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    if device.type == 'cuda' or device.type == 'mps':
        num_workers = 12
        batch_size = 128
    else:
        num_workers = 1
        batch_size = 16

    train_loaders = []
    val_loaders = []

    for train_data, train_labels, val_data, val_labels in train_test_splits:
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train_loaders.append(train_loader)

        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loaders.append(val_loader)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss, total_samples, total_correct, total_train_abstained, non_abstained_correct, non_abstained_total = 1.0, 1, 1, 1, 1, 1

        for train_loader in tqdm(train_loaders):
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                # labels = labels.view(-1, 1).float()
                # labels = labels.long().squeeze()
                optimizer.zero_grad()
                outputs = model(data)
                if epoch < pretrain_epochs:
                    loss = criterion(outputs[:, :-1], labels)
                    predictions = torch.argmax(outputs[:, :-1], dim=1)
                    total_correct += (predictions == labels).sum().item()
                else:
                    outputs = F.softmax(outputs, dim=1)
                    class_outputs, abstention_output = outputs[:, :-1], outputs[:, -1]
                    gain = torch.gather(class_outputs, dim=1, index=labels.unsqueeze(1).long()).squeeze()
                    doubling_rate = (gain.add(abstention_output.div(reward))).log()
                    loss = -doubling_rate.mean()

                    predictions = torch.argmax(class_outputs, dim=1)
                    is_abstained = (abstention_output > gain).float()
                    total_train_abstained += is_abstained.sum().item()

                    # Calculate accuracy only for non-abstained predictions
                    non_abstained_mask = is_abstained == 0
                    non_abstained_correct += ((predictions == labels) & non_abstained_mask).sum().item()
                    non_abstained_total += non_abstained_mask.sum().item()

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        average_loss = total_train_loss / total_samples
        train_accuracy = total_correct / total_samples if epoch < pretrain_epochs else non_abstained_correct / non_abstained_total
        train_abstain_rate = total_train_abstained / total_samples

        # Validation Step with Abstention Mechanism
        model.eval()
        val_loss, val_correct, val_abstained_correct, total_val_samples, total_val_abstained, non_abstained_val_correct, non_abstained_val_total = 1.0, 1, 1, 1, 1, 1, 1
        with torch.no_grad():
            for val_loader in val_loaders:

                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    # labels = labels.view(-1, 1).float()
                    # labels = labels.long().squeeze()
                    outputs = model(data)
                    if epoch < pretrain_epochs:
                        loss = criterion(outputs[:, :-1], labels)
                        predictions = torch.argmax(outputs[:, :-1], dim=1)
                        val_correct += (predictions == labels).sum().item()
                    else:
                        outputs = F.softmax(outputs, dim=1)
                        class_outputs, abstention_output = outputs[:, :-1], outputs[:, -1]
                        gain = torch.gather(class_outputs, dim=1, index=labels.unsqueeze(1).long()).squeeze()
                        doubling_rate = (gain.add(abstention_output.div(reward))).log()
                        loss = -doubling_rate.mean()

                        predictions = torch.argmax(class_outputs, dim=1)
                        is_abstained = (abstention_output > gain).float()
                        total_val_abstained += is_abstained.sum().item()

                        # Calculate accuracy only for non-abstained predictions
                        non_abstained_mask = is_abstained == 0
                        non_abstained_val_correct += ((predictions == labels) & non_abstained_mask).sum().item()
                        non_abstained_val_total += non_abstained_mask.sum().item()

                    val_loss += loss.item() * data.size(0)
                    total_val_samples += data.size(0)

        average_val_loss = val_loss / total_val_samples
        val_accuracy = val_correct / total_val_samples if epoch < pretrain_epochs else non_abstained_val_correct / non_abstained_val_total
        val_abstain_rate = total_val_abstained / total_val_samples

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Abstain Rate: {train_abstain_rate:.4f}, Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Abstain Rate: {val_abstain_rate:.4f}, Total Val Abstained: {total_val_abstained}')

        # Early Stopping Check with Patience
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping!')
                break

    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        model.load_state_dict(best_model_state)

    return model













def train_and_save_model(model, criterion, train_test_splits, device, model_config={}, model_save_path="best_model.pth", num_classes=2):

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    n_epochs = 1000
    patience = 5
    best_val_loss = np.inf
    counter = 0
    max_norm = 1

    if device.type == 'cuda' or device.type == 'mps':
        num_workers = 12
        batch_size = 128
    else:
        num_workers = 1
        batch_size = 16

    train_loaders = []
    val_loaders = []
    for train_data, train_labels, val_data, val_labels in train_test_splits:
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train_loaders.append(train_loader)

        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loaders.append(val_loader)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        total_correct = 0
        total_samples = 0
        

        for train_loader in tqdm(train_loaders):

            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                # labels = labels.view(-1, 1).float()
                # labels = labels.squeeze(1)
                optimizer.zero_grad()
                mask = model.create_lookahead_mask(data.size(1)).to(device)  # Mask of size sequence length
                outputs = model(data, src_mask=mask)
                # output=outputs[:, :-1]
                loss = criterion(outputs, labels) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

                total_train_loss += loss.item() * data.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds== labels).sum().item()
                total_samples += labels.size(0)

        average_train_loss = total_train_loss / total_samples
        train_accuracy = total_correct / total_samples

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for val_loader in val_loaders:

                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    # labels = labels.view(-1, 1).float()
                    mask = model.create_lookahead_mask(data.size(1)).to(device) 
                    outputs = model(data, src_mask=mask)
                    # output=outputs[:, :-1]
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * data.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    total_val_samples += labels.size(0)

        average_val_loss = val_loss / total_val_samples
        val_accuracy = val_correct / total_val_samples

        print(f'Epoch {epoch+1}/{n_epochs}, Average Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Average Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Early Stopping Check
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping!')
                break

    best_model_state = torch.load(model_save_path, map_location=device)
    model.load_state_dict(best_model_state)
    return model



def load_data(path, data_type, start=None):
    df = pd.read_csv(path)
    
    # Convert 'date' to datetime without considering timezone
    df['date'] = pd.to_datetime(df['date'])

    # Filtering the relevant columns
    columns_to_keep = data_type + ['TICKER', 'date']
    df = df[columns_to_keep]

    # Drop rows with missing values in the specified columns
    df = df.dropna(subset=data_type)

    # Determining the start and end dates
    if start is not None:
        start_date = pd.to_datetime(start)
    else:
        start_date = df['date'].min()

    end_date = df['date'].max()

    return df, start_date, end_date


def extract_tensors_and_labels(tensor_ticker_pairs):
    train_data, test_data = tensor_ticker_pairs

    train_tensors = [tensor for (tensor, label, ticker) in train_data]
    train_labels = [label for (tensor, label, ticker) in train_data]

    test_tensors = [tensor for (tensor, label, ticker) in test_data]
    test_labels = [label for (tensor, label, ticker) in test_data]

    return train_tensors, train_labels, test_tensors, test_labels

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
