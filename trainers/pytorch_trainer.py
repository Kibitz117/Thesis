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

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import copy
from tqdm import tqdm

def selective_train_and_save_model(model_name, task_type, loss_name, train_test_splits, device, model_config={}, model_save_path="best_model.pth", pretrain_epochs=2, initial_reward=2.0,num_classes=3):
    factory = ModelFactory()
    # Ensure the model has an extra output for abstention
    # Assuming 'num_classes' is the total number of classes including abstention
    model, criterion = factory.create(model_name, task_type, loss_name, selective=True, model_config=model_config, num_classes=num_classes)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    n_epochs = 1000
    reward = initial_reward
    patience = pretrain_epochs + 10
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    if device.type == 'cuda' or device.type == 'mps':
        num_workers = 4
        batch_size = 64
    else:
        num_workers = 1
        batch_size = 16

    for epoch in range(n_epochs):
        model.train()
        total_train_loss, total_samples, total_correct, total_train_abstained, non_abstained_correct, non_abstained_total = 0.0, 0, 0, 0, 0, 0

        for train_data, train_labels, _, _ in tqdm(train_test_splits):
            train_dataset = TensorDataset(train_data, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
        val_loss, val_correct, val_abstained_correct, total_val_samples, total_val_abstained, non_abstained_val_correct, non_abstained_val_total = 0.0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for _, _, val_data, val_labels in train_test_splits:
                val_dataset = TensorDataset(val_data, val_labels)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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













def train_and_save_model(model_name, task_type, loss_name, train_test_splits, device, model_config={}, model_save_path="best_model.pth",num_classes=2):
    factory = ModelFactory()
    model, criterion = factory.create(model_name, task_type, loss_name, model_config=model_config,num_classes=num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    n_epochs = 1000
    patience = 10
    best_val_loss = np.inf
    counter = 0
    max_norm = 1

    if device.type == 'cuda' or device.type == 'mps':
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

        for train_data, train_labels,_,_ in tqdm(train_test_splits):
            train_dataset = TensorDataset(train_data, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
            for _,_,val_data, val_labels in train_test_splits:
                val_dataset = TensorDataset(val_data, val_labels)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
    columns_to_keep =data_type + ['TICKER','date']
    # List of columns to drop
    columns_to_drop = df.columns.difference(columns_to_keep)

    df = df.drop(columns=columns_to_drop)

    df=df.dropna(subset=data_type)
    if start is not None:
        start_date=start
    else:
        start_date=df['date'].min()
    end_date=df['date'].max()

    return df,start_date,end_date

def main():
    #Parameters
    model_name = 'transformer'
    #cross_sectional_median, raw_returns, buckets
    target = 'buckets'
    num_classes=3
    data_type='RET'
    #extra features:'Mkt-RF','SMB','HML','RF'
    features=['RET','Mkt-RF','SMB','HML','RF']
    selective=False
    sequence_length=240
    if target=='cross_sectional_median' or target=='direction' or target=='cross_sectional_mean':
        loss_func = 'ce'
    elif target=='buckets' or 'quintiles':
        loss_func='ce'
    else:
        loss_func = 'mse'
    model_config={
        'd_model': 128,
        'num_heads': 4,
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




    # path='data/crsp_ff_adjusted.csv'
    # path='data/merged_data.csv'
    path='data/stock_data_with_factors.csv'
    # path='data/spy_universe.csv'
    # path='data/corrected_crsp_ff_adjusted.csv'
    start=datetime(2010,1,1)
    df,start_date,end_date=load_data(path,features,start)
    df = df[df['TICKER'].isin(['AAPL','MSFT','AMZN','GOOG','IBM'])]

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
        model = selective_train_and_save_model(model_name, task_types[0],loss_func, train_test_splits, device,model_config,num_classes=num_classes)
        #Test method
    else:
        model=train_and_save_model(model_name, task_types[0],loss_func, train_test_splits, device,model_config,num_classes=num_classes)
        #Test method
    #export model config
    torch.save(model.state_dict(), 'model_state_dict.pth')


if __name__ == "__main__":
    main()
