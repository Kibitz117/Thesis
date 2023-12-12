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
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def selective_train_and_save_model(model_name, task_type, loss_name, train_test_splits, device, model_config={}, model_save_path="best_model.pth", pretrain_epochs=3, initial_reward=6.0, min_coverage=0.3):
    factory = ModelFactory()
    model, criterion = factory.create(model_name, task_type, loss_name, selective=True, model_config=model_config)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 1000
    reward = initial_reward
    dynamic_threshold = 0.9  # Adjusted initial dynamic rejection threshold
    penalty_factor = 60.0  # Adjusted hyperparameter for coverage penalty
    accuracy_boost_factor = 3.0  # Adjusted hyperparameter for accuracy reward

    best_loss = float('inf')
    best_model_state = None

    if device == 'cuda':
        num_workers = 4
        batch_size = 128
    else:
        num_workers = 1
        batch_size = 16

    for epoch in range(n_epochs):
        model.train()
        total_train_loss, total_coverage, total_samples = 0.0, 0.0, 0
        total_non_rejected_correct, total_non_rejected_samples = 0, 0

        for train_data, train_labels ,_, _ in tqdm(train_test_splits):
            train_dataset = TensorDataset(train_data, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                labels = labels.view(-1, 1).float()
                optimizer.zero_grad()

                main_output, reservation = model(data)

                if epoch < pretrain_epochs:
                    # Pretraining phase without rejection option
                    probabilities = torch.sigmoid(main_output)
                    loss = criterion(probabilities, labels)
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
                    train_loss = base_loss.item() * data.size(0)
                    total_train_loss += train_loss
                    total_coverage += coverage.item() * data.size(0)
                    total_samples += data.size(0)
                    total_non_rejected_correct += non_rejected_correct.item()
                    total_non_rejected_samples += accepted.sum().item()

                    # Penalty and rewards
                    coverage_penalty = max(0, min_coverage - coverage) * penalty_factor
                    accuracy_term = non_rejected_correct / accepted.sum()
                    accuracy_reward = accuracy_boost_factor * accuracy_term
                    loss = base_loss + coverage_penalty - accuracy_reward

                loss.backward()
                optimizer.step()

            # Calculate and update best model state based on loss
            average_loss = total_train_loss / total_samples
            if average_loss < best_loss:
                best_loss = average_loss
                best_model_state = copy.deepcopy(model.state_dict())

        # Print epoch metrics
        average_coverage = total_coverage / total_samples
        non_rejected_accuracy = total_non_rejected_correct / total_non_rejected_samples if total_non_rejected_samples > 0 else 0
        print(f'Epoch {epoch+1}/{n_epochs}, Average Train Loss: {average_loss:.4f}, '
              f'Coverage: {average_coverage:.2f}, Non-Rejected Accuracy: {non_rejected_accuracy:.4f}')

        # Adjust dynamic threshold and reward based on coverage
        if average_coverage < min_coverage:
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

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 1000
    patience = 5
    best_loss = np.inf
    counter = 0
    #USE RSYNC TO MOVE FILES TO VM
    if device == 'cuda':
        num_workers = 4
        batch_size = 128
    else:
        num_workers = 1
        batch_size = 16

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        total_correct = 0
        total_samples = 0

        for train_data, train_labels, _, _ in tqdm(train_test_splits):
            train_dataset = TensorDataset(train_data, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            train_loss = 0.0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                labels = labels.view(-1, 1).float()

                optimizer.zero_grad()
                outputs, _ = model(data)
                # outputs = outputs.squeeze()
                loss = criterion(outputs, labels) 
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

                # Calculate accuracy
                preds = torch.sigmoid(outputs) >= 0.5
                preds = preds.view(-1,1).float()  # Reshape and convert to float for comparison
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            total_train_loss += train_loss / len(train_loader.dataset)

        average_train_loss = total_train_loss / len(train_test_splits)
        train_accuracy = total_correct / total_samples

        print(f'Epoch {epoch+1}/{n_epochs}, Average Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        if average_train_loss < best_loss:
            best_loss = average_train_loss
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


def main():
    #Parameters
    model_name = 'transformer'
    #cross_sectional_median, raw_returns, buckets
    target = 'buckets'
    selective=True
    if target=='cross_sectional_median':
        loss_func = 'bce'
        num_classes=1
    elif target=='buckets':
        loss_func='ce'
        num_classes=10
    else:
        loss_func = 'mse'
        num_classes=0
    model_config={
        'd_model': 16,
        'num_heads': 4,
        'd_ff': 32,
        'num_encoder_layers': 1,
        'dropout': .1,

    }
     # Check if CUDA, MPS, or CPU should be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    df = pd.read_csv('data/crsp_ff_adjusted.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.dropna(subset=['RET'], inplace=True)
    df = df.drop(columns='Unnamed: 0')
    #subset df to 2014-2015
    df = df[df['date'] >= datetime(2014, 1, 1)]
    
    # Create tensors
    study_periods = create_study_periods(df, n_periods=23, window_size=240, trade_size=250, train_size=750, forward_roll=250, 
                                         start_date=datetime(1990, 1, 1), end_date=datetime(2015, 12, 31), target_type=target,num_classes=num_classes)
    train_test_splits, task_types = create_tensors(study_periods)


    if selective==True:
        model = selective_train_and_save_model(model_name, task_types[0],loss_func, train_test_splits, device,model_config,num_classes=num_classes)
        #Test method
    else:
        model=train_and_save_model(model_name, task_types[0],loss_func, train_test_splits, device,model_config,num_classes=num_classes)
        #Test method
    #export model config
    torch.save(model.state_dict(), 'model_state_dict.pth')

    # in_sample_long_portfolios, in_sample_short_portfolios, out_of_sample_long_portfolios, out_of_sample_short_portfolios = evaluate_model(model, train_test_splits, k=10, device=device)

    # Export portfolios
    # in_sample_long_portfolios.to_csv(f'../data/{model_name}_results/in_sample_long_portfolios.csv')
    # in_sample_short_portfolios.to_csv(f'../data/{model_name}_results/in_sample_short_portfolios.csv')
    # out_of_sample_long_portfolios.to_csv(f'../data/{model_name}_results/out_of_sample_long_portfolios.csv')
    # out_of_sample_short_portfolios.to_csv(f'../data/{model_name}_results/out_of_sample_short_portfolios.csv')

if __name__ == "__main__":
    main()
