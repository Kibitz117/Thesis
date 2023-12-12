import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from joblib import Parallel, delayed
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from tqdm import tqdm

def create_study_periods(df, n_periods, window_size, trade_size, train_size, forward_roll, start_date, end_date, target_type='cross_sectional_median', standardize=True, return_type='RET'):
    """
    Create a list of study periods, each consisting of a training period and a trading period.
    ...
    return_type: The type of return to use for standardization and target creation
    """
    study_periods = []
    for p in tqdm(pd.date_range(start=start_date, freq=f'{forward_roll}D', end=end_date)):
        train_start = p
        train_end = p + pd.DateOffset(days=train_size)
        trade_end = train_end + pd.DateOffset(days=trade_size)
        
        if trade_end > end_date:
            print("Reached the end of the dataset.")
            break

        # Create separate DataFrames for training and trading periods
        train_df = df[(df['date'] >= train_start) & (df['date'] < train_end)].copy()
        trade_df = df[(df['date'] >= train_end) & (df['date'] < trade_end)].copy()
        
        if train_df.empty or trade_df.empty:
            print(f"No data available for the period {train_start.date()} to {trade_end.date()}. Skipping.")
            continue

        # Drop other return columns and rename the specified return column to 'RET'
        return_cols = ['Adj_RET_Mkt', 'Adj_RET_Mkt_SMB', 'Adj_RET_Mkt_SMB_HML', 'RET']
        return_cols.remove(return_type)
        train_df = train_df.drop(columns=return_cols)
        trade_df = trade_df.drop(columns=return_cols)
        train_df = train_df.rename(columns={return_type: 'RET'})
        trade_df = trade_df.rename(columns={return_type: 'RET'})
        
        if standardize:
            # Standardize returns with a rolling window
            train_df['rolling_mean'] = train_df.groupby('TICKER')['RET'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
            mu = train_df['rolling_mean'].mean()
            sigma = train_df['rolling_mean'].std()
            train_df['standardized_return'] = (train_df['rolling_mean'] - mu) / sigma

            # Standardize returns for trade_df using the mean and std dev from train_df
            trade_df['rolling_mean'] = trade_df.groupby('TICKER')['RET'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
            trade_df['standardized_return'] = (trade_df['rolling_mean'] - mu) / sigma
            
        # Assign binary targets based on whether the standardized return is above the median
        trade_df, train_df = create_targets(train_df, trade_df, target_type)

        # Store the training and trading DataFrames in the study_periods list
        study_periods.append((train_df, trade_df))

    return study_periods


def create_targets(train_df, trade_df, target):
    if target == 'cross_sectional_median':
        # do stuff
        median_standardized_return = train_df['standardized_return'].median()
        train_df['target'] = (train_df['standardized_return'] >= median_standardized_return).astype(int)
        trade_df['target'] = (trade_df['standardized_return'] >= median_standardized_return).astype(int)

    elif target == 'raw_return':
        # Use raw returns as the target
        train_df['target'] = train_df['RET']
        trade_df['target'] = trade_df['RET']

    elif target == 'buckets':
        # Make 10 buckets of returns as targets
        train_df['target'] = train_df.groupby('TICKER')['standardized_return'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')).astype(int)
        trade_df['target'] = trade_df.groupby('TICKER')['standardized_return'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')).astype(int)
        
    else:
        print('Invalid target type')
    return trade_df, train_df




def process_window(in_sample_df, out_of_sample_df, sequence_length):
    in_sample_data = []
    in_sample_labels = []

    for i in range(0, len(in_sample_df) - sequence_length + 1):
        window = in_sample_df.iloc[i: i + sequence_length]
        features = window[['standardized_return']].values
        label = window['target'].iloc[-1]  # Getting the last day's label
        in_sample_data.append(features)
        in_sample_labels.append(label)

    # Converting lists to numpy arrays
    in_sample_data = np.array(in_sample_data)
    in_sample_labels = np.array(in_sample_labels)

    # Determine if the task is classification or regression based on the type of the labels
    task_type = 'classification' if in_sample_labels.dtype == 'int64' else 'regression'
    
    # Creating tensors from numpy arrays
    if task_type == 'classification':
        train_data = torch.tensor(in_sample_data, dtype=torch.float32)
        train_labels = torch.tensor(in_sample_labels, dtype=torch.long)
    else:
        train_data = torch.tensor(in_sample_data, dtype=torch.float32)
        train_labels = torch.tensor(in_sample_labels, dtype=torch.float32)

    out_of_sample_data = []
    out_of_sample_labels = []

    for i in range(0, len(out_of_sample_df) - sequence_length + 1):
        window = out_of_sample_df.iloc[i: i + sequence_length]
        features = window[['standardized_return']].values
        label = window['target'].iloc[-1]  # Getting the last day's label
        out_of_sample_data.append(features)
        out_of_sample_labels.append(label)

    out_of_sample_data = np.array(out_of_sample_data)
    out_of_sample_labels = np.array(out_of_sample_labels)

    # Creating tensors from numpy arrays
    if task_type == 'classification':
        test_data = torch.tensor(out_of_sample_data, dtype=torch.float32)
        test_labels = torch.tensor(out_of_sample_labels, dtype=torch.long)
    else:
        test_data = torch.tensor(out_of_sample_data, dtype=torch.float32)
        test_labels = torch.tensor(out_of_sample_labels, dtype=torch.float32)

    return train_data, train_labels, test_data, test_labels, task_type


def create_tensors(study_periods, n_jobs=6, sequence_length=240):
    """
    Create tensors from the study periods.
    study_periods: List of study periods
    n_jobs: Number of cores to use
    sequence_length: size of tensor sequence
    """
    # Parallelizing the process_window function across multiple cores
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_window)(in_sample_df, out_of_sample_df, sequence_length) 
        for in_sample_df, out_of_sample_df in study_periods
    )

    # Separating the results and task types
    train_test_splits, task_types = zip(*[(result[:-1], result[-1]) for result in results])
    
    return train_test_splits, task_types