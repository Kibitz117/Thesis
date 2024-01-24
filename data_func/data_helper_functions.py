import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from joblib import Parallel, delayed
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pywt
import pandas as pd
from tqdm import tqdm



def create_study_periods(df, window_size, trade_size, train_size, forward_roll, start_date, end_date, target_type='cross_sectional_median', standardize=True, data_type='RET', apply_wavelet_transform=False):
    """
    Create a list of study periods, each consisting of a training period and a trading period.
    ...
    data_type: The type of data to use for standardization and target creation ('PRICE' or 'RET')
    """
    study_periods = []
    for p in tqdm(pd.date_range(start=start_date, freq=f'{forward_roll}D', end=end_date)):
        train_start = p
        train_end = p + pd.DateOffset(days=train_size)
        trade_end = train_end + pd.DateOffset(days=trade_size)

        if trade_end > end_date:
            print("Reached the end of the dataset.")
            break

        train_df = df[(df['date'] >= train_start) & (df['date'] < train_end)].copy()
        trade_df = df[(df['date'] >= train_end) & (df['date'] < trade_end)].copy()

        train_df[data_type] = pd.to_numeric(train_df[data_type], errors='coerce')
        trade_df[data_type] = pd.to_numeric(trade_df[data_type], errors='coerce')

        if standardize:
            train_df, trade_df = standardize_data(train_df, trade_df, data_type)

        if apply_wavelet_transform:
            train_df=apply_wavelets_to_df(train_df,data_type)
            trade_df=apply_wavelets_to_df(trade_df,data_type)

        if standardize:
            trade_df, train_df = create_targets(train_df, trade_df, target_type, window_size, 'standardized_data')
        else:
            trade_df, train_df = create_targets(train_df, trade_df, target_type, window_size, data_type)

        train_df.drop(columns=[data_type], inplace=True)
        trade_df.drop(columns=[ data_type], inplace=True)

        #)
        study_periods.append((train_df, trade_df))

    return study_periods

def standardize_data(train_df, trade_df, data_type):
    """
    Standardizes the data based on the training set
    """
    # Calculate mean and standard deviation from the training set
    mu = train_df[data_type].mean()
    sigma = train_df[data_type].std()

    # Standardize the daily returns in both datasets
    train_df['standardized_data'] = (train_df[data_type] - mu) / sigma
    trade_df['standardized_data'] = (trade_df[data_type] - mu) / sigma

    return train_df, trade_df




def create_targets(train_df, trade_df, target_type, window_size, data_type):
    #TODO: make direction/cross sectional targets based on current day returns
    if target_type == 'direction':
        # Shift the return series by one day backward to avoid lookahead bias
        train_df['target'] = (train_df[data_type].shift(-1) > 0).astype(int)
        trade_df['target'] = (trade_df[data_type].shift(-1) > 0).astype(int)

    elif target_type in ['cross_sectional_median', 'cross_sectional_mean']:
        # Calculate the median/mean for each day
        for df in [train_df, trade_df]:
            if target_type == 'cross_sectional_median':
                cross_sectional_value = df.groupby('date')[data_type].transform('median')
            else:  # cross_sectional_mean
                cross_sectional_value = df.groupby('date')[data_type].transform('mean')

            # Shift the median/mean target by one day
            df['target'] = df[data_type].shift(-1) >= cross_sectional_value.shift(-1)
            df['target'] = df['target'].astype(int)

    elif target_type in ['buckets', 'quintiles']:
        # Handle buckets and quintiles with a shift
        num_buckets = 3 if target_type == 'buckets' else 5

        # Calculate buckets/quintiles and then shift
        for df in [train_df, trade_df]:
            for ticker, group in df.groupby('TICKER'):
                buckets = pd.qcut(group[data_type], num_buckets, labels=False, duplicates='drop')
                df.loc[group.index, 'target'] = buckets.shift(-1)
            # Handle NaN values in 'target' column
            df['target'].fillna(0, inplace=True)  # Replace 0 with a suitable default value
            df['target'] = df['target'].astype(int)

    elif target_type == 'raw_return':
        # Use raw data as the target
        train_df['target'] = train_df[data_type]
        trade_df['target'] = trade_df[data_type]


    else:
        print('Invalid target type')

    return trade_df, train_df



def process_window(in_sample_df, out_of_sample_df, sequence_length, feature_columns=None):
    in_sample_df.dropna(subset=['target'], inplace=True)
    out_of_sample_df.dropna(subset=['target'], inplace=True)

    # If feature_columns is not provided, use all columns except 'date', 'TICKER', and 'target'
    if feature_columns is None:
        feature_columns = [col for col in in_sample_df.columns if col not in ['date', 'TICKER', 'target']]

    in_sample_data = []
    in_sample_labels = []
    in_sample_tickers=[]
    in_sample_dates=[]

    for i in range(0, len(in_sample_df) - sequence_length + 1):
        window = in_sample_df.iloc[i: i + sequence_length]
        in_sample_tickers.append(window['TICKER'].iloc[-1])
        features = window[feature_columns].values
        label = window['target'].iloc[-1]
        in_sample_data.append(features)
        in_sample_labels.append(label)
        in_sample_dates.append(window['date'].iloc[-1]) 

    in_sample_data = np.array(in_sample_data)
    in_sample_labels = np.array(in_sample_labels)

    task_type = 'classification' if in_sample_labels.dtype == 'int64' else 'regression'
    train_data = torch.tensor(in_sample_data, dtype=torch.float32)
    train_labels = torch.tensor(in_sample_labels, dtype=torch.long if task_type == 'classification' else torch.float32)

    out_of_sample_data = []
    out_of_sample_labels = []
    out_of_sample_tickers=[]
    out_of_sample_dates=[]

    for i in range(0, len(out_of_sample_df) - sequence_length + 1):
        window = out_of_sample_df.iloc[i: i + sequence_length]
        features = window[feature_columns].values
        out_of_sample_tickers.append(window['TICKER'].iloc[-1])
        label = window['target'].iloc[-1]
        out_of_sample_data.append(features)
        out_of_sample_labels.append(label)
        out_of_sample_dates.append(window['date'].iloc[-1]) 

    out_of_sample_data = np.array(out_of_sample_data)
    out_of_sample_labels = np.array(out_of_sample_labels)

    test_data = torch.tensor(out_of_sample_data, dtype=torch.float32)
    test_labels = torch.tensor(out_of_sample_labels, dtype=torch.long if task_type == 'classification' else torch.float32)

    return train_data, train_labels, test_data, test_labels, task_type,in_sample_tickers,out_of_sample_tickers,in_sample_dates,out_of_sample_dates


def create_tensors(study_periods, n_jobs=6, sequence_length=240, feature_columns=None):
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_window)(in_sample_df, out_of_sample_df, sequence_length, feature_columns) 
        for in_sample_df, out_of_sample_df in study_periods
    )

    train_test_splits = []
    ticker_date_mapping = {'train': [], 'test': []}
    ticker_tensor_mapping = {'train': [], 'test': []}
    task_types = []

    for result in results:
        # Unpack the result
        train_data, train_labels, test_data, test_labels, task_type, in_sample_tickers, out_of_sample_tickers,in_sample_dates,out_of_sample_dates = result

        # Append train and test data to train_test_splits
        train_test_splits.append((train_data, train_labels, test_data, test_labels))

        # Append mappings to ticker_tensor_mapping
        for i, ticker in enumerate(in_sample_tickers):
            ticker_tensor_mapping['train'].append((ticker, train_data[i], train_labels[i]))

        for i, ticker in enumerate(out_of_sample_tickers):
            ticker_tensor_mapping['test'].append((ticker, test_data[i], test_labels[i]))
            
        for i, (ticker, date) in enumerate(zip(in_sample_tickers, in_sample_dates)):
            ticker_date_mapping['train'].append((ticker, date, train_data[i], train_data[i]))

        for i, (ticker, date) in enumerate(zip(out_of_sample_tickers, out_of_sample_dates)):
            ticker_date_mapping['test'].append((ticker, date, test_data[i], test_labels[i]))


        task_types.append(task_type)

    # Check for consistent task type
    if len(set(task_types)) == 1:
        task_type = task_types[0]
    else:
        raise ValueError("Inconsistent task types across study periods.")

    return train_test_splits, ticker_tensor_mapping,ticker_date_mapping, task_type







def wavelet_mra(signal, wavelet, level):
    # Ensure the signal is at least as long as the minimum length required for the chosen level
    min_length = pywt.Wavelet(wavelet).dec_len * (2 ** level)
    if len(signal) < min_length:
        return [np.zeros(len(signal))] * (level + 1)  # Return arrays of zeros for each level

    # Perform Discrete Wavelet Transform
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Generate a list of arrays for each decomposition level
    mra_components = []
    for i in range(level):
        detail_coeffs = [np.zeros_like(signal) for _ in range(level)]
        detail_coeffs[i] = pywt.waverec([c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)], wavelet)[:len(signal)]
        mra_components.append(detail_coeffs[i])

    # Add the approximation coefficients at the last level
    approximation = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)[:len(signal)]
    mra_components.append(approximation)

    return mra_components

def apply_wavelets_to_df(df, column_name, wavelet='db6', level=3):
    transformed_df = df.copy()
    wavelets = {}

    for ticker in df['TICKER'].unique():
        signal = df[df['TICKER'] == ticker][column_name].values
        transformed_signals = wavelet_mra(signal, wavelet, level)
        wavelets[ticker] = transformed_signals

    # Add new wavelet columns to transformed df
    for i in range(level + 1):
        transformed_df['wavelet_' + str(i)] = 0  # Initialize with zeros

    for ticker in df['TICKER'].unique():
        for i in range(level + 1):
            indices = transformed_df['TICKER'] == ticker
            wavelet_data = wavelets[ticker][i]

            # Direct assignment ensuring the lengths match
            transformed_df.loc[indices, 'wavelet_' + str(i)] = wavelet_data

    return transformed_df

