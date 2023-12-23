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



def create_study_periods(df, window_size, trade_size, train_size, forward_roll, start_date, end_date, target_type='direction', standardize=True, data_type='PRICE', apply_wavelet_transform=False):
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
            train_df, trade_df = standardize_data(train_df, trade_df, window_size, data_type)

        if apply_wavelet_transform:
            train_df = apply_wavelet_transform_to_df(train_df,data_type)
            trade_df = apply_wavelet_transform_to_df(trade_df,data_type)

        trade_df, train_df = create_targets(train_df, trade_df, target_type, window_size, data_type)

        study_periods.append((train_df, trade_df))

    return study_periods

def standardize_data(train_df, trade_df, window_size, data_type):
    """
    Standardizes the data based on a rolling window
    """
    train_df['rolling_mean'] = train_df.groupby('TICKER')[data_type].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    mu = train_df['rolling_mean'].mean()
    sigma = train_df['rolling_mean'].std()
    train_df['standardized_data'] = (train_df['rolling_mean'] - mu) / sigma

    trade_df['rolling_mean'] = trade_df.groupby('TICKER')[data_type].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    trade_df['standardized_data'] = (trade_df['rolling_mean'] - mu) / sigma

    return train_df, trade_df




def create_targets(train_df, trade_df, target_type, window_size, data_type):
    if target_type == 'direction':
        # Create binary targets based on the direction of price or return movement
        train_df['target'] = (train_df.groupby('TICKER')[data_type].shift(-1) > train_df[data_type]).astype(int)
        trade_df['target'] = (trade_df.groupby('TICKER')[data_type].shift(-1) > trade_df[data_type]).astype(int)

    elif target_type == 'cross_sectional_median':
        # Calculate rolling median for each point in time in the training set
        rolling_median = train_df.groupby('TICKER')[data_type].transform(lambda x: x.rolling(window=window_size, min_periods=1).median())

        # Assign binary targets based on whether the data is above the rolling median
        train_df['target'] = (train_df[data_type] >= rolling_median).astype(int)
        
        # For the trading set, use the last calculated rolling median from the training set
        last_rolling_median = rolling_median.iloc[-1]
        trade_df['target'] = (trade_df[data_type] >= last_rolling_median).astype(int)

    elif target_type == 'raw_return':
        # Use raw data as the target
        train_df['target'] = train_df[data_type]
        trade_df['target'] = trade_df[data_type]

    elif target_type == 'buckets':
        # Handle NaN or infinite values in data
        for df in [train_df, trade_df]:
            # Replace infinite values with NaN
            df[data_type].replace([np.inf, -np.inf], np.nan, inplace=True)
            # Fill NaN values with a specific value, e.g., the median
            df[data_type].fillna(df[data_type].median(), inplace=True)

        # Bucketing for train_df
        for ticker, group in train_df.groupby('TICKER'):
            # Create buckets with qcut
            buckets = pd.qcut(group[data_type], 10, labels=False, duplicates='drop')
            # Assign buckets to the train_df
            train_df.loc[group.index, 'target'] = buckets

        # Bucketing for trade_df
        for ticker, group in trade_df.groupby('TICKER'):
            # Create buckets with qcut
            buckets = pd.qcut(group[data_type], 10, labels=False, duplicates='drop')
            # Assign buckets to the trade_df
            trade_df.loc[group.index, 'target'] = buckets

    elif target_type == 'quintiles':
        # Handle NaN or infinite values in data
        for df in [train_df, trade_df]:
            df[data_type].replace([np.inf, -np.inf], np.nan, inplace=True)
            df[data_type].fillna(df[data_type].median(), inplace=True)

        # Quintile bucketing for train_df
        for ticker, group in train_df.groupby('TICKER'):
            quintiles = pd.qcut(group[data_type], 5, labels=False, duplicates='drop')
            train_df.loc[group.index, 'target'] = quintiles

        # Quintile bucketing for trade_df
        for ticker, group in trade_df.groupby('TICKER'):
            quintiles = pd.qcut(group[data_type], 5, labels=False, duplicates='drop')
            trade_df.loc[group.index, 'target'] = quintiles

    else:
        print('Invalid target type')

    return trade_df, train_df



def process_window(in_sample_df, out_of_sample_df, sequence_length):
    in_sample_df.dropna(subset=['target'], inplace=True)
    out_of_sample_df.dropna(subset=['target'], inplace=True)
    in_sample_data = []
    in_sample_labels = []

    for i in range(0, len(in_sample_df) - sequence_length + 1):
        window = in_sample_df.iloc[i: i + sequence_length]
        features = window[['standardized_data']].values  # Updated to 'standardized_data'
        label = window['target'].iloc[-1]  # Getting the last day's label
        in_sample_data.append(features)
        in_sample_labels.append(label)

    # Converting lists to numpy arrays
    in_sample_data = np.array(in_sample_data)
    in_sample_labels = np.array(in_sample_labels)

    task_type = 'classification' if in_sample_labels.dtype == 'int64' else 'regression'
    train_data = torch.tensor(in_sample_data, dtype=torch.float32)
    train_labels = torch.tensor(in_sample_labels, dtype=torch.long if task_type == 'classification' else torch.float32)

    out_of_sample_data = []
    out_of_sample_labels = []

    for i in range(0, len(out_of_sample_df) - sequence_length + 1):
        window = out_of_sample_df.iloc[i: i + sequence_length]
        features = window[['standardized_data']].values  # Updated to 'standardized_data'
        label = window['target'].iloc[-1]  # Getting the last day's label
        out_of_sample_data.append(features)
        out_of_sample_labels.append(label)

    out_of_sample_data = np.array(out_of_sample_data)
    out_of_sample_labels = np.array(out_of_sample_labels)

    test_data = torch.tensor(out_of_sample_data, dtype=torch.float32)
    test_labels = torch.tensor(out_of_sample_labels, dtype=torch.long if task_type == 'classification' else torch.float32)

    return train_data, train_labels, test_data, test_labels, task_type

def create_tensors(study_periods, n_jobs=6, sequence_length=240):
    """
    Create tensors from the study periods.
    study_periods: List of study periods
    n_jobs: Number of cores to use
    sequence_length: size of tensor sequence
    """
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_window)(in_sample_df, out_of_sample_df, sequence_length) 
        for in_sample_df, out_of_sample_df in study_periods
    )

    train_test_splits, task_types = zip(*[(result[:-1], result[-1]) for result in results])
    
    return train_test_splits, task_types

def wavelet_transform(signal, wavelet='db2', mode='soft', level=None, control_coefficient=0.5, window_size=100):
    if np.isnan(signal).any():
        signal = np.nan_to_num(signal)

    denoised_signal = np.zeros_like(signal)
    
    for i in range(0, len(signal), window_size):
        window = signal[i:i+window_size]

        # Adjusting the window size for the last segment if it's smaller than window_size
        current_window_size = len(window)
        if current_window_size < window_size:
            window = np.pad(window, (0, window_size - current_window_size), 'constant')
        
        coeffs = pywt.wavedec(window, wavelet, level=level)
        sigma = np.sqrt(np.mean(np.square(coeffs[-1])))
        epsilon = 1e-10
        threshold = max(control_coefficient * sigma, epsilon)
        denoised_coeffs = [pywt.threshold(coeff, value=threshold, mode=mode) for coeff in coeffs]
        denoised_coeffs = [np.nan_to_num(coeff) for coeff in denoised_coeffs]
        denoised_window = pywt.waverec(denoised_coeffs, wavelet)

        # Adjust the denoised_window size to match the current window size
        denoised_window = denoised_window[:current_window_size]

        denoised_signal[i:i+current_window_size] = denoised_window

    return denoised_signal


def apply_wavelet_transform_to_df(df, column_name, wavelet='db6', mode='soft', level=None, window_size=100):
    transformed_df = df.copy()

    for ticker in df['TICKER'].unique():
        signal = df[df['TICKER'] == ticker][column_name]
        transformed_signal = wavelet_transform(signal, wavelet, mode, level, window_size=window_size)
        transformed_df.loc[transformed_df['TICKER'] == ticker, column_name] = transformed_signal

    return transformed_df



def wavelet_mra(signal, wavelet='db6', level=None):
    # Handle NaNs in the signal
    if np.any(np.isnan(signal)):
        return [np.full_like(signal, np.nan)] * (level + 1)  # Return arrays of NaNs for each level

    # Determine the maximum level of decomposition if not specified
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)

    # Perform Discrete Wavelet Transform
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Generate a list of arrays for each decomposition level
    mra_components = []
    for i in range(level):
        # For each level, create a zero-filled array for detail coefficients
        detail_coeffs = [np.zeros_like(signal) for _ in range(level)]
        # Place the detail coefficients of the current level in the appropriate position
        detail_coeffs[i] = pywt.waverec([c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)], wavelet)[:len(signal)]
        mra_components.append(detail_coeffs[i])

    # Add the approximation coefficients at the last level
    approximation = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)[:len(signal)]
    mra_components.append(approximation)

    return mra_components

def apply_wavelets_to_df(df, column_name, wavelet='db2', mode='soft', level=None, window_size=100):
    transformed_df = df.copy()
    wavelets = {}
    
    # Determine the level if not provided
    if level is None:
        # Assuming the signal length is consistent for all tickers
        sample_signal = df[column_name].dropna()
        level = pywt.dwt_max_level(len(sample_signal), pywt.Wavelet(wavelet).dec_len)

    for ticker in df['TICKER'].unique():
        signal = df[df['TICKER'] == ticker][column_name]
        transformed_signals = wavelet_mra(signal, wavelet, level)
        wavelets[ticker] = transformed_signals

    # Add new wavelet columns to transformed df
    for i in range(level + 1):
        transformed_df['wavelet_' + str(i)] = np.nan

    for ticker in df['TICKER'].unique():
        for i in range(level + 1):
            indices = transformed_df['TICKER'] == ticker
            transformed_df.loc[indices, 'wavelet_' + str(i)] = wavelets[ticker][i][:len(indices)]

    return transformed_df

