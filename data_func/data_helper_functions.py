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

def create_study_periods(df, n_periods, window_size, trade_size, train_size, forward_roll, start_date, end_date, target_type='cross_sectional_median', standardize=True, return_type='RET', apply_wavelet_transform=False):
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
        train_df[return_type] = pd.to_numeric(train_df[return_type], errors='coerce')
        trade_df[return_type] = pd.to_numeric(trade_df[return_type], errors='coerce')
        if train_df.empty or trade_df.empty:
            print(f"No data available for the period {train_start.date()} to {trade_end.date()}. Skipping.")
            continue

        # Drop other return columns and rename the specified return column to 'RET'
        return_cols = ['Adj_RET_Mkt', 'Adj_RET_Mkt_SMB', 'Adj_RET_Mkt_SMB_HML', 'RET']
        return_cols.remove(return_type)
        if return_cols in train_df.columns.to_list():
            train_df = train_df.drop(columns=return_cols)
            trade_df = trade_df.drop(columns=return_cols)
            train_df = train_df.rename(columns={return_type: 'RET'})
            trade_df = trade_df.rename(columns={return_type: 'RET'})
        if apply_wavelet_transform:
            # Apply wavelet transform to the 'RET' column of both training and trading DataFrames
            train_df = apply_wavelet_transform_to_df(train_df,'RET')
            trade_df = apply_wavelet_transform_to_df(trade_df,'RET')
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
        trade_df, train_df = create_targets(train_df, trade_df, target_type,window_size)

        # Store the training and trading DataFrames in the study_periods list
        study_periods.append((train_df, trade_df))

    return study_periods


def create_targets(train_df, trade_df, target,window_size):
    # if target == 'cross_sectional_median':
    #     # do stuff
    #     #VERIFY THAT THIS IS A ROLLING MEDIAN
    #     median_standardized_return = train_df['standardized_return'].median()
    #     train_df['target'] = (train_df['standardized_return'] >= median_standardized_return).astype(int)
    #     trade_df['target'] = (trade_df['standardized_return'] >= median_standardized_return).astype(int)

    if target == 'cross_sectional_median':
        # Calculate rolling median for each point in time in the training set
        # The window size for the rolling median can be set as needed
        rolling_median = train_df.groupby('TICKER')['standardized_return'].transform(lambda x: x.rolling(window=window_size, min_periods=1).median())

        # Assign binary targets based on whether the standardized return is above the rolling median
        train_df['target'] = (train_df['standardized_return'] >= rolling_median).astype(int)
        
        # For the trading set, use the last calculated rolling median from the training set
        last_rolling_median = rolling_median.iloc[-1]
        trade_df['target'] = (trade_df['standardized_return'] >= last_rolling_median).astype(int)


    elif target == 'raw_return':
        # Use raw returns as the target
        train_df['target'] = train_df['RET']
        trade_df['target'] = trade_df['RET']

    elif target == 'buckets':
        # Handle NaN or infinite values in 'standardized_return'
        for df in [train_df, trade_df]:
            # Replace infinite values with NaN
            df['standardized_return'].replace([np.inf, -np.inf], np.nan, inplace=True)
            # Fill NaN values with a specific value, e.g., the median
            df['standardized_return'].fillna(df['standardized_return'].median(), inplace=True)

        # Bucketing for train_df
        for ticker, group in train_df.groupby('TICKER'):
            # Create buckets with qcut
            buckets = pd.qcut(group['standardized_return'], 10, labels=False, duplicates='drop')
            # Assign buckets to the train_df
            train_df.loc[group.index, 'target'] = buckets

        # Bucketing for trade_df
        for ticker, group in trade_df.groupby('TICKER'):
            # Create buckets with qcut
            buckets = pd.qcut(group['standardized_return'], 10, labels=False, duplicates='drop')
            # Assign buckets to the trade_df
            trade_df.loc[group.index, 'target'] = buckets
    elif target == 'quintiles':
        # Handle NaN or infinite values in 'standardized_return'
        for df in [train_df, trade_df]:
            df['standardized_return'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['standardized_return'].fillna(df['standardized_return'].median(), inplace=True)

        # Quintile bucketing for train_df
        for ticker, group in train_df.groupby('TICKER'):
            quintiles = pd.qcut(group['standardized_return'], 5, labels=False, duplicates='drop')
            train_df.loc[group.index, 'target'] = quintiles

        # Quintile bucketing for trade_df
        for ticker, group in trade_df.groupby('TICKER'):
            quintiles = pd.qcut(group['standardized_return'], 5, labels=False, duplicates='drop')
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
def wavelet_transform(signal, wavelet='haar', mode='soft', level=None, control_coefficient=0.5):
    """
    Apply wavelet transform to denoise time series data using the Haar wavelet and a custom threshold.
    
    Args:
    signal (array-like): The time series data.
    wavelet (str): Type of wavelet to use, default is 'haar'.
    mode (str): Thresholding mode - 'soft' or 'hard', default is 'soft'.
    level (int): Level of decomposition. If None, max level is used.
    control_coefficient (float): Control coefficient for threshold adjustment.

    Returns:
    array: Denoised signal.
    """
    # Determine the maximum level of decomposition
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)

    # Perform Discrete Wavelet Transform
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Calculate the mean square error (sigma)
    sigma = np.sqrt(np.mean(coeffs[-1]**2))
    
    # Adjust the threshold based on the control coefficient
    threshold = control_coefficient * sigma

    # Apply thresholding based on the mode
    denoised_coeffs = []
    for coeff in coeffs:
        if mode == 'soft':
            denoised_coeff = pywt.threshold(coeff, value=threshold, mode=mode)
        elif mode == 'hard':
            denoised_coeff = pywt.threshold(coeff, value=threshold, mode=mode)
        denoised_coeffs.append(denoised_coeff)

    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    # Handle mismatch in lengths due to reconstruction
    if len(denoised_signal) != len(signal):
        denoised_signal = denoised_signal[:len(signal)]

    return denoised_signal
def apply_wavelet_transform_to_df(df, column_name, wavelet='db2', mode='soft', level=None):
    """
    Apply wavelet transform individually to each stock in the DataFrame.

    Args:
    df (DataFrame): DataFrame containing stock data.
    column_name (str): Name of the column to which the wavelet transform will be applied.
    wavelet, mode, level: Parameters for the wavelet transform.

    Returns:
    DataFrame: DataFrame with the wavelet-transformed column.
    """
    transformed_df = df.copy()

    for ticker in df['TICKER'].unique():
        # Select the time series for the current stock
        signal = df[df['TICKER'] == ticker][column_name]

        # Apply the wavelet transform to the time series
        transformed_signal = wavelet_transform(signal, wavelet, mode, level)

        # Assign the transformed signal back to the DataFrame
        transformed_df.loc[transformed_df['TICKER'] == ticker, column_name] = transformed_signal

    return transformed_df


