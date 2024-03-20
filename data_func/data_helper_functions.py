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
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas_market_calendars as mcal
from auto_encoder import StockAutoencoder
from sklearn.linear_model import LinearRegression

def create_study_period(p_idx, trading_days, df, train_size, val_size, trade_size, target_type, standardize, data_type, features):
    # Calculate the indexes for the end of each period
    train_end_idx = p_idx + train_size
    val_end_idx = train_end_idx + val_size
    trade_end_idx = val_end_idx + trade_size

    # Get the end dates for each period
    train_end = trading_days[train_end_idx - 1]  # -1 because the end day is inclusive
    val_end = trading_days[val_end_idx - 1] if val_end_idx < len(trading_days) else trading_days[-1]
    trade_end = trading_days[trade_end_idx - 1] if trade_end_idx < len(trading_days) else trading_days[-1]

    # Split the dataframe into the periods
    train_df = df[(df['date'] >= trading_days[p_idx]) & (df['date'] <= train_end)].copy()
    val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    trade_df = df[(df['date'] > val_end) & (df['date'] <= trade_end)].copy()

    # Standardize and create targets if necessary
    if standardize:
        train_df, val_df, trade_df = standardize_data(train_df, val_df, trade_df, features)
        train_df, val_df, trade_df = create_targets(train_df, val_df, trade_df, target_type, data_type)

    return (train_df, val_df, trade_df)





def create_study_periods_parallel(df,train_size, val_size,trade_size, start_date, end_date, target_type='cross_sectional_median', standardize=True, data_type='RET',features=None):
    # Create a calendar for NYSE
    nyse = mcal.get_calendar('NYSE')
    
    # Get the trading days within the specified date range
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    trading_days = trading_days.tz_localize(None)
    
    # Calculate the start indexes for each study period based on trading days
    study_period_indexes = []
    for i in range(0, len(trading_days), trade_size):
        if i + train_size + trade_size > len(trading_days):
            break  # Stop if there aren't enough days left for a full study period
        study_period_indexes.append(i)

    study_periods = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for p_idx in study_period_indexes:
            # Submit the task to the executor
            # Collect futures in order of submission
            futures = [executor.submit(create_study_period, p_idx, trading_days, df, train_size, val_size, trade_size, target_type, standardize, data_type, features) for p_idx in study_period_indexes]

            # Wait for all futures to complete and collect results in submission order
            study_periods = [future.result() for future in tqdm(futures, total=len(futures)) if future.result() is not None]


    return study_periods

def standardize_data(train_df, val_df, trade_df, features):
    """
    Standardizes continuous features based on the training set.
    A feature is considered continuous if it has more than 20 unique values.
    """
    for feature in features:
        if train_df[feature].nunique() > 20:
            mu = train_df[feature].mean()
            sigma = train_df[feature].std()
            if sigma > 0:
                train_df[feature] = (train_df[feature] - mu) / sigma
                val_df[feature] = (val_df[feature] - mu) / sigma
                trade_df[feature] = (trade_df[feature] - mu) / sigma

    return train_df, val_df, trade_df
def create_targets(train_df, val_df, trade_df, target_type, data_type):
    if target_type == 'direction':
        train_df['target'] = (train_df[data_type] > 0).astype(int)
        val_df['target'] = (val_df[data_type] > 0).astype(int)
        trade_df['target'] = (trade_df[data_type] > 0).astype(int)

    elif target_type in ['cross_sectional_median', 'cross_sectional_mean']:
        for df in [train_df, val_df, trade_df]:
            if target_type == 'cross_sectional_median':
                cross_sectional_value = df.groupby('date')[data_type].transform('median')
            else:
                cross_sectional_value = df.groupby('date')[data_type].transform('mean')
            df['target'] = df[data_type] >= cross_sectional_value
            df['target'] = df['target'].astype(int)

    elif target_type in ['buckets', 'quintiles']:
        num_buckets = 5 if target_type == 'quintiles' else 3
        for df in [train_df, val_df, trade_df]:
            df['target'] = pd.qcut(df.groupby('date')[data_type], num_buckets, labels=False, duplicates='drop')
            df['target'].fillna(0, inplace=True)
            df['target'] = df['target'].astype(int)

    elif target_type == 'raw_return':
        train_df['target'] = train_df[data_type]
        val_df['target'] = val_df[data_type]
        trade_df['target'] = trade_df[data_type]

    else:
        print('Invalid target type')

    return train_df, val_df, trade_df

def process_with_autoencoder(window, input_features, target_feature, embedding_dim=8, num_epochs=500, batch_size=64):
    # Assuming 'df' contains your window's data with input features and a target feature
    
    # Initialize the Autoencoder
    input_size = len(input_features)
    autoencoder = StockAutoencoder(input_size=input_size, embedding_dim=embedding_dim)
    
    # Train the autoencoder
    features_df = window[input_features]
    scaler = autoencoder.train_autoencoder(features_df.values, num_epochs=num_epochs, batch_size=batch_size)
    
    # Generate embeddings
    embeddings = autoencoder.encode(features_df.values, scaler)

    model = LinearRegression().fit(embeddings, window[target_feature])
    # Calculate residuals
    return window[target_feature] - model.predict(embeddings)


def process_period(train_df, val_df, test_df, sequence_length, feature_columns, use_autoencoder=True):
    train_windows, train_labels, train_tickers, train_dates = [], [], [], []
    val_windows, val_labels, val_tickers, val_dates = [], [], [], []
    test_windows, test_labels, test_tickers, test_dates = [], [], [], []


    # Process train, val, and test datasets
    for dataset, windows, labels, tickers, dates in [
        (train_df, train_windows, train_labels, train_tickers, train_dates),
        (val_df, val_windows, val_labels, val_tickers, val_dates),
        (test_df, test_windows, test_labels, test_tickers, test_dates)
    ]:
        for ticker, group in dataset.groupby('TICKER'):
            group = group.sort_values('date')
            for i in range(sequence_length, len(group)):
                window = group.iloc[i-sequence_length:i][feature_columns].values
                label = group.iloc[i]['target']
                date = group.iloc[i]['date']

                if use_autoencoder:
                    # Assuming `feature_columns` are the input features for the autoencoder
                    residuals = process_with_autoencoder(group.iloc[i-sequence_length:i], feature_columns, 'RET', embedding_dim=8, num_epochs=200, batch_size=64)
                    # Now, use residuals as your new features
                    combined_features = residuals
                else:
                    combined_features = window

                windows.append(combined_features)
                labels.append(label)
                tickers.append(ticker)
                dates.append(date)

    # Convert lists to tensors
    train_tensor_data = torch.tensor(np.array(train_windows), dtype=torch.float32).unsqueeze(-1)
    train_tensor_labels = torch.tensor(np.array(train_labels), dtype=torch.long)
    val_tensor_data = torch.tensor(np.array(val_windows), dtype=torch.float32).unsqueeze(-1)
    val_tensor_labels = torch.tensor(np.array(val_labels), dtype=torch.long)
    test_tensor_data = torch.tensor(np.array(test_windows), dtype=torch.float32).unsqueeze(-1)
    test_tensor_labels = torch.tensor(np.array(test_labels), dtype=torch.long)

    return (train_tensor_data, train_tensor_labels, train_tickers, train_dates,
            val_tensor_data, val_tensor_labels, val_tickers, val_dates,
            test_tensor_data, test_tensor_labels, test_tickers, test_dates)












def create_tensors(study_periods, sequence_length, feature_columns, n_jobs=-1, use_autoencoder=True):
    tasks = [(period[0], period[1], period[2], sequence_length, feature_columns, use_autoencoder) for period in study_periods]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_period)(*task) for task in tqdm(tasks, desc="Processing rolling windows")
    )

    study_period_splits = []
    for result in results:
        study_period_split = {
            'train': (result[0], result[1], result[2], result[3]),  # Data, Labels, Tickers, Dates
            'val': (result[4], result[5], result[6], result[7]),  # Data, Labels, Tickers, Dates
            'test': (result[8], result[9], result[10], result[11])  # Data, Labels, Tickers, Dates
        }
        study_period_splits.append(study_period_split)

    return study_period_splits

def wavelet_mra(data, wavelet, level):
    # Perform Discrete Wavelet Transform
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Thresholding (Universal Threshold)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    denoised_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    
    # Reconstruction
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    
    return denoised_signal







# def wavelet_mra(signal, wavelet, level, index):
#     # Ensure the signal is at least as long as the minimum length required for the chosen level
#     min_length = pywt.Wavelet(wavelet).dec_len * (2 ** level)
#     if len(signal) < min_length:
#         return pd.DataFrame(np.zeros((len(signal), level + 1)), index=index)  # Return DataFrame of zeros for each level

#     # Perform Discrete Wavelet Transform
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
#     mra_components = []

#     # Generate arrays for each decomposition level including approximation at last level
#     for i in range(level + 1):
#         if i < level:  # For detail coefficients
#             reconstructed_detail = pywt.waverec([np.zeros_like(c) if j != i else c for j, c in enumerate(coeffs)], wavelet)[:len(signal)]
#             mra_components.append(reconstructed_detail)
#         else:  # For approximation at the last level
#             reconstructed_approximation = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)[:len(signal)]
#             mra_components.append(reconstructed_approximation)

#     # Correctly reconstruct the original signal from the wavelet coefficients
#     reconstructed_signal = pywt.waverec(coeffs, wavelet)[:len(signal)]

#     # Convert to DataFrame to maintain index association, including detail components and approximation
#     df_mra_components = pd.DataFrame(mra_components).T.set_index(index)
#     # df_mra_components['reconstructed_signal'] = reconstructed_signal  # Add correctly reconstructed signal to the DataFrame

#     return df_mra_components

# def apply_wavelets_to_df(df, column_name, wavelet='db6', level=3):
#     transformed_df = df.copy()

#     # Initialize columns for wavelet features and the reconstructed signal
#     for i in range(level + 1):
#         transformed_df[f'wavelet_{i}'] = np.nan  # Initialize with NaNs
#     # transformed_df['reconstructed_signal'] = np.nan  # Initialize reconstructed signal column

#     for ticker in df['TICKER'].unique():
#         # Extract signal and index for the current ticker
#         ticker_df = df[df['TICKER'] == ticker]
#         signal = ticker_df[column_name].values
#         index = ticker_df.index
        
#         # Apply wavelet transform and get a DataFrame of transformed signals
#         transformed_signals_df = wavelet_mra(signal, wavelet, level, index)

#         # Assign wavelet features and the reconstructed signal to the transformed DataFrame
#         for i in range(level + 1):
#             transformed_df.loc[index, f'wavelet_{i}'] = transformed_signals_df.iloc[:, i]
#         # transformed_df.loc[index, 'reconstructed_signal'] = transformed_signals_df['reconstructed_signal']

#     return transformed_df


