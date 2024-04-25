
import torch
import sys
import torch.nn as nn
sys.path.append('trainers')
sys.path.append('analysis')
sys.path.append('utils')
from config import config
import pickle
from pathlib import Path
from pytorch_trainer import selective_train_and_save_model,load_data
from model_analysis import  create_study_periods_parallel, create_tensors, analyze_portfolio_performance,generate_portfolio_with_simplified_metrics
from cnn_transformer import CNNTransformer
from datetime import datetime
from joblib import Parallel, delayed
import quantstats as qs
import pandas_market_calendars as mcal
import os
import tqdm
import numpy as np
import pandas as pd
import optuna
import json
import torch.optim as optim
from data_splitter import save_train_test_splits
from transformer_model import TimeSeriesTransformer
import pickle

def main():
#     #Load data
#load in datasets dict

    df, start_date, end_date = load_data(config['data_path'], config['start_date'])
    start_date=df['date'].min()
    end_date=df['date'].max()
    # df=df.reset_index()
    # df=df.drop(['level_0'],axis=1)
    if len(config['tickers'])!=0:
        df = df[df['TICKER'].isin(config['tickers'])]
    if len(config['features'])==0:
        config['features']=[col for col in df.columns if col.startswith('emb')]
        # config['features']=[col for col in df.columns if col not in ['date', 'TICKER','PERMNO',"PRC",	"VOL",		"NUMTRD",	"MktCap"]]
        config['features'].append('RET')
        # config['features'] = [col for col in df.columns if col not in ['date', 'TICKER',"PERMNO"]]
        # config['features']=[col for col in df.columns if col.startswith('alpha')]
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
    # df = df.rename(columns={'S_DQ_PCTCHANGE': 'RET'})
    if 'RET' not in config['features']:
        config['features'].append('RET')
    df = df[['date', 'TICKER','volatility_decile'] + config['features']]
    df=df.dropna()
    decile_ranges = {
        # "top_10%":(9,9),
        # "top_20%":(8,9), # Including the first two deciles # Including the first three deciles
        # "top_50%":(5,9),
        # "top_100%": (0, 9), # Including all deciles
    }
    for curriculum in decile_ranges:
        start_decile, end_decile = decile_ranges[curriculum]
# Filter the DataFrame to include all deciles within the specified range
        df_filtered = df[(df['volatility_decile'] >= start_decile) & (df['volatility_decile'] <= end_decile)]


        
# # #     # # Prepare study periods
        study_periods = create_study_periods_parallel(
            df_filtered,
            train_size=800,
            val_size=40,
            trade_size=160,
            start_date=start_date,
            end_date=end_date,
            target_type=config['target'],
            standardize=True,
            data_type=config['data_type'],
    features=config['features'],use_autoencoder=False
        
        )
        #subset target balance:target
    # 0    20139
    # 1    18859
    # dtype: int64
    #     split_periods_by_decile = {}  # Prepare a dictionary to hold the filtered periods for each decile

    # # # Loop through each decile
    #     for decile in range(10):  # Deciles are 0 (highest volatility) to 3 (lowest volatility)
    #         split_periods = []  # Prepare a list to hold the filtered periods for the current decile

    #         for period in study_periods:  # Loop through each period
    #             # Filter each DataFrame in the period by the decile, creating a new filtered period tuple
    #             filtered_period = tuple(df[df['volatility_decile'] == decile] for df in period)
                
    #             # Add the filtered period to the list of split periods for the current decile
    #             split_periods.append(filtered_period)
            
    #         # Add the split periods for the current decile to the dictionary
    #         split_periods_by_decile[f"decile_{decile}"] = split_periods
            
    #     feature_columns = config['features']
    #     if len(feature_columns)==0:
    #         feature_columns
        feature_columns=config['features']
        if 'RET' in feature_columns:
            feature_columns.remove('RET')

    #     decile_ranges = { # Including the first two deciles # Including the first three deciles
    #         # "top_50%":(0,5),
    #         "top_100%": (0, 9),
    #         "top_10%":(9,9) # Including all deciles
    #     }


    #     # Path for saving processed data

    #     for label, (start_decile, end_decile) in decile_ranges.items():
    #         print(f"Processing: {label}")

    #         # Initialize an empty list to hold the processed periods for this subset
    #         processed_periods = []

    #         # Process each period according to the current decile range
    #         for period in study_periods:  # study_periods is a list of tuples (train_df, val_df, test_df)
    #             filtered_period = tuple(
    #                 df[(df['volatility_decile'] >= start_decile) & (df['volatility_decile'] <= end_decile)]
    #                 for df in period
    #             )
    #             processed_periods.append(filtered_period)

            # At this point, 'processed_periods' contains the same number of periods as the original
            # but filtered for the current decile range. Now, you can create tensors for these periods.
        train_test_splits = create_tensors(study_periods, config['sequence_length'], feature_columns, min(8, len(study_periods)), config['auto_encoder'])
        
            
            # Create tensors for the combined periods
            # n_jobs = min(10, len(combined_periods))
            # train_test_splits = create_tensors(combined_periods, config['sequence_length'], config['features'], n_jobs, config['auto_encoder'])
            
            # Save the tensor data in a dedicated folder for the subset
        subset_folder = os.path.join(config['split_periods_path'], curriculum)
        if not os.path.exists(subset_folder):
            os.makedirs(subset_folder)
        
        # Save the train_test_splits in the corresponding folder
        # path = os.path.join(subset_folder, 'train_test_splits.pt')
        torch.save({'train_test_splits': train_test_splits}, config['tensors_path'])
        # p=config['split_periods_path']
        # path=f'{p}/subset_{subset_labels[subset_idx]}'
        save_train_test_splits(config['tensors_path'],subset_folder)
#         # #     # #Changing train data
#         loaded_data = torch.load(config['tensors_path'])
#         train_test_splits = loaded_data['train_test_splits']
#         split_periods=config['split_periods_path']
#         path=f'{subset_folder}'
    # # feature_columns=['RET']

    # n_jobs = min(10, len(study_periods))
    # # # #Single variate
    # train_test_splits= create_tensors(study_periods, config['sequence_length'], feature_columns,n_jobs,config['auto_encoder'])
    # torch.save({'train_test_splits':train_test_splits},config['tensors_path'])
    task_type = 'classification'

# #     # Load the configuration for model features
    # n_periods=len(train_test_splits)
    feature_columns = len(config['features'])
    # Remove returns if needed
    feature_columns-=1


    #FOR GRID SEARCH
    # period_data = prepare_period_data(config,n_periods)  # Ensure this is defined
    
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100)  # Number of trials
    

#     # Initialize the model factory # Define or fetch the number of stocks if necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f'Device: {device}')
    criterion=nn.CrossEntropyLoss()


    model=CNNTransformer(num_classes=config['num_classes'] + 1)

    model.to(device)
    if config['selective']:
        model,reservation_weight,score_weight = selective_train_and_save_model(model=model, criterion=criterion, periods_folder=config['split_periods_path'], device=device,
                                               initial_reward=config['reward'],
                                               num_classes=config['num_classes'] + 1)
    # else:
    #     # Define or adjust your train_and_save_model function accordingly
    #     model = train_and_save_model(model, criterion, train_test_splits_path, device,
    #                                  config['model_config'], 'best_model.pth')

#     # Save the trained model
    torch.save(model.state_dict(), config['model_path'])
    print(f"Model saved to {config['model_path']}!")
    # # os.system('sudo shutdown now')
# #     # # # Testing
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    input_file = config['tensors_path']
    output_dir = config['split_periods_path']
    # save_train_test_splits(input_file, output_dir)
    period_data = {}  # Dictionary to store data for each period
    path=config['split_periods_path']
    dataset_folder = f'{path}/top_10%'
    files = os.listdir(dataset_folder)
    period_files = [file for file in files if file.startswith("period_")]
    n_periods=len(period_files)
    for period_index in range(n_periods):
        file_path = os.path.join(dataset_folder, f'period_{period_index}.pt')
        period_split = torch.load(file_path, map_location='cpu')['test']
        
        test_data_tensor, test_labels_tensor, test_tickers, test_dates = period_split
        
        # Convert and filter the data for the current period
        filtered_test = [(ticker, date, data, label) 
                        for ticker, date, data, label in zip(test_tickers, test_dates, test_data_tensor, test_labels_tensor)]
        
        # Store filtered data for the current period
        period_data[period_index] = filtered_test

    if config['selective']==False:
        predictions = generate_predictions(model, test_data,device)
        stats=analyze_portfolio_performance(df,predictions)
        print("Portfolio Statistics:", stats)
    else:
        k=10
        combined_portfolios_performance = pd.DataFrame()
        combined_k_portfolios_performance = pd.DataFrame()
        for period in period_data.keys():
            test_data=period_data[period]
            date_start=test_data[0][1].strftime('%Y-%m-%d')
            
            daily_portfolios,daily_k_portfolio,portfolio_accuracies, mean_accuracy, misclassification_rates, correlation_coefficient=generate_portfolio_with_simplified_metrics(test_data,model,device,k,config['num_classes'])
            # Define a dictionary to store the variables
            data_to_save = {"mean_accuracy": mean_accuracy,
                            "portfolio_accuracies": portfolio_accuracies,
                            "misclassification_rates": misclassification_rates,
                            "correlation_coefficient": correlation_coefficient
                            }

            # Define the directory path
            dir_path = Path(config['results_path']) / f"period_{date_start}"

            # Ensure the directory exists
            dir_path.mkdir(parents=True, exist_ok=True)

            # Save model stats to JSON
            # with open(dir_path / 'model_stats.json', 'w') as json_file:
            #     json.dump(data_to_save, json_file)
            # Assuming analyze_portfolio_performance returns something that can be saved with qs.reports.html
            result = analyze_portfolio_performance(df, daily_portfolios)
            result2 = analyze_portfolio_performance(df, daily_k_portfolio)
            combined_portfolios_performance = pd.concat([combined_portfolios_performance, result])
            combined_k_portfolios_performance = pd.concat([combined_k_portfolios_performance, result2])

            # # Save QuantStats reports to HTML
            # qs.reports.html(result, output=dir_path / 'results.html')
            # qs.reports.html(result2, output=dir_path / 'k_results.html')

            # all_daily_portfolios_list = []
            # for date, date_data in daily_portfolios.items():
            #     for side, p in date_data.items():
            #         p['Date'] = date
            #         p['Side'] = side
            #         all_daily_portfolios_list.append(p)

            # all_daily_portfolios_df = pd.concat(all_daily_portfolios_list, ignore_index=True)
            # all_daily_portfolios_df.to_csv(dir_path / f'period_{date_start}_all_daily_portfolios.csv', index=False)

            # # Prepare and save all daily k_portfolios to a single CSV file
            # all_daily_k_portfolios_list = []
            # for date, date_data in daily_k_portfolio.items():
            #     for side, p in date_data.items():
            #         p['Date'] = date
            #         p['Side'] = side
            #         all_daily_k_portfolios_list.append(p)

            # all_daily_k_portfolios_df = pd.concat(all_daily_k_portfolios_list, ignore_index=True)
            # all_daily_k_portfolios_df.to_csv(dir_path / f'period_{date_start}_all_daily_k_portfolios.csv', index=False)


if __name__ == "__main__":
    main()
