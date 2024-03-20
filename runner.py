import torch
import sys
import torch.nn as nn
sys.path.append('trainers')
sys.path.append('analysis')
sys.path.append('utils')
from config import config
import pickle
from pytorch_trainer import selective_train_and_save_model,load_data
from model_analysis import  create_study_periods_parallel, create_tensors, analyze_portfolio_performance,generate_portfolio_with_simplified_metrics
from cnn_transformer import CNNTransformer
from datetime import datetime
from joblib import Parallel, delayed
import quantstats as qs
import os
import tqdm
import numpy as np
import pandas as pd
import optuna
import torch.optim as optim
from data_splitter import save_train_test_splits
from transformer_model import TimeSeriesTransformer
def prepare_period_data(config,n_periods):
    input_file = config['tensors_path']
    output_dir = config['split_periods_path']
    save_train_test_splits(input_file, output_dir)
    
    period_data = {}  # Ensure this is defined in your config
    
    for period_index in range(n_periods):
        file_path = os.path.join(output_dir, f'period_{period_index}.pt')
        period_split = torch.load(file_path, map_location='cpu')['test']
        
        test_data_tensor, test_labels_tensor, test_tickers, test_dates = period_split
        
        filtered_test = [(ticker, date, data, label) 
                         for ticker, date, data, label in zip(test_tickers, test_dates, test_data_tensor, test_labels_tensor)]
        
        period_data[period_index] = filtered_test
    return period_data
# def objective(trial):
#     # Suggest hyperparameters
#     device = torch.device("cuda" if torch.cuda.is_available() else "mps")
#     dropout = trial.suggest_float('dropout', 0.1, 0.5)
#     filter_size = trial.suggest_int('filter_size', 2,2)
#     hidden_units_factor = trial.suggest_categorical('hidden_units_factor', [4, 8,16,32])
#     learning_rate = trial.suggest_float('learning_rate', .01, .05)
    
#     embed_dim_second_value = trial.suggest_categorical('embed_dim_second_value', [16, 32, 64, 128, 256])
#     attention_heads = trial.suggest_categorical('attention_heads', [4, 8])

#     # Ensure embed_dim (filter_numbers[-1]) is divisible by num_heads
#     while embed_dim_second_value % attention_heads != 0:
#         # If not divisible, adjust attention_heads within the constraints
#         attention_heads = trial.suggest_categorical('attention_heads', [factor for factor in [1, 2, 4, 8] if embed_dim_second_value % factor == 0])

#     filter_numbers = [8, embed_dim_second_value]  # First value is fixed at 8
#     # Initialize the model with suggested hyperparameters
#     print(f"Trial {trial.number}:")
#     print(f"Device: {device}, Dropout: {dropout}, Filter Size: {filter_size}, Hidden Units Factor: {hidden_units_factor}, Learning Rate: {learning_rate}, Embed Dim Second Value: {embed_dim_second_value}, Attention Heads: {attention_heads}, Filter Numbers: {filter_numbers}")
#     model = CNNTransformer(
#         dropout=dropout,
#         filter_size=filter_size,
#         attention_heads=attention_heads,
#         filter_numbers=filter_numbers,
#         hidden_units=None,
#         hidden_units_factor=hidden_units_factor,
#         num_classes=config['num_classes'] + 1,  # +1 for abstention class
#     ).to(device)
    
#     # Setup optimizer
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
#     criterion = torch.nn.CrossEntropyLoss()
    
#     # Assuming selective_train_and_save_model integrates training and validation
#     model = selective_train_and_save_model(
#         model, criterion,optimizer, config['split_periods_path'], device,
#         'temp_best_model.pth', pretrain_epochs=config['pretrain_epochs'],
#         initial_reward=config['reward'], num_classes=config['num_classes'] + 1
#     )
#     loaded_data = torch.load(config['tensors_path'])
#     train_test_splits = loaded_data['train_test_splits']

# # #     # Load the configuration for model features
#     n_periods=len(train_test_splits)
#     period_data = prepare_period_data(config,n_periods) 
#     # Initialize lists to store metrics for each period
#     all_mean_accuracies = []
#     all_mean_abstention_accuracies = []
    
#     for period_index, test_data in period_data.items():
#         _, mean_accuracy, _, mean_abstention_accuracy, _ = generate_portfolio_with_simplified_metrics(
#             test_data, model, device, 10, config['num_classes']   # +1 for abstention
#         )
#         all_mean_accuracies.append(mean_accuracy)
#         all_mean_abstention_accuracies.append(mean_abstention_accuracy)
    
#     overall_mean_accuracy = sum(all_mean_accuracies) / len(all_mean_accuracies)
#     overall_mean_abstention_accuracy = sum(all_mean_abstention_accuracies) / len(all_mean_abstention_accuracies)
#     overall_performance = overall_mean_accuracy * 0.5 + overall_mean_abstention_accuracy * 0.5
    
#     return overall_performance
def main():
#     #Load data
    df, start_date, end_date = load_data(config['data_path'], config['start_date'])
    if len(config['tickers'])!=0:
        df = df[df['TICKER'].isin(config['tickers'])]
    if len(config['features'])==0:
        # config['features']=[col for col in df.columns if col.startswith('emb')]
        # config['features'].append('RET')
        config['features'] = [col for col in df.columns if col not in ['date', 'TICKER',"PERMNO"]]
        # config['features']=[col for col in df.columns if col.startswith('alpha')]
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
    # df = df.rename(columns={'S_DQ_PCTCHANGE': 'RET'})
    df = df[['date', 'TICKER'] + config['features']]
    df=df.dropna()

# Renaming the column 'S_DQ_PCTCHANGE' to 'RET'
    
# # #     # # Prepare study periods
#     study_periods = create_study_periods_parallel(
#         df,
#         train_size=650,
#         val_size=150,
#         trade_size=250,
#         start_date=start_date,
#         end_date=end_date,
#         target_type=config['target'],
#         standardize=True,
#         data_type=config['data_type'],
# features=config['features']
       
#     )

#     # # Prepare training and testing data
#     # feature_columns = config['features']
#     # if len(feature_columns)==0:
#     #     feature_columns
#     feature_columns=config['features']
#     feature_columns.remove('RET')

#     n_jobs = min(8, len(study_periods))
#     #Single variate
#     train_test_splits= create_tensors(study_periods, config['sequence_length'], feature_columns,n_jobs,config['auto_encoder'])
#     torch.save({'train_test_splits':train_test_splits},config['tensors_path'])
    task_type = 'classification'
    #Changing train data
    loaded_data = torch.load(config['tensors_path'])
    train_test_splits = loaded_data['train_test_splits']
    # save_train_test_splits(config['tensors_path'],config['split_periods_path'])

# #     # Load the configuration for model features
    n_periods=len(train_test_splits)
    feature_columns = 8
    #Remove returns if needed
    # feature_columns-=1


    #FOR GRID SEARCH
    # period_data = prepare_period_data(config,n_periods)  # Ensure this is defined
    
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100)  # Number of trials
    
    # best_trial = study.best_trial
    # print(f"Best trial: {best_trial.value}")
    # for key, value in best_trial.params.items():
    #     print(f"  {key}: {value}")
    # config['model_config']['input_features'] = feature_columns

#     # Initialize the model factory # Define or fetch the number of stocks if necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f'Device: {device}')
    criterion=nn.CrossEntropyLoss()






# # #     # Selective training
    model=CNNTransformer(num_classes=config['num_classes'] + 1)
    # model = TimeSeriesTransformer(**config['model_config'],input_features=feature_columns)

    model.to(device)
    if config['selective']:
        model = selective_train_and_save_model(model, criterion,None, config['split_periods_path'], device,
                                               config['model_config'], 'best_model.pth',
                                               pretrain_epochs=config['pretrain_epochs'],
                                               initial_reward=config['reward'],
                                               num_classes=config['num_classes'] + 1,n_periods=n_periods)
    else:
        # Define or adjust your train_and_save_model function accordingly
        model = train_and_save_model(model, criterion, train_test_splits_path, device,
                                     config['model_config'], 'best_model.pth')

#     # Save the trained model
    torch.save(model.state_dict(), config['model_path'])
    print(f"Model saved to {config['model_path']}!")
# #     # # os.system('sudo shutdown now')
# #     # # # Testing
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    input_file = config['tensors_path']
    output_dir = config['split_periods_path']
    # save_train_test_splits(input_file, output_dir)
    period_data = {}  # Dictionary to store data for each period
    path=config['split_periods_path']
    for period_index in range(n_periods):
        file_path = os.path.join(path, f'period_{period_index}.pt')
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
        for period in period_data.keys():
            test_data=period_data[period]
            date_start=test_data[0][1].strftime('%Y-%m-%d')
            daily_portfolios, mean_accuracy, misclassification_rates, mean_abstention_accuracy, correlation_coefficient=generate_portfolio_with_simplified_metrics(test_data,model,device,k,config['num_classes'])
            # Define a dictionary to store the variables
            data_to_save = {
            "daily_portfolios": daily_portfolios,
            "mean_accuracy": mean_accuracy,
            "misclassification_rates": misclassification_rates,
            "mean_abstention_accuracy": mean_abstention_accuracy,
            "correlation_coefficient": correlation_coefficient}

            # # Access the variables
            portfolio = data_to_save['daily_portfolios']

            result=analyze_portfolio_performance(df,portfolio)
            qs.reports.html(result, output=f'results/period_{date_start}.html')

if __name__ == "__main__":
    main()
