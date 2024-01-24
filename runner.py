import torch
import sys
sys.path.append('trainers')
sys.path.append('analysis')
sys.path.append('utils')
from config import config
from pytorch_trainer import train_and_save_model, selective_train_and_save_model,load_data
from model_analysis import  create_study_periods, create_tensors, generate_predictions,generate_predictions_with_abstention, analyze_portfolio_performance
from model_factory import ModelFactory

def main():
    # Load data
    df, start_date, end_date = load_data(config['data_path'], config['features'], config['start_date'])
    if len(config['tickers'])!=0:
        df = df[df['TICKER'].isin(config['tickers'])]

    # Prepare study periods
    study_periods = create_study_periods(
        df,
        window_size=240,
        trade_size=250,
        train_size=750,
        forward_roll=250,
        start_date=start_date,
        end_date=end_date,
        target_type=config['target'],
        data_type=config['data_type'],
        apply_wavelet_transform=True
    )

    # Prepare training and testing data
    feature_columns = [col for col in study_periods[0][0].columns if col not in ['date', 'TICKER', 'target']]
    config['model_config']['input_features'] = len(feature_columns)
    n_jobs = min(10, len(study_periods))
    train_test_splits, ticker_tensor_mapping,ticker_date_mapping, task_type = create_tensors(study_periods, n_jobs, config['sequence_length'], feature_columns)

    # Train the model
    factory = ModelFactory()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model, criterion = factory.create(config['model_name'], task_type, config['loss_func'], selective=config['selective'], model_config=config['model_config'], num_classes=config['num_classes'])
    model.to(device)
    if config['selective']:
        model = selective_train_and_save_model(model, criterion, train_test_splits, device, config['model_config'], 'best_model.pth')
    else:
        model = train_and_save_model(model, criterion, train_test_splits, device, config['model_config'], 'best_model.pth')

    # Save the trained model
    torch.save(model.state_dict(), config['model_path'])

    # Testing
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()
    test_data = [(ticker, date, data.to(device), label) for ticker, date, data, label in ticker_date_mapping['test']]
    if config['selective']==False:
        predictions = generate_predictions(model, test_data,device)
    else:
        predictions,abstentions = generate_predictions_with_abstention(model,test_data,config['reward'],device)
    portfolio = construct_portfolio(predictions, bucket_strategy=False)
    stats = analyze_portfolio_performance(df, portfolio)

    print("Portfolio Statistics:", stats)

if __name__ == "__main__":
    main()
