# Configuration parameters
config = {
    'model_name': 'cnn_transformer',
    'target': 'cross_sectional_median',
    'num_classes': 2,
    'data_type': 'RET',
    # 'features': ["RET","alpha025","alpha033","alpha009","alpha101","alpha010","alpha049","alpha001"],#I can try all alphas again
    'features':["PVT","RSI","MACD","VWAP","ATR","RET"],
    # "features":["RET","PVT","RSI","ROC","MACD","EMA_TradeCount","ATR","UpperBand","LowerBand","day"."SMA20"],
    'selective': True,
    #2,8 reward ~=50/60% rejection rate
    #Large reward means less abstain
    'reward':1.9,
    'sequence_length': 30,
    'loss_func': 'ce',
    # 'model_config': {
    #     'd_model': 256,
    #     'num_heads': 8,
    #     'd_ff': 512,
    #     'num_encoder_layers': 4,
    #     'dropout': 0.25,
    # },
#     'hyperparameter_grid' : {
#     'filter_numbers': [[8, 16], [8, 128]],
#     'attention_heads': [4, 8],
#     'dropout': [0.1, 0.5],
#     'filter_size': [2, 3],
#     'learning_rate': [1e-3, 1e-4]
# },
    'pretrain_epochs':50,
    'data_path': 'data/cir_technical_stock_data.parquet',
    'auto_encoder':False,
    'start_date': '2000-01-01',
    'tickers': [],
    'model_path': 'cir_learn.pth',
    'results_path':'cir_results2',
    'tensors_path':'data/train_test_splits',
    'split_periods_path':'data/split_periods_expanded_quintiles'
}
#'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW'
# 'IBM', 'AMZN','NVDA',AAPL','MSFT'