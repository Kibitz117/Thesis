# Configuration parameters
config = {
    'model_name': 'transformer',
    'target': 'cross_sectional_median',
    'num_classes': 2,
    'data_type': 'RET',
    #extra features:'Mkt-RF','SMB','HML','RF'
    'features': ['RET','RF'],
    'selective': False,
    'reward':2.5,
    'sequence_length': 240,
    'loss_func': 'ce',
    'model_config': {
        'd_model': 128,
        'num_heads': 2,
        'd_ff': 256,
        'num_encoder_layers': 3,
        'dropout': 0.1,
    },
    'data_path': 'data/stock_data_with_factors.csv',
    'start_date': '2000-01-01',
    'tickers': [],
    'model_path': 'model_state_dict.pth'
}
#'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW'