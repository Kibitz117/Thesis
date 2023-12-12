import torch
import torch.nn as nn
import sys
sys.path.append('models')
sys.path.append('utils')
from transformer_model import TimeSeriesTransformer
from selective_loss import SelectiveLoss
from sharpe_loss import SharpeLoss

class ModelFactory:
    def __init__(self):
        self.models = {
            'transformer': TimeSeriesTransformer,
            # Add other models here
        }
        self.losses = {
            'bce': nn.BCEWithLogitsLoss(),
            'ce':nn.CrossEntropyLoss(),
            'mse': nn.MSELoss(),
            'sharpe': SharpeLoss(),
            'selective': SelectiveLoss(loss_func=nn.MSELoss(),coverage=.8),
        }
        self.task_types = {
            'classification': 'classification',
            'regression': 'regression',
        }

    def create(self, model_name, target_type, loss_name,selective=False, model_config={}):
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        if loss_name not in self.losses:
            raise ValueError(f"Unknown loss: {loss_name}")
        if target_type not in self.task_types:
            raise ValueError(f"Unknown target type: {target_type}")
        if selective==True:
            model = self.models[model_name](**model_config, task_type=target_type,num_classes=3)
        else:
            model = self.models[model_name](**model_config, task_type=target_type)
        loss = self.losses[loss_name]
        return model, loss

# Usage:
# factory = ModelFactory()

# model_config = {
#     'd_model': 64,
#     'num_heads': 8,
#     'd_ff': 256,
#     'num_encoder_layers': 2,
#     'dropout': .1,
# }

# loss_name = 'nll'  # or 'mse' or any other loss function name
# loss_config = {}  # Additional configurations for loss function if needed

# model, criterion = factory.create('transformer', 'classification', loss_name, model_config=model_config)
# model = model.to(device)

