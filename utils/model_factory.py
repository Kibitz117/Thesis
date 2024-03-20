import torch
import torch.nn as nn
import sys
sys.path.append('models')
sys.path.append('utils')
from transformer_model import TimeSeriesTransformer
from cnn_transformer import CNNTransformer

class ModelFactory:
    def __init__(self):
        self.models = {
            'transformer': TimeSeriesTransformer,
            'cnn_transformer':CNNTransformer
            # Add other models here
        }
        self.losses = {
            'bce': nn.BCEWithLogitsLoss(),
            'ce':nn.CrossEntropyLoss(),
            'mse': nn.MSELoss(),
        }
        self.task_types = {
            'classification': 'classification',
            'regression': 'regression',
        }

    def create(self, model_name,  loss_name, selective=False, model_config={}, num_classes=None,):
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        if loss_name not in self.losses:
            raise ValueError(f"Unknown loss: {loss_name}")

        # Update the model configuration to include the number of classes
        model_config_updated = model_config.copy()
        if selective==True:
            num_classes=num_classes+1
        if num_classes:
            model_config_updated['num_classes'] = num_classes


        # model = self.models[model_name](**model_config_updated, task_type=target_type)
        model = self.models[model_name](**model_config_updated)
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

