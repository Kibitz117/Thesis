import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
import math
from datetime import datetime
#import relu
from torch.nn import functional as F

#Idea make sharpe ratio loss and analyze model confidence with attention(reject stock if confidence is low)
#Test this with StoAttention and Normal Attention
#Tune confidence threshold
class ScaledMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(ScaledMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads  # Dimension of each head: commonly referred to as d_k

        self.fc_q = nn.Linear(d_model, d_model)  # Query
        self.fc_k = nn.Linear(d_model, d_model)  # Key
        self.fc_v = nn.Linear(d_model, d_model)  # Value
  # Value
        self.fc_o = nn.Linear(d_model, d_model)  # Output

        self.relative_positional_encoding = RelativePositionalEncoding(self.head_dim)
        self.reset_parameters()
    @staticmethod
    def create_look_ahead_mask(batch_size, sequence_length):
        mask = torch.triu(torch.ones((sequence_length, sequence_length)), diagonal=1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask  # Returns 1 for positions to be attended to and 0 for positions to be masked


    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        length = query.size(1)


        Q = self.fc_q(query).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.fc_k(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.fc_v(value).view(batch_size, -1, self.num_heads, self.head_dim)



        # Calculate Q * K^T attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Incorporate relative positional embeddings
        # Inside ScaledMultiHeadAttention's forward method
       # Inside the ScaledMultiHeadAttention forward method:
        # Inside the ScaledMultiHeadAttention forward method:
        relative_positions_embeddings = self.relative_positional_encoding(length, batch_size, self.num_heads).to(query.device)

         # Now add them to the attention scores


        attention_scores = attention_scores + relative_positions_embeddings



        # Apply mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) 


            attention_scores = attention_scores.masked_fill(mask == 0, float('-1e10'))
        
        # Calculate attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        value_out = torch.matmul(attention_weights, V)
        value_out = value_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_o(value_out)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.fc_k.reset_parameters()
        self.fc_v.reset_parameters()
        self.fc_o.reset_parameters()


class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=.1):
        super(PositionWiseFeedForward,self).__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        #RELU activation
        x=self.linear1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = ScaledMultiHeadAttention(num_heads, d_model)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):


        # Self attention
        attention = self.self_attention(x, x, x, mask)
        # Add and norm
        x = self.layer_norm1(x + self.dropout(attention))
        # Position wise feed forward
        feed_forward = self.position_wise_feed_forward(x)
        # Add and norm
        x = self.layer_norm2(x + self.dropout(feed_forward))

        return x

#Relative positional encoding to capture short term patterns
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#         self.encoding[:, 0::2] = torch.sin(position * div_term)
#         self.encoding[:, 1::2] = torch.cos(position * div_term)
#         self.encoding = self.encoding.unsqueeze(0)
    
#     def forward(self, x):
#         return self.encoding[:, :x.size(1)].detach()
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
        super(RelativePositionalEncoding, self).__init__()
        # Initialize the relative positions matrix for each position, not for each head
        self.relative_positions_matrix = nn.Parameter(torch.randn((max_len * 2 - 1, d_model)), requires_grad=True)
        self.max_len = max_len
        self.d_model = d_model  # Assume d_model is divisible by num_heads

    def forward(self, length, batch_size, num_heads):
        # Create a matrix for relative positions ranging from -length+1 to length-1
        positions = torch.arange(length).unsqueeze(0) - torch.arange(length).unsqueeze(1)
        # Shift values to be >= 0 and clamp to valid indices
        positions = positions + self.max_len - 1
        positions = positions.long().clamp_(0, self.max_len * 2 - 2)
        
        # Fetch the embeddings for the relative positions
        relative_positions_embeddings = self.relative_positions_matrix.index_select(0, positions.view(-1))
        relative_positions_embeddings = relative_positions_embeddings.view(length, length, -1)
        
        # Prepare for broadcasting over num_heads by adding an extra dimension for heads
        relative_positions_embeddings = relative_positions_embeddings.unsqueeze(0).unsqueeze(0)
        
        # Expand embeddings to include batch size and the same encoding for all heads
        relative_positions_embeddings = relative_positions_embeddings.expand(batch_size, num_heads, -1, -1, -1)
        
        # Now the shape of relative_positions_embeddings should be [batch_size, num_heads, seq_length, seq_length, d_model]
        return relative_positions_embeddings






class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, dropout=0.1, task_type='regression', num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.task_type = task_type
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        
        if task_type == 'classification':
            self.fc = nn.Linear(d_model, num_classes)
        else:  # regression
            self.fc = nn.Linear(d_model, 1)

    def forward(self, src, src_mask=None):
        # Ensure src is a float tensor
        src = src.float()

        # Scale input embeddings (removed positional encoding addition)
        src = src * math.sqrt(self.d_model)

        # Pass through each layer of the encoder
        for layer_idx, layer in enumerate(self.encoder_layers):

            src = layer(src, src_mask)

        # Capture the context from the last time step of the encoded sequence
        context = src[:, -1, :]
        
        # Final linear layer
        output = self.fc(context)

        if self.task_type == 'classification':
            output = nn.functional.log_softmax(output, dim=-1)

        return output

