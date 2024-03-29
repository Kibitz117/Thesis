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
import torch.nn.init as init

#Idea make sharpe ratio loss and analyze model confidence with attention(reject stock if confidence is low)
#Test this with StoAttention and Normal Attention
#Tune confidence threshold
# class ScaledMultiHeadAttention(nn.Module):
#     def __init__(self, num_heads, d_model):
#         super(ScaledMultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0
#         self.num_heads = num_heads
#         self.d_model = d_model
#         self.head_dim = d_model // num_heads  # Dimension of each head: commonly referred to as d_k
#         self.fc_q = nn.Linear(d_model, d_model)  # Query
#         self.fc_k = nn.Linear(d_model, d_model)  # Key
#         self.fc_v = nn.Linear(d_model, d_model)  # Value

#         self.fc_o = nn.Linear(d_model, d_model)  # Output

#         self.relative_positional_encoding = RelativePositionalEncoding(self.d_model)
#         self.reset_parameters()
   

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
#         seq_length = query.size(1)
#         Q = self.fc_q(query).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         K = self.fc_k(key).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = self.fc_v(value).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         # Q = self.fc_q(query)
#         # K = self.fc_k(key)
#         # V = self.fc_v(value)

#         # Calculate attention scores with scaling for stability
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
#         # Generate relative positional embeddings and scores
#         relative_pos_embeddings = self.relative_positional_encoding(seq_length, batch_size, self.num_heads)
#         relative_position_scores = self.compute_relative_position_scores(Q, relative_pos_embeddings)
#         scores += relative_position_scores

#         # Apply a large negative number for masked positions instead of -inf
#         if mask is not None:
#             mask = mask.expand(batch_size, -1, -1, -1)
#             scores = scores.masked_fill(mask == 0, -1e9)  # Using a large negative number because numerical instability with -inf

#         # Apply softmax to get the attention weights
#         attention_weights = F.softmax(scores + 1e-9, dim=-1)

#         # Apply mask after softmax (if required) sets straggling small values to 0 to avoid numerical instability
#         if mask is not None:
#             attention_weights = attention_weights.masked_fill(mask == 0, 0)

#         # Apply attention to the value vector
#         output = torch.matmul(attention_weights, V)

#         # Concatenate heads and put through the final linear layer
#         output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)
#         output = self.fc_o(output)

#         return output

#     def compute_relative_position_scores(self, query, relative_pos_embeddings):
#         # Extract batch_size, num_heads, seq_length, and head_dim from the query's shape
#         batch_size, num_heads, seq_length,head_dim= query.size()

#         # Ensure that relative positional embeddings have the correct shape
#         # relative_pos_embeddings shape expected: [batch_size, num_heads, seq_length, seq_length, head_dim]
#         assert relative_pos_embeddings.size() == (batch_size, num_heads, seq_length, seq_length, head_dim), \
#             "Shape mismatch in relative_pos_embeddings"

#         # Compute attention scores using einsum
#         attn_scores = torch.einsum('bhqd,bhqkd->bhqk', query, relative_pos_embeddings)

#         return attn_scores



        




#     def reset_parameters(self):
#         self.fc_q.reset_parameters()
#         self.fc_k.reset_parameters()
#         self.fc_v.reset_parameters()
#         self.fc_o.reset_parameters()


# class PositionWiseFeedForward(nn.Module):
#     def __init__(self,d_model,d_ff,dropout=.1):
#         super(PositionWiseFeedForward,self).__init__()
#         self.linear1=nn.Linear(d_model,d_ff)
#         self.linear2=nn.Linear(d_ff,d_model)
#         self.dropout=nn.Dropout(dropout)
#     def forward(self,x):
#         #RELU activation
#         x=self.linear1(x)
#         x=F.relu(x)
#         x=self.dropout(x)
#         x=self.linear2(x)
#         return x

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
#         super(EncoderLayer, self).__init__()
#         self.self_attention = ScaledMultiHeadAttention(num_heads, d_model)
#         self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.layer_norm2 = nn.LayerNorm(d_model)

#     def forward(self, x, mask):
#     # Self attention
#         attention = self.self_attention(x, x, x, mask)
        
#         # Add and norm
#         x = x + self.dropout(attention)
#         x = self.layer_norm1(x)
#         #Debug statement
#         # x.register_hook(lambda grad: print("Gradient norm after LayerNorm1:", torch.norm(grad).item()))

#         # Position wise feed forward
#         feed_forward = self.position_wise_feed_forward(x)

#         # Add and norm
#         x = x + self.dropout(feed_forward)
#         x = self.layer_norm2(x)
#         # x.register_hook(lambda grad: print("Gradient norm after LayerNorm2:", torch.norm(grad).item()))

#         return x

# #Relative positional encoding to capture short term patterns
# # class PositionalEncoding(nn.Module):
# #     def __init__(self, d_model, max_len=5000):
# #         super(PositionalEncoding, self).__init__()
# #         self.encoding = torch.zeros(max_len, d_model)
# #         position = torch.arange(0, max_len).unsqueeze(1).float()
# #         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
# #         self.encoding[:, 0::2] = torch.sin(position * div_term)
# #         self.encoding[:, 1::2] = torch.cos(position * div_term)
# #         self.encoding = self.encoding.unsqueeze(0)
    
# #     def forward(self, x):
# #         return self.encoding[:, :x.size(1)].detach()
# class RelativePositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super(RelativePositionalEncoding, self).__init__()
#         self.max_len = max_len
#         self.d_model = d_model
#         self.embeddings_table = nn.Parameter(torch.Tensor(max_len * 2 + 1, d_model))
#         nn.init.xavier_uniform_(self.embeddings_table)

#     def forward(self, length, batch_size, num_heads):
#         head_dim = self.d_model // num_heads
#         assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         range_vec = torch.arange(length)
#         distance_mat = range_vec[None, :] - range_vec[:, None]
#         clipped_distance_mat = torch.clamp(distance_mat, -self.max_len, self.max_len)
#         relative_position_mat = clipped_distance_mat + self.max_len

#         embeddings = self.embeddings_table[relative_position_mat]

#         # Reshape to add the head dimension and permute to get the correct order
#         embeddings = embeddings.view(length, length, num_heads, head_dim).permute(2, 0, 1, 3)

#         # Expand to include the batch dimension
#         embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

#         return embeddings











# class TimeSeriesTransformer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, input_features=1, dropout=0.3, task_type='regression', num_classes=2):
#         super(TimeSeriesTransformer, self).__init__()

#         self.d_model = d_model
#         self.num_heads = num_heads  # Store num_heads as an instance attribute
#         self.task_type = task_type

#         # Adjusted input projection layer for multiple input features
#         self.input_projection = nn.Conv1d(in_channels=input_features, out_channels=d_model, kernel_size=3, padding=1)
#         init.xavier_uniform_(self.input_projection.weight)
#         init.zeros_(self.input_projection.bias)

        
#         # Encoder layers
#         self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        
#         self.fc = nn.Linear(d_model, num_classes)
#         init.xavier_uniform_(self.fc.weight)
#         init.zeros_(self.fc.bias)

#     def forward(self, src, src_mask=None):
#         # Ensure src is a float tensor
#         src = src.float()

#         # Input processing
#         src = src.permute(0, 2, 1)  # Flattening to [batch_size, sequence_length, features]
#         src = self.input_projection(src) 
#         src = src.permute(0, 2, 1)  # [batch_size, sequence_length, d_model]
#         src *= math.sqrt(self.d_model)


#         # Encoder
#         for layer in self.encoder_layers:
#             src = layer(src, src_mask)

#         # Context from the last time step
#         context = src[:, -1, :]
        
#         # Output layer
#         output = self.fc(context)

#         return output


#     @staticmethod
#     def create_lookahead_mask(size):
#         mask = torch.triu(torch.ones(size, size), diagonal=1)
#         mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
#         return mask.unsqueeze(0).unsqueeze(1)




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, input_features=1, dropout=0.3, num_classes=2):
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels=input_features, out_channels=d_model, kernel_size=3, padding=1)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, num_classes)
        self.src_mask = None

    def forward(self, src):
        src = src.permute(0, 2, 1)  # Change to [batch, features, seq_length]
        src = self.input_projection(src)
        src = src.permute(2, 0, 1)  # Change to [seq_length, batch, features] as expected by Transformer
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self.create_lookahead_mask(src.size(0)).to(src.device)
        output = self.transformer_encoder(src, mask=self.src_mask)
        output = output[-1]  # Use the last sequence output
        return self.fc_out(output)

    @staticmethod
    def create_lookahead_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)
        return mask
