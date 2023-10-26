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
sys.path.append('../data_func')
from data_helper_functions import create_study_periods,create_tensors

df=pd.read_csv('../data/crsp_ff_adjusted.csv')
#drop unamed 0
df['date'] = pd.to_datetime(df['date'])
df.dropna(subset=['RET'],inplace=True)
df=df.drop(columns='Unnamed: 0')

#select returns to use
returns='RET'
df=df[['date','TICKER',f'{returns}']]
if returns!='RET':
    #rename returns column
    df.rename(columns={f'{returns}':'RET'},inplace=True)

#Optional parameter target_type: 'cross_sectional_median(default)','buckets(10 buckets)','raw_returns'.
study_periods=create_study_periods(df,n_periods=23,window_size=240,trade_size=250,train_size=750,forward_roll=250,start_date=datetime(1990,1,1),end_date=datetime(2015,12,31),target_type='raw_returns')

train_test_splits=create_tensors(study_periods)

class ScaledMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_dim):
        super(ScaledMultiHeadAttention, self).__init__()
        assert n_dim % num_heads == 0
        self.num_heads = num_heads
        self.n_dim = n_dim
        self.head_dim = n_dim // num_heads  # Dimension of each head: commonly referred to as d_k

        self.fc_q = nn.Linear(n_dim, n_dim)  # Query
        self.fc_k = nn.Linear(n_dim, n_dim)  # Key
        self.fc_v = nn.Linear(n_dim, n_dim)  # Value
        self.fc_o = nn.Linear(n_dim, n_dim)  # Output

    def create_look_ahead_mask(size):
        mask = 1 - torch.tril(torch.ones((size, size)))
        return mask  # Returns 0 for positions to be masked

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.fc_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.fc_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_out = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # Dot product of query and key to get attention scores

        if mask is not None:  # Look ahead mask to prevent information leakage
            key_out = key_out.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(key_out, dim=-1)  # Apply softmax to get probabilities
        value_out = torch.matmul(attention, V)  # Multiply attention scores by value
        value_out = value_out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_dim)  # Reshape to get back to original shape
        return self.fc_o(value_out)  # Apply final linear layer


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
    def __init__(self,d_model,num_heads,d_ff,dropout=.1):
        super(EncoderLayer,self).__init__()
        self.self_attention=ScaledMultiHeadAttention(num_heads,d_model)
        self.position_wise_feed_forward=PositionWiseFeedForward(d_model,d_ff,dropout)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm1=nn.LayerNorm(d_model)
        self.layer_norm2=nn.LayerNorm(d_model)
    def forward(self,x,mask):
        #Self attention
        attention=self.self_attention(x,x,x,mask)
        #Add and norm
        x=self.layer_norm1(x+self.dropout(attention))
        #Position wise feed forward
        feed_forward=self.position_wise_feed_forward(x)
        #Add and norm
        x=self.layer_norm2(x+self.dropout(feed_forward))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, dropout=0.1, max_len=512):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)
        ])

        # Final linear layer to denoise output
        self.fc = nn.Linear(d_model, 1)  

    def forward(self, src, src_mask=None):
        # Add positional encoding to the input
        src = self.positional_encoding(src)
        
        # Pass through each layer of the encoder
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # Capture the context from the last time step of the encoded sequence
        context = src[:, -1, :]
        
        # Final linear layer
        output = self.fc(context)
        
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = TimeSeriesTransformer(d_model=64, num_heads=8, d_ff=256, num_encoder_layers=2, 
                               dropout=.1, max_len=240).to(device)

# Loss depends on target, MAE for returns, Cross Entropy for above/below cross-sectional median. Also have selective loss in utils
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
patience = 5
best_loss = np.inf
counter = 0

for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0

    for train_data, train_labels, val_data, val_labels in tqdm(train_test_splits):

        # Generate look-ahead masks
        train_mask = ScaledMultiHeadAttention.create_look_ahead_mask(train_data.size(1))
        val_mask = ScaledMultiHeadAttention.create_look_ahead_mask(val_data.size(1))

        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        train_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data, src_mask=train_mask)  # Adjusted here to use the look-ahead mask
            loss = criterion(outputs.squeeze(), labels.float())  # Adjust based on your specific use case
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        total_train_loss += train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)

                outputs = model(data, src_mask=val_mask)  # Adjusted here to use the look-ahead mask
                loss = criterion(outputs.squeeze(), labels.float())  # Adjust based on your specific use case
                val_loss += loss.item() * data.size(0)

        total_val_loss += val_loss / len(val_loader.dataset)

    average_train_loss = total_train_loss / len(train_test_splits)
    average_val_loss = total_val_loss / len(train_test_splits)
    
    print(f'Epoch {epoch+1}/{n_epochs}, '
          f'Average Train Loss: {average_train_loss:.4f}, '
          f'Average Validation Loss: {average_val_loss:.4f}')

    if average_val_loss < best_loss:
        best_loss = average_val_loss
        counter = 0
    else:
        counter += 1

    if counter == patience:
        print('Early stopping!')
        break

model.load_state_dict(best_model_state)
model.eval()

in_sample_long_portfolios = pd.DataFrame()
out_of_sample_long_portfolios = pd.DataFrame()

in_sample_short_portfolios = pd.DataFrame()
out_of_sample_short_portfolios = pd.DataFrame()

k = 10  # Number of top assets to select in portfolios

for train_data, train_labels, val_data, val_labels in tqdm(train_test_splits):
    # Here, train_data, val_data are your training and validation data respectively
    
    train_mask = ScaledMultiHeadAttention.create_look_ahead_mask(train_data.size(1))
    val_mask = ScaledMultiHeadAttention.create_look_ahead_mask(val_data.size(1))

    with torch.no_grad():
        train_predictions = model(train_data.to(device), src_mask=train_mask.to(device))
        val_predictions = model(val_data.to(device), src_mask=val_mask.to(device))

        train_probs = torch.softmax(train_predictions, dim=1)[:, 1].cpu().numpy()
        val_probs = torch.softmax(val_predictions, dim=1)[:, 1].cpu().numpy()

    # Assuming you have a dataframe or similar structure to hold the date and TICKER information
    train_df['predicted_prob'] = train_probs
    val_df['predicted_prob'] = val_probs

    # In-Sample Portfolio Construction
    for date in train_df['date'].unique():
        date_data = train_df[train_df['date'] == date].sort_values(by='predicted_prob', ascending=False)
        
        long_tickers = date_data.head(k)
        short_tickers = date_data.tail(k)
        
        in_sample_long_portfolios = pd.concat([in_sample_long_portfolios, long_tickers])
        in_sample_short_portfolios = pd.concat([in_sample_short_portfolios, short_tickers])

    # Out-of-Sample Portfolio Construction
    for date in val_df['date'].unique():
        date_data = val_df[val_df['date'] == date].sort_values(by='predicted_prob', ascending=False)
        
        long_tickers = date_data.head(k)
        short_tickers = date_data.tail(k)
        
        out_of_sample_long_portfolios = pd.concat([out_of_sample_long_portfolios, long_tickers])
        out_of_sample_short_portfolios = pd.concat([out_of_sample_short_portfolios, short_tickers])

# At this point, in_sample_long_portfolios, out_of_sample_long_portfolios, etc. hold your portfolios

in_sample_long_portfolios.to_csv('../data/transformer_results/in_sample_long_portfolios.csv')
in_sample_short_portfolios.to_csv('../data/transformer_results/in_sample_short_portfolios.csv')
out_of_sample_long_portfolios.to_csv('../data/transformer_results/out_of_sample_long_portfolios.csv')
out_of_sample_short_portfolios.to_csv('../data/transformer_results/out_of_sample_short_portfolios.csv')