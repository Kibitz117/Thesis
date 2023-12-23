from transformer_model import EncoderLayer
class ConvolutionalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super(ConvolutionalFeatureExtractor, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
class TimeSeriesTransformerWithCNN(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, input_features=1, dropout=0.3, task_type='regression', num_classes=1):
        super(TimeSeriesTransformerWithCNN, self).__init__()

        self.d_model = d_model
        self.task_type = task_type

        # Convolutional Feature Extractor layer
        self.feature_extractor = ConvolutionalFeatureExtractor(in_channels=input_features, out_channels=d_model, kernel_size=3, dropout=dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        
        # Output layer
        if task_type == 'classification':
            self.fc = nn.Linear(d_model, num_classes)
        else:  # regression
            self.fc = nn.Linear(d_model, 1)

    def forward(self, src, src_mask=None):
        # Ensure src is a float tensor
        src = src.float()

        # Apply convolutional feature extractor
        src = src.permute(0, 2, 1)  # Reshape to [batch_size, input_features, sequence_length]
        src = self.feature_extractor(src)  # Apply CNN
        src = src.permute(0, 2, 1)  # Reshape back to [batch_size, sequence_length, d_model]

        # Scale input embeddings
        src *= math.sqrt(self.d_model)

        # Pass through each layer of the encoder
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Context from the last time step
        context = src[:, -1, :]
        
        # Final linear layer for output
        output = self.fc(context)

        return output