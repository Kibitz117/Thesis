import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd

class StockAutoencoder(nn.Module):
    def __init__(self, input_size, embedding_dim=32):
        super(StockAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, embedding_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, input_size),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_autoencoder(self, features, num_epochs=100, batch_size=64, learning_rate=0.001):
        # Data normalization
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        # Create a DataLoader
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        dataset = TensorDataset(features_tensor, features_tensor)  # Using features as both input and target
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.train()  # Set the model to training mode
        for epoch in range(num_epochs):
            for data in dataloader:
                inputs, targets = data
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
            
        return scaler  # Return the scaler for inverse transformation later

    def encode(self, features, scaler):
        # Apply the same scaling as during training
        scaled_features = scaler.transform(features)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            embeddings = self.encoder(features_tensor)
        return embeddings.numpy()  # Convert embeddings to NumPy array for easier handling
