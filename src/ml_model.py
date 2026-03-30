import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from processing import CSIProcessor
from sklearn.preprocessing import LabelEncoder
import os

class CSIDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CSINet(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(CSINet, self).__init__()
        # Input shape will be window features
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        
        # Adding an LSTM layer conceptually
        # Assuming input can be reshaped to sequence later, but for now we take flattened vector
        # Let's say we just use dense for the flattened PCA+STFT features
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        # Reshape for LSTM (batch, seq, features) -> treat as seq length 1
        x = x.unsqueeze(1)
        x, (hn, cn) = self.lstm(x)
        
        x = x[:, -1, :] # Take last output
        x = self.fc3(x)
        return x

def train_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    data_path = os.path.join(models_dir, 'synthetic_csi.csv')
    
    print("Loading synthetic data...")
    if not os.path.exists(data_path):
        print(f"{data_path} not found. Run generate_synthetic_csi.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    processor = CSIProcessor(window_size=200)
    
    features_list = []
    labels_list = []
    
    # Process windows
    window_pts = 200
    stride = 100
    
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    print("Extracting features from windows...")
    num_subcarriers = 52
    
    for i in range(0, len(df) - window_pts, stride):
        window = df.iloc[i:i+window_pts]
        label = window['label_encoded'].mode()[0]
        
        I_cols = [f'I_{sc}' for sc in range(num_subcarriers)]
        Q_cols = [f'Q_{sc}' for sc in range(num_subcarriers)]
        
        I_data = window[I_cols].values
        Q_data = window[Q_cols].values
        
        features, _, _, _ = processor.process_window(I_data, Q_data)
        
        features_list.append(features)
        labels_list.append(label)
        
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Save label classes to use later
    np.save(os.path.join(models_dir, 'classes.npy'), label_encoder.classes_)
    
    dataset = CSIDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = CSINet(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model...")
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")
        
    model_save_path = os.path.join(models_dir, 'csi_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return X.shape[1] # Return input dim

if __name__ == '__main__':
    train_model()