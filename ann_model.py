"""
ANN Model cho phân loại Melanoma
"""

import torch
import torch.nn as nn
import numpy as np


class MelanomaANN(nn.Module):
    """Mạng 2 lớp: Input -> Hidden (sigmoid) -> Output"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(torch.softmax(logits, dim=1), dim=1)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_weights_as_vector(self):
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights_from_vector(self, weight_vector):
        idx = 0
        for param in self.parameters():
            size = param.numel()
            param.data = torch.FloatTensor(
                weight_vector[idx:idx+size].reshape(param.shape)
            ).to(param.device)
            idx += size
    
    def get_weight_bounds(self):
        n = self.get_num_params()
        return -1.0 * np.ones(n), 1.0 * np.ones(n)


def calculate_mse(model, X, y, device='cpu'):
    """Tính MSE theo công thức paper"""
    model.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)
        
        X, y = X.to(device), y.to(device)
        
        probs = torch.softmax(model(X), dim=1)
        y_onehot = torch.zeros(y.size(0), 2).to(device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        
        mse = 0.5 * torch.mean((probs - y_onehot) ** 2).item()
    return mse


def calculate_accuracy(model, X, y, device='cpu'):
    model.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)
        
        X, y = X.to(device), y.to(device)
        preds = model.predict(X)
        return (preds == y).float().mean().item()


if __name__ == "__main__":
    model = MelanomaANN(input_dim=100, hidden_dim=64)
    print(f"Params: {model.get_num_params()}")
    
    x = torch.randn(32, 100)
    print(f"Output: {model(x).shape}")
