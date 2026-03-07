"""Pure PyTorch LSTM Implementation independent of Qlib."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from src.models.utils.losses import PearsonLoss, CCCLoss

class NativeStockDataset(Dataset):
    """
    A PyTorch Dataset that efficiently slices 3D time-series windows from a 2D panel.
    Assumes data is pre-aligned and sorted by (symbol, date).
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 stock_indices: np.ndarray, lookback: int = 20):
        """
        Parameters
        ----------
        features : np.ndarray
            Shape (N, F). Standardized feature matrix.
        labels : np.ndarray
            Shape (N,). Target labels (e.g., T+1 open to T+2 open return).
        stock_indices : np.ndarray
            Shape (N,). Integer IDs representing the stock symbol for each row, 
            used to prevent slicing across different stocks.
        lookback : int
            Time-series window length.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.lookback = lookback
        
        # Pre-compute valid indices where a full lookback window exists for the SAME stock
        # This prevents taking the last 10 days of AAPL and first 10 days of MSFT
        valid_mask = (stock_indices[lookback - 1:] == stock_indices[:-lookback + 1])
        # Valid starting points in the array
        self.valid_starts = np.where(valid_mask)[0]

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.lookback
        
        # X: (lookback, F)
        x = self.features[start_idx:end_idx]
        # Y: scalar (the label at the END of the window)
        y = self.labels[end_idx - 1]
        
        return x, y


class PureLSTM(nn.Module):
    def __init__(self, d_feat: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: (Batch, Time, Features)
        out, _ = self.rnn(x)
        # Take the output of the last time step
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out).squeeze(-1)


class NativeLSTMTrainer:
    """Trainer class decoupled from Qlib."""
    def __init__(self, d_feat: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, lr: float = 0.0005, loss_type: str = "pearson",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = PureLSTM(d_feat, hidden_size, num_layers, dropout).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Loss function
        if loss_type == "pearson":
            self.loss_fn = PearsonLoss()
        elif loss_type == "ccc":
            self.loss_fn = CCCLoss()
        else:
            self.loss_fn = nn.MSELoss()
            
        self.scaler = torch.amp.GradScaler('cuda') if 'cuda' in self.device else None

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device, non_blocking=True)
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        pred = self.model(x)
                else:
                    pred = self.model(x)
                all_preds.append(pred.cpu())
                all_labels.append(y)
                
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        # Calculate full-set Pearson Correlation
        if len(all_preds) > 1:
            vx = all_preds - torch.mean(all_preds)
            vy = all_labels - torch.mean(all_labels)
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
            return corr.item()
        return 0.0

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, 
            epochs: int = 200, early_stop: int = 10):
        best_score = -np.inf
        stop_count = 0
        best_state = None
        
        for epoch in range(epochs):
            self.train_epoch(train_loader)
            valid_score = self.evaluate(valid_loader)
            print(f"Epoch {epoch} | Valid IC: {valid_score:.6f}")
            
            if valid_score > best_score:
                best_score = valid_score
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                stop_count = 0
            else:
                stop_count += 1
                if stop_count >= early_stop:
                    print("Early stopping triggered.")
                    break
                    
        if best_state:
            self.model.load_state_dict(best_state)
        return best_score
