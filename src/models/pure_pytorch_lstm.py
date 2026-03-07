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
    Preserves zero-copy memmap behavior by keeping the full array and mapping indices.
    """
    def __init__(self, full_features: np.ndarray, full_labels: np.ndarray, 
                 full_symbols: np.ndarray, mask: np.ndarray, lookback: int = 20):
        """
        Parameters
        ----------
        full_features : np.ndarray (Memmap)
            Shape (N, F). The complete, memory-mapped feature matrix.
        full_labels : np.ndarray (Memmap)
            Shape (N,). The complete target labels.
        full_symbols : np.ndarray
            Shape (N,). Integer IDs representing the stock symbol for each row.
        mask : np.ndarray (bool)
            Shape (N,). Boolean mask indicating which rows belong to this dataset split.
        lookback : int
            Time-series window length.
        """
        self.features = torch.from_numpy(full_features)
        self.labels = torch.from_numpy(full_labels)
        self.lookback = lookback
        
        # 1. Find all rows where the stock symbol has been continuous for 'lookback' days
        # A valid window ending at 'i' must have the same symbol at 'i' and 'i - lookback + 1'
        # Since the data is sorted by symbol -> date, this ensures the whole window is the same stock.
        continuous_mask = np.zeros_like(mask, dtype=bool)
        continuous_mask[lookback - 1:] = (full_symbols[lookback - 1:] == full_symbols[:-lookback + 1])
        
        # 2. Intersect with the user-provided mask (e.g. train/valid split mask)
        # We only want to yield windows whose TARGET (end of window) falls within the split mask.
        final_valid_mask = continuous_mask & mask
        
        # 3. Store the actual global integer indices of the END of each valid window
        self.valid_end_indices = np.where(final_valid_mask)[0]

    def __len__(self):
        return len(self.valid_end_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_end_indices[idx]
        # The window slice in the global array is [end_idx - lookback + 1 : end_idx + 1]
        start_idx = end_idx - self.lookback + 1
        
        # X: (lookback, F)
        x = self.features[start_idx : end_idx + 1]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp extreme values to prevent AMP (float16) overflow (max is 65504)
        # Financial features should generally be small after standardization.
        # Extreme unscaled values like 10^15 will destroy the network.
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Y: scalar (the label at the END of the window)
        y = self.labels[end_idx]
        
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

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mask out NaN labels
            mask = ~torch.isnan(y) & ~torch.isinf(y)
            if not mask.any():
                continue

            x_masked = x[mask]
            y_masked = y[mask]            
            
            pred = self.model(x_masked)
            loss = self.loss_fn(pred, y_masked)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
                
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                # Mask out NaN and Inf labels
                mask = ~torch.isnan(y) & ~torch.isinf(y)
                if not mask.any():
                    continue
                    
                x_masked = x[mask]
                y_masked = y[mask]
                
                pred = self.model(x_masked)
                    
                # Mask out any predictions that somehow became NaN
                valid_pred_mask = ~torch.isnan(pred)
                
                all_preds.append(pred[valid_pred_mask].cpu())
                all_labels.append(y_masked[valid_pred_mask].cpu())
                
        if not all_preds:
            return 0.0
            
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
