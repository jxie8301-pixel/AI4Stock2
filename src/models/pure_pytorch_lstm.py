"""Pure PyTorch LSTM Implementation independent of Qlib."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from src.models.utils.losses import PearsonLoss, CCCLoss


def _as_datetime64_ns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]", copy=False)
    return pd.to_datetime(arr).to_numpy(dtype="datetime64[ns]", copy=False)


def _pearson_corr(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or ys.size < 2:
        return float("nan")
    xs = xs.astype(np.float64, copy=False)
    ys = ys.astype(np.float64, copy=False)
    xs_centered = xs - xs.mean()
    ys_centered = ys - ys.mean()
    xs_ss = float(np.dot(xs_centered, xs_centered))
    ys_ss = float(np.dot(ys_centered, ys_centered))
    if not np.isfinite(xs_ss) or not np.isfinite(ys_ss) or xs_ss <= 0.0 or ys_ss <= 0.0:
        return float("nan")
    return float(np.dot(xs_centered, ys_centered) / np.sqrt(xs_ss * ys_ss))


def compute_daily_ic(predictions: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> float:
    """Compute mean daily cross-sectional IC for validation."""
    if len(predictions) == 0:
        return 0.0

    pred_arr = np.asarray(predictions, dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.float32)
    date_arr = _as_datetime64_ns(dates)
    if len(pred_arr) != len(label_arr) or len(pred_arr) != len(date_arr):
        raise ValueError("predictions, labels, and dates must have the same length")

    valid_mask = np.isfinite(pred_arr) & np.isfinite(label_arr) & ~np.isnat(date_arr)
    if not valid_mask.any():
        return 0.0

    pred_arr = pred_arr[valid_mask]
    label_arr = label_arr[valid_mask]
    date_arr = date_arr[valid_mask]
    order = np.argsort(date_arr, kind="stable")
    pred_sorted = pred_arr[order]
    label_sorted = label_arr[order]
    date_sorted = date_arr[order]

    boundaries = np.flatnonzero(date_sorted[1:] != date_sorted[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(date_sorted)]

    values: list[float] = []
    for start, end in zip(starts, ends, strict=False):
        corr = _pearson_corr(pred_sorted[start:end], label_sorted[start:end])
        if np.isfinite(corr):
            values.append(float(corr))
    if not values:
        return 0.0
    return float(np.mean(values))


class NativeStockDataset(Dataset):
    """
    A PyTorch Dataset that efficiently slices 3D time-series windows from a 2D panel.
    Preserves zero-copy memmap behavior by keeping the full array and mapping indices.
    """
    def __init__(
        self,
        full_features: np.ndarray,
        full_labels: np.ndarray,
        full_symbols: np.ndarray,
        mask: np.ndarray,
        lookback: int = 20,
        full_dates: np.ndarray | None = None,
        feature_indices: np.ndarray | None = None,
        continuous_mask: np.ndarray | None = None,
        sanitize_features: bool = True,
    ):
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
        self.full_dates = np.asarray(full_dates) if full_dates is not None else None
        self.feature_indices = None if feature_indices is None else torch.as_tensor(feature_indices, dtype=torch.long)
        self.sanitize_features = bool(sanitize_features)
        
        # 1. Find all rows where the stock symbol has been continuous for 'lookback' days
        if continuous_mask is None:
            continuous_mask = np.zeros_like(mask, dtype=bool)
            if lookback <= 1:
                continuous_mask[:] = True
            else:
                continuous_mask[lookback - 1:] = (full_symbols[lookback - 1:] == full_symbols[:-lookback + 1])
        else:
            continuous_mask = np.asarray(continuous_mask, dtype=bool)
            if len(continuous_mask) != len(mask):
                raise ValueError("continuous_mask must have the same length as mask")
        
        # 2. Intersect with the user-provided mask (e.g. train/valid split mask)
        final_valid_mask = continuous_mask & mask
        
        # 3. Store the actual global integer indices of the END of each valid window
        self.valid_end_indices = np.where(final_valid_mask)[0]

    def get_dates_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if self.full_dates is None:
            raise ValueError("Dataset was created without full_dates.")
        return self.full_dates[indices]

    def __len__(self):
        return len(self.valid_end_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_end_indices[idx]
        start_idx = end_idx - self.lookback + 1
        
        # X: (lookback, F)
        x = self.features[start_idx : end_idx + 1]
        if self.feature_indices is not None:
            x = x[:, self.feature_indices]
        if self.sanitize_features:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # Clamp extreme features (Winsorization) to stabilize training
            x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Y: scalar (the label at the END of the window)
        y = self.labels[end_idx]

        return x, y


class PureLSTM(nn.Module):
    def __init__(self, d_feat: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # LayerNorm is crucial for raw, un-normalized technical features
        self.norm = nn.LayerNorm(d_feat)
        
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
        x = self.norm(x)
        out, _ = self.rnn(x)
        # Take the output of the last time step
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out).squeeze(-1)


class NativeLSTMTrainer:
    """Trainer class decoupled from Qlib, optimized for stability."""
    def __init__(self, d_feat: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.0, lr: float = 0.001, loss_type: str = "pearson",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = PureLSTM(d_feat, hidden_size, num_layers, dropout).to(self.device)
        
        # Optimizer - matching Qlib baseline (Adam, not AdamW)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Loss function
        if loss_type == "pearson":
            self.loss_fn = PearsonLoss()
        elif loss_type == "ccc":
            self.loss_fn = CCCLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader):
        if len(dataloader) == 0:
            return 0.0
        self.model.train()
        total_loss = 0.0
        steps = 0
        for x, y in dataloader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mask out NaN and Inf labels
            mask = ~torch.isnan(y) & ~torch.isinf(y)
            if not mask.any():
                continue

            x_masked = x[mask]
            y_masked = y[mask]            
            
            pred = self.model(x_masked)
            loss = self.loss_fn(pred, y_masked)
            loss.backward()

            # Match Qlib's gradient value clipping (3.0 is a safe threshold)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.optimizer.step()
                
            total_loss += loss.item()
            steps += 1
        return total_loss / steps if steps > 0 else 0.0

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_dates = []
        dataset = dataloader.dataset
        offset = 0
        with torch.no_grad():
            for x, y in dataloader:
                batch_size = len(y)
                batch_end_indices = dataset.valid_end_indices[offset : offset + batch_size]
                offset += batch_size

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
                valid_indices = batch_end_indices[mask.cpu().numpy()]
                valid_dates = dataset.get_dates_for_indices(valid_indices[valid_pred_mask.cpu().numpy()])
                all_dates.append(valid_dates)
                
        if not all_preds:
            return 0.0
            
        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels).numpy()
        all_dates_np = np.concatenate(all_dates)
        return compute_daily_ic(all_preds_np, all_labels_np, all_dates_np)

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, 
            epochs: int = 200, early_stop: int = 10):
        if len(train_loader) == 0:
            raise ValueError("Training loader is empty. Reduce batch_size or widen the training window.")
        if len(valid_loader) == 0:
            raise ValueError("Validation loader is empty. Reduce batch_size or widen the validation window.")

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
