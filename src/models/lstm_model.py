"""Stable LSTM model for AI4Stock2. Reverted extreme optimizations for stability."""

import torch
import torch.nn as nn
import torch.optim as optim
from qlib.contrib.model.pytorch_lstm_ts import LSTM
from .utils.losses import PearsonLoss, CCCLoss

def build_lstm_model(
    d_feat: int = 158,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    n_epochs: int = 200,
    lr: float = 0.001,
    early_stop: int = 20,
    batch_size: int = 2048,
    loss: str = "mse",
    optimizer: str = "adam",
    GPU: int = 0,
    seed: int = 42,
    n_jobs: int = 10,
) -> LSTM:
    """Build a stable LSTM model with AdamW and custom loss support."""
    
    model = LSTM(
        d_feat=d_feat,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_epochs=n_epochs,
        lr=lr,
        early_stop=early_stop,
        batch_size=batch_size,
        loss="mse", # Satisfy base class init
        optimizer="adam",
        GPU=GPU,
        seed=seed,
        n_jobs=n_jobs,
    )
    
    # Use AdamW + Weight Decay for better generalization (This part was stable)
    weight_decay = 1e-4
    model.train_optimizer = optim.AdamW(model.LSTM_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Inject Custom Loss
    if loss.lower() == "pearson":
        model.loss_fn = PearsonLoss()
        model.loss = "pearson" 
    elif loss.lower() == "ccc":
        model.loss_fn = CCCLoss()
        model.loss = "ccc"
    
    print(f"LSTM model built (Stable): d_feat={d_feat}, batch={batch_size}, loss={loss}, n_jobs={n_jobs}")
    return model
