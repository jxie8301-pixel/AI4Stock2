"""LSTM model with AMP (Automatic Mixed Precision) and AdamW for maximum performance."""

import torch
import torch.nn as nn
import torch.optim as optim
from qlib.contrib.model.pytorch_lstm_ts import LSTM
from .utils.losses import PearsonLoss, CCCLoss

class AMPLSTM(LSTM):
    """Extended LSTM that supports AMP for faster training."""
    
    def train_epoch(self, data_loader):
        self.LSTM_model.train()
        scaler = torch.cuda.amp.GradScaler()

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            self.train_optimizer.zero_grad()
            
            # Use Mixed Precision
            with torch.cuda.amp.autocast():
                pred = self.LSTM_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))

            scaler.scale(loss).backward()
            scaler.step(self.train_optimizer)
            scaler.update()

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
    """Build an AMP-enabled LSTM model."""
    
    # We use our custom AMPLSTM instead of standard LSTM
    model = AMPLSTM(
        d_feat=d_feat,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_epochs=n_epochs,
        lr=lr,
        early_stop=early_stop,
        batch_size=batch_size,
        loss="mse",
        optimizer="adam",
        GPU=GPU,
        seed=seed,
        n_jobs=n_jobs,
    )
    
    # Upgrade to AdamW + Weight Decay
    weight_decay = 1e-4
    model.train_optimizer = optim.AdamW(model.LSTM_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Inject Custom Loss
    if loss.lower() == "pearson":
        model.loss_fn = PearsonLoss()
        model.loss = "pearson" 
    elif loss.lower() == "ccc":
        model.loss_fn = CCCLoss()
        model.loss = "ccc"
    
    print(f"LSTM model built (AMP+AdamW): d_feat={d_feat}, hidden={hidden_size}, "
          f"loss={loss}, n_jobs={n_jobs}, early_stop=10")
    return model
