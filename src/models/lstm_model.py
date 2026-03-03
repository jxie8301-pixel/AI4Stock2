"""LSTM model for stock prediction (Phase 1).

Uses Qlib's built-in LSTM implementation designed for TSDatasetH.
The model reads (batch, step_len, d_feat) tensors and predicts a score per stock.
"""

import torch.nn as nn
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
) -> LSTM:
    """Build an LSTM model for time-series stock prediction.

    Parameters
    ----------
    d_feat : int
        Number of input features (158 for Alpha158).
    hidden_size : int
        LSTM hidden state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate between LSTM layers.
    n_epochs : int
        Maximum training epochs.
    lr : float
        Learning rate.
    early_stop : int
        Stop training if validation metric doesn't improve for this many epochs.
    batch_size : int
        Training batch size.
    loss : str
        Loss function. Can be 'mse', 'pearson', or 'ccc'.
    GPU : int
        GPU device id. Set to -1 for CPU.
    """
    
    # Map custom losses
    # Note: Qlib's base PyTorch model expects loss to be a string like "mse"
    # or a custom callable. However, Qlib's internal metric_fn might struggle 
    # if it doesn't recognize the string. We will override it cleanly.
    
    model = LSTM(
        d_feat=d_feat,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_epochs=n_epochs,
        lr=lr,
        early_stop=early_stop,
        batch_size=batch_size,
        loss="mse", # Base Qlib expects a recognized string during init
        optimizer=optimizer,
        GPU=GPU,
        seed=seed,
    )
    
    # Inject our custom loss function dynamically
    if loss.lower() == "pearson":
        model.loss_fn = PearsonLoss()
        model.loss = "pearson" # For logging
    elif loss.lower() == "ccc":
        model.loss_fn = CCCLoss()
        model.loss = "ccc"
    
    print(f"LSTM model built: d_feat={d_feat}, hidden={hidden_size}, "
          f"layers={num_layers}, dropout={dropout}, loss={loss}")
    return model
