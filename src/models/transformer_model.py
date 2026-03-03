"""Transformer model for stock prediction (Phase 2).

Uses Qlib's built-in Transformer implementation designed for TSDatasetH.
Self-attention captures long-range temporal dependencies without sequential processing.
"""

import torch.nn as nn
from qlib.contrib.model.pytorch_transformer_ts import Transformer
from .utils.losses import PearsonLoss, CCCLoss

def build_transformer_model(
    d_feat: int = 158,
    hidden_size: int = 64,
    num_layers: int = 2,
    n_head: int = 4,
    dropout: float = 0.1,
    n_epochs: int = 200,
    lr: float = 0.0001,
    early_stop: int = 20,
    batch_size: int = 2048,
    loss: str = "mse",
    optimizer: str = "adam",
    GPU: int = 0,
    seed: int = 42,
) -> Transformer:
    """Build a Transformer model for time-series stock prediction.

    Parameters
    ----------
    d_feat : int
        Number of input features.
    hidden_size : int
        Model dimension (d_model).
    num_layers : int
        Number of TransformerEncoder layers.
    n_head : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate (typically smaller than LSTM, e.g. 1e-4).
    """
    model = Transformer(
        d_feat=d_feat,
        d_model=hidden_size,
        nhead=n_head,
        num_layers=num_layers,
        dropout=dropout,
        n_epochs=n_epochs,
        lr=lr,
        early_stop=early_stop,
        batch_size=batch_size,
        loss="mse", # Init with base string to satisfy base class
        optimizer=optimizer,
        GPU=GPU,
        seed=seed,
    )
    
    # Inject custom loss function dynamically
    if loss.lower() == "pearson":
        model.loss_fn = PearsonLoss()
        model.loss = "pearson"
    elif loss.lower() == "ccc":
        model.loss_fn = CCCLoss()
        model.loss = "ccc"
        
    print(f"Transformer model built: d_feat={d_feat}, d_model={hidden_size}, "
          f"heads={n_head}, layers={num_layers}, loss={loss}")
    return model
