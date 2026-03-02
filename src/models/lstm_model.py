"""LSTM model for stock prediction (Phase 1).

Uses Qlib's built-in LSTM implementation designed for TSDatasetH.
The model reads (batch, step_len, d_feat) tensors and predicts a score per stock.
"""

from qlib.contrib.model.pytorch_lstm_ts import LSTM


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
    GPU : int
        GPU device id. Set to -1 for CPU.
    """
    model = LSTM(
        d_feat=d_feat,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_epochs=n_epochs,
        lr=lr,
        early_stop=early_stop,
        batch_size=batch_size,
        loss=loss,
        optimizer=optimizer,
        GPU=GPU,
        seed=seed,
    )
    print(f"LSTM model built: d_feat={d_feat}, hidden={hidden_size}, "
          f"layers={num_layers}, dropout={dropout}")
    return model
