"""Dataset construction for time-series deep learning models."""

from qlib.data.dataset import TSDatasetH, DatasetH


def build_ts_dataset(handler, segments: dict, step_len: int = 60) -> TSDatasetH:
    """Build a TSDatasetH for time-series models (LSTM, Transformer, etc.).

    TSDatasetH organizes data into (step_len, n_features) windows per sample,
    suitable for sequential models.

    Parameters
    ----------
    handler : DataHandler
        Alpha158 or other data handler instance.
    segments : dict
        Time segments, e.g. {"train": ("2008-01-01", "2018-12-31"), ...}.
    step_len : int
        Number of lookback trading days per sample.
    """
    dataset = TSDatasetH(
        handler=handler,
        segments=segments,
        step_len=step_len,
    )
    print(f"TSDatasetH built: step_len={step_len}, segments={list(segments.keys())}")
    return dataset


def build_tabular_dataset(handler, segments: dict) -> DatasetH:
    """Build a DatasetH for tabular models (LightGBM, XGBoost, etc.).

    Each sample is a flat feature vector for one stock on one day.
    Useful as a baseline comparison.
    """
    dataset = DatasetH(
        handler=handler,
        segments=segments,
    )
    print(f"DatasetH built: segments={list(segments.keys())}")
    return dataset
