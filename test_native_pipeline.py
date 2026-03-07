"""Minimal test script for the pure PyTorch native pipeline."""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.models.pure_pytorch_lstm import NativeStockDataset, NativeLSTMTrainer

def load_memmap_data(cache_dir: str):
    """Load pre-generated numpy arrays using memmap for zero-copy loading."""
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cache metadata not found at {meta_path}. Run gen_feature.py first.")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    shape = tuple(meta["shape"])
    num_rows = meta["num_rows"]
    
    print(f"[*] Loading data from {cache_dir}")
    print(f"[*] Expected shape: {shape}, Features: {meta['num_features']}")
    
    # Read memmaps
    X = np.lib.format.open_memmap(os.path.join(cache_dir, "X.npy"), mode="r", dtype=np.float32, shape=shape)
    y = np.lib.format.open_memmap(os.path.join(cache_dir, "y.npy"), mode="r", dtype=np.float32, shape=(num_rows,))
    dates = np.lib.format.open_memmap(os.path.join(cache_dir, "date.npy"), mode="r", dtype=np.int64, shape=(num_rows,))
    symbols = np.lib.format.open_memmap(os.path.join(cache_dir, "symbol.npy"), mode="r", dtype=np.int32, shape=(num_rows,))
    
    return X, y, dates, symbols, meta

def run_native_test():
    """Run a minimal training loop using the native dataset."""
    cache_dir = "data/cache/alpha158_panel"
    lookback = 20
    batch_size = 4096
    
    # 1. Load Data
    try:
        X, y, dates, symbols, meta = load_memmap_data(cache_dir)
    except FileNotFoundError as e:
        print(f"[!] {e}")
        print("[!] Please generate features first by running: python src/gen_feature.py")
        return

    # Convert nano-second timestamps to pandas datetime for easy filtering
    print("[*] Parsing dates for time-split...")
    dt_index = pd.to_datetime(dates)
    
    # 2. Time-based Split (Vectorized, much faster than Qlib)
    # Train: < 2023, Valid: 2023
    train_mask = (dt_index >= pd.Timestamp("2016-01-01")) & (dt_index < pd.Timestamp("2023-01-01"))
    valid_mask = (dt_index >= pd.Timestamp("2023-01-01")) & (dt_index < pd.Timestamp("2024-01-01"))
    
    print(f"[*] Train samples: {train_mask.sum():,}")
    print(f"[*] Valid samples: {valid_mask.sum():,}")
    
    # We must pass the FULL arrays to the dataset, but we'll create subset arrays for the loaders
    # Wait, because of lookback, we can't just slice train_mask directly without breaking the sequence.
    # The Dataset handles the lookback, so we can pass the sliced contiguous chunks.
    # Actually, slicing the memmap directly keeps it contiguous per symbol if the original was sorted by symbol->date.
    # Let's get the indices of train and valid data
    
    # Extracting the actual data chunks in memory. 
    # For a test, we will pull them into RAM to avoid memmap disk IO bottlenecks during training.
    # If RAM is constrained, we can keep them as memmaps.
    print("[*] Slicing Train Data...")
    X_train = np.array(X[train_mask])
    y_train = np.array(y[train_mask])
    sym_train = np.array(symbols[train_mask])
    
    print("[*] Slicing Valid Data...")
    X_valid = np.array(X[valid_mask])
    y_valid = np.array(y[valid_mask])
    sym_valid = np.array(symbols[valid_mask])
    
    # 3. Create PyTorch Datasets & DataLoaders
    print("[*] Initializing Native PyTorch Datasets...")
    train_dataset = NativeStockDataset(X_train, y_train, sym_train, lookback=lookback)
    valid_dataset = NativeStockDataset(X_valid, y_valid, sym_valid, lookback=lookback)
    
    print(f"[*] Effective Train Windows: {len(train_dataset):,}")
    print(f"[*] Effective Valid Windows: {len(valid_dataset):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True)
    
    # 4. Initialize Trainer
    print("[*] Initializing Native LSTM Trainer...")
    trainer = NativeLSTMTrainer(
        d_feat=meta["num_features"],
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        lr=0.0005,
        loss_type="pearson",
    )
    
    # 5. Run a quick 5-epoch test
    print("\n" + "="*50)
    print("STARTING NATIVE TRAINING TEST (5 Epochs)")
    print("="*50)
    trainer.fit(train_loader, valid_loader, epochs=5, early_stop=5)
    print("="*50)
    print("NATIVE TEST COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_native_test()
