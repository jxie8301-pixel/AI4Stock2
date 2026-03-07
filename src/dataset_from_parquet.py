"""从 collector_akshare.py 生成的 Parquet 文件直接训练 LSTM。

数据格式：
- data/processed/combined/000001.SZ.parquet
- data/processed/combined/000002.SZ.parquet
- ...

每个文件包含：date, open, high, low, close, volume, amount, turnover,
              total_mv, circ_mv, pe_ttm, pb, ps, pcf, peg, ...
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


class StockParquetDataset(Dataset):
    """从 collector_akshare.py 的 Parquet 文件构建时序数据集"""

    def __init__(
        self,
        parquet_dir: str = "data/processed/combined",
        feature_cols: Optional[List[str]] = None,
        label_type: str = "return",  # "return" or "price"
        lookback: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        parquet_dir : str
            Parquet 文件目录（每个股票一个文件）
        feature_cols : list, optional
            特征列。如果为 None，使用默认的量价+估值特征
        label_type : str
            标签类型：
            - "return": 未来收益率 (open_t+1 / open_t - 1)
            - "price": 未来价格 (open_t+1)
        lookback : int
            时间窗口长度
        start_date : str, optional
            开始日期（YYYY-MM-DD）
        end_date : str, optional
            结束日期（YYYY-MM-DD）
        """
        self.parquet_dir = Path(parquet_dir)
        self.lookback = lookback
        self.label_type = label_type

        # 默认特征：量价 + 估值
        if feature_cols is None:
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover',
                'total_mv', 'circ_mv', 'pe_ttm', 'pb', 'ps', 'pcf', 'peg'
            ]
        self.feature_cols = feature_cols

        # 加载所有股票数据
        self.samples = []
        self._load_all_stocks(start_date, end_date)

        print(f"Dataset: {len(self.samples)} samples, "
              f"{len(feature_cols)} features, lookback={lookback}")

    def _load_all_stocks(self, start_date, end_date):
        """加载所有股票的 Parquet 文件"""
        parquet_files = list(self.parquet_dir.glob("*.parquet"))

        for file in tqdm(parquet_files, desc="Loading stocks"):
            try:
                df = pd.read_parquet(file)

                # 确保有 date 列
                if 'date' not in df.columns:
                    continue

                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                # 时间过滤
                if start_date:
                    df = df[df['date'] >= start_date]
                if end_date:
                    df = df[df['date'] <= end_date]

                # 检查必需的列
                if not all(col in df.columns for col in self.feature_cols):
                    continue

                # 构建样本
                self._build_samples_from_stock(df, file.stem)

            except Exception as e:
                print(f"Error loading {file.name}: {e}")
                continue

    def _build_samples_from_stock(self, df: pd.DataFrame, symbol: str):
        """从单个股票构建滑动窗口样本"""
        # 计算标签
        if self.label_type == "return":
            # 未来收益率：(open_t+1 / open_t - 1)
            df['label'] = df['open'].shift(-1) / df['open'] - 1
        elif self.label_type == "price":
            # 未来价格
            df['label'] = df['open'].shift(-1)
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")

        # 滑动窗口
        for i in range(self.lookback, len(df) - 1):  # -1 因为需要未来标签
            # 特征：过去 lookback 天
            X = df[self.feature_cols].iloc[i-self.lookback:i].values

            # 标签：未来收益率
            y = df['label'].iloc[i]

            # 跳过缺失值
            if np.isnan(X).any() or np.isnan(y) or np.isinf(y):
                continue

            self.samples.append((X.astype(np.float32), np.float32(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.from_numpy(X), torch.tensor([y])


# ═══════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════

def create_datasets(
    parquet_dir: str = "data/processed/combined",
    train_period: tuple = ("2018-01-01", "2020-12-31"),
    valid_period: tuple = ("2021-01-01", "2021-12-31"),
    test_period: tuple = ("2022-01-01", "2023-12-31"),
    lookback: int = 20,
):
    """创建训练/验证/测试数据集"""

    train_dataset = StockParquetDataset(
        parquet_dir=parquet_dir,
        lookback=lookback,
        start_date=train_period[0],
        end_date=train_period[1],
    )

    valid_dataset = StockParquetDataset(
        parquet_dir=parquet_dir,
        lookback=lookback,
        start_date=valid_period[0],
        end_date=valid_period[1],
    )

    test_dataset = StockParquetDataset(
        parquet_dir=parquet_dir,
        lookback=lookback,
        start_date=test_period[0],
        end_date=test_period[1],
    )

    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    # 测试数据加载
    train_ds, valid_ds, test_ds = create_datasets()

    print(f"\nTrain: {len(train_ds)} samples")
    print(f"Valid: {len(valid_ds)} samples")
    print(f"Test: {len(test_ds)} samples")

    # 查看一个样本
    X, y = train_ds[0]
    print(f"\nSample shape: X={X.shape}, y={y.shape}")
    print(f"Label (return): {y.item():.4f}")
