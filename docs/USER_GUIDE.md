# AI4Stock2 用户使用指南

## 1. 数据准备 (Data Preparation)

本项目使用 `akshare` 并结合 Cookie 劫持来稳定获取全市场 A 股（量价+估值）数据。你需要手动执行数据同步。

### 首次导入（若有旧项目数据）
如果你从旧的 AI4Stock 项目迁移，请先将旧的 `data/raw` 和 `data/processed` 文件夹复制到本项目的 `data/` 目录下。

### 增量更新与转换
运行以下命令，它会自动下载缺失的最新数据，并将其转换为 Qlib 高效的二进制格式（存入 `data/qlib_data_cn`）：
```bash
python src/collector_akshare.py --update --convert --workers 8
```

## 2. 运行模型与策略 (Training & Evaluation)

默认配置下，系统将执行：因子计算 -> 数据集切分 -> 模型训练 -> 信号评价 -> 回测。

### 基础运行 (默认 LSTM)
```bash
python main.py
```

### 切换模型 (支持 lgbm / lstm / transformer)
```bash
python main.py --model lgbm
```

### 仅测试信号 (跳过耗时的回测引擎)
如果你只想查看 IC/ICIR 等模型预测指标：
```bash
python main.py --skip-backtest
```

## 3. 模型保存与快速复用 (Save & Load)

深度学习模型训练耗时较长。你可以将训练好的模型保存下来，后续修改策略参数（如换手率、手续费）时直接加载，实现秒级回测。

**保存模型：**
```bash
python main.py --model lstm --save-model results/lstm/best_model.pkl
```

**加载模型并极速回测（跳过训练）：**
```bash
python main.py --model lstm --load-model results/lstm/best_model.pkl
```

## 4. 结果分析 (Analysis)

所有运行结果将按模型分类存储在 `results/` 目录下：
- `results/<model>/ic_series.png`: 信号质量（IC）随时间的变化图。
- `results/<model>/cumulative_return.png`: 策略累计收益率曲线（对比基准）。
- `results/<model>/drawdown.png`: 策略回撤曲线。
- `results/<model>/signal_metrics.json`: IC, ICIR, Rank IC 等核心指标。

## 5. 参数微调

你可以直接修改 `configs/config.yaml` 来调整：
- **损失函数** (`loss`): 可选 `mse` / `pearson` / `ccc`。强烈建议使用 `pearson` 以优化排名。
- **回看天数** (`lookback`): 影响模型输入的历史序列长度（建议 20 天）。
- **训练/验证/测试集比例** (`time`): 控制模型的训练范围与回测区间。
- **策略参数** (`topk` 和 `n_drop`): 控制选股数量和换手率。
