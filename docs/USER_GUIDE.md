# AI4Stock2 用户使用指南

## 1. 快速开始 (Quick Start)

### 第一步：准备数据
数据下载只需执行一次。该脚本会自动从 Qlib 官方仓库获取下载工具并同步 A 股 CSI300 数据。
```bash
python main.py --download-only
```

### 第二步：运行实验 (默认 LSTM)
默认配置下，系统将执行：因子计算 -> 数据集切分 -> 模型训练 -> 信号评价 -> 回测。
```bash
python main.py
```

## 2. 常用运行模式

### 切换模型 (Transformer)
```bash
python main.py --model transformer
```

### 仅测试模型信号 (跳过回测)
如果你只想查看 IC/ICIR 等模型预测指标，不希望运行缓慢的回测引擎：
```bash
python main.py --skip-backtest
```

### 指定计算设备
- **GPU 训练** (默认使用 GPU 0):
  ```bash
  python main.py --gpu 0
  ```
- **强制使用 CPU**:
  ```bash
  python main.py --gpu -1
  ```

## 3. 结果分析 (Analysis)

所有运行结果将按模型分类存储在 `results/` 目录下：
- `results/lstm/ic_series.png`: 信号质量（IC）随时间的变化图。
- `results/lstm/cumulative_return.png`: 策略累计收益率曲线（对比基准）。
- `results/lstm/drawdown.png`: 策略回撤曲线。
- `results/lstm/signal_metrics.json`: IC, ICIR, Rank IC 等核心指标。

## 4. 参数微调

你可以直接修改 `configs/config.yaml` 来调整：
- **回看天数** (`lookback`): 影响模型输入的历史序列长度。
- **训练/验证/测试集比例** (`time`): 默认以 2021 年后为测试期。
- **策略参数** (`topk` 和 `n_drop`): 控制选股数量和换手率。
