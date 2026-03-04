# AI4Stock2 用户使用指南

## 1. 数据准备 (Data Preparation)

本项目使用 `akshare` 并结合 Cookie 劫持来稳定获取全市场 A 股（量价+估值）数据。你需要手动执行数据同步。

### 同步旧项目数据
建议直接将旧 `AI4Stock` 项目的 `data/raw` 和 `data/processed` 文件夹复制到本项目的 `data/` 目录下，实现无缝衔接。

### 增量更新与转换
运行以下命令，自动补全缺失数据并生成 Qlib 二进制格式：
```bash
# 更新数据并转换
python src/collector_akshare.py --update --convert --workers 8
```

## 2. 核心运行模式 (Core Workflow)

### 滚动训练与回测 (推荐实战模式)
这是目前最强大的模式，每 6 个月重训一次模型以应对风格漂移：
```bash
python run_rolling.py --model lstm --horizon 120 --save-models --gpu 0
```

### 单次实验 (研究模式)
用于快速验证想法：
```bash
python main.py --model lstm --save-model results/lstm/model.pkl
```

## 3. 模型复用 (Save & Load)

利用之前保存的滚动专家模型，实现秒级快速回测：
```bash
# 加载滚动模型库
python run_rolling.py --model lstm --horizon 120 --load-models
```

## 4. 关键参数微调 (`configs/config.yaml`)

- **股票池** (`universe`): 强烈建议使用 `csi300_real`（纯净版沪深300）。
- **回看天数** (`lookback`): 建议设为 `20`（抓取短期时序特征）。
- **训练历史**: 建议从 `2016-01-01` 开始，以适应当前的机构化行情。
- **交易约束**: 在 `src/backtest.py` 中已默认开启“涨跌停禁止交易”和“开盘价成交”。

## 5. 结果分析 (Analysis)

所有运行结果保存在 `results/rolling_lstm/` 目录下：
- `monthly_heatmap.png`: 月度收益红绿矩阵图。
- `monthly_report.csv`: **数字化月度报表**，方便 AI 进一步分析。
- `cumulative_return.png`: 包含真实滑点与限制的收益曲线。
- `models/`: 存放各时间段的专家模型权重。
