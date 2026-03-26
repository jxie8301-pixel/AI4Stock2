# AI4Stock2 用户使用指南

## 1. 数据准备 (Data Preparation)

本项目使用 `akshare` 并结合 Cookie 劫持来稳定获取全市场 A 股（量价+估值）数据。你需要手动执行数据同步。

### 同步旧项目数据
建议直接将旧 `AI4Stock` 项目的 `data/raw` 和 `data/processed` 文件夹复制到本项目的 `data/` 目录下，实现无缝衔接。

### 增量更新
运行以下命令，自动补全缺失数据并刷新本地 Parquet 数据：
```bash
# 更新数据
python src/collector_akshare.py --update --workers 8
```

Native 训练前，建议先按当前配置生成本地特征缓存：
```bash
uv run python src/gen_feature.py --config configs/config.yaml --workers 8
```

如果要保留全量 Alpha158 作为对照组：
```bash
uv run python src/gen_feature.py --config configs/config_baseline.yaml --workers 8
```

## 2. 核心运行模式 (Core Workflow)

### 滚动训练与回测 (推荐实战模式)
当前推荐优先使用 native + LightGBM，先做稳健基线：
```bash
uv run python run_native_rolling.py --model lgbm --horizon 20 --run-tag compact_lgbm
```

同一模型做不同策略对比时，建议直接从命令行覆写策略参数，并加上一个 `run tag`：
```bash
uv run python run_native_rolling.py --model lgbm --horizon 20 --topk 20 --n-drop 4 --run-tag top20_drop4
uv run python run_native_rolling.py --model lgbm --horizon 20 --topk 30 --n-drop 5 --run-tag top30_drop5
```

### 单次实验 (研究模式)
用于快速验证想法：
```bash
uv run python main.py --model lgbm --save-model results/lgbm/model.pkl
```

也可以让系统自动把模型和实验元数据归档到本地实验库：
```bash
uv run python main.py --model lgbm --topk 25 --n-drop 5 --run-tag alpha25
```

## 3. 模型复用 (Save & Load)

利用之前保存的滚动专家模型，实现秒级快速回测：
```bash
# 加载 native rolling 模型库
uv run python run_native_rolling.py --model lgbm --horizon 20 --load-models
```

## 4. 本地实验库 (Local Experiment Store)

默认启用本地实验归档，根目录为 `results/experiments/`：
- 每次运行会生成一个独立目录，保存配置快照、指标清单、结果工件副本。
- 单次实验在未显式传入 `--save-model` 时，会自动保存模型到该目录。
- 全局对比索引保存在 `results/experiments/experiment_index.csv`，方便同模型不同策略横向比较。

如需关闭：
```bash
uv run python main.py --model lgbm --disable-local-store
```

## 5. 关键参数微调 (`configs/config.yaml`)

- **股票池** (`universe`): 强烈建议使用 `csi300_real`（纯净版沪深300）。
- **回看天数** (`lookback`): 建议设为 `20`（抓取短期时序特征）。
- **特征 Profile** (`features.profile`): 默认使用 `alpha158_compact_v1`，保留 `config_baseline.yaml` 的 `alpha158_full` 作为对照。具体定义保存在 `configs/features/*.yaml`。
- **模型 Preset** (`model.preset`): 主配置只引用模型预设，具体超参保存在 `configs/models/*.yaml`。
- **训练期选列** (`features.selected_columns`): 可以在不重建 cache 的前提下，只挑选全集中的部分因子参与训练。
- **训练历史**: 建议从 `2016-01-01` 开始，以适应当前的机构化行情。
- **交易约束**: 在 `src/backtest.py` 中已默认开启“涨跌停禁止交易”和“开盘价成交”。

示例：
```yaml
features:
  profile: alpha158_full
  selected_columns:
    - KMID
    - MA20
    - RSV20
```

改完 `selected_columns` 后，不需要重新执行 `gen_feature.py`；直接重新训练即可。

LightGBM 训练会自动输出特征重要性：
- 单次实验：`results/native/lgbm/feature_importance_gain.csv`
- 滚动实验：`results/native_rolling_lgbm/feature_importance_gain_mean.csv`

## 6. 结果分析 (Analysis)

当前推荐结果目录一般为 `results/native_rolling_lgbm/`：
- `native_monthly_heatmap.png`: 月度收益红绿矩阵图。
- `native_monthly_report.csv`: **数字化月度报表**，方便 AI 进一步分析。
- `native_cumulative_return.png`: 包含真实滑点与限制的收益曲线。
- `models/`: 存放各时间段的专家模型权重。
