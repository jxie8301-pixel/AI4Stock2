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

如需构建或刷新常用股票池文件：
```bash
uv run python src/build_universes.py
```

当前脚本默认生成：
- `csi300`
- `csi500`
- `zz1000`

Native 训练前，建议先按当前配置生成本地特征缓存：
```bash
uv run python src/gen_feature.py --workers 8
```

现在默认就是先生成一个足够大的全集 cache，然后训练时按需选列：
```bash
uv run python src/gen_feature.py --workers 8
```

如果只是更新了部分 Parquet，希望尽量少重算特征，可以使用增量模式：
```bash
uv run python src/gen_feature.py --workers 8 --incremental
```

当前增量模式的语义是：
- 未变化的股票 Parquet 复用已有 shard，不重复计算因子
- 变化过的股票 Parquet 才重算因子
- 因子库存储在 `data/factor_store/full_factor_space/shards/` 下
- 会同步刷新根目录 `meta.json`

这意味着它减少的是“因子重算量”，并让训练时可以只读取需要的列。

默认输出目录为 `data/factor_store/full_factor_space`，其中包含：
- Alpha158 全量因子，保留原始列名
- LightGBM 净化因子，列名前缀为 `LGBM_`
- 统一时间窗口因子，列名前缀为 `TEMP_`

当前统一时间窗口因子默认按这些窗口展开：
- `1, 5, 10, 20, 30, 60, 120`

当前默认展开的时间算子包括：
- `ret`
- `ma_gap`
- `std`
- `rsv`
- `price_rank`
- `volume_ratio`
- `turnover_mean`
- `amihud`
- `high_gap`
- `low_gap`
- `corr_cv`

之后无论是单次训练还是 rolling，都可以只在训练阶段通过 `features.selected_columns` 挑选子集，不需要重复生成 cache。
`features.profile` 现在更适合理解为“默认选列模板”：
- `core_v1`: 默认核心策略因子集
- `all_factors_full`: 使用全集
- `alpha158_full`: 默认只用 Alpha158 子集
- `alpha158_compact_v1`: 默认只用紧凑版 Alpha158 子集
- `lgbm_purified_v1`: 默认只用 LightGBM 净化子集

`gen_feature.py` 现在只负责生成统一全集 cache，不再暴露 `alpha158/alpha360` 这类历史因子库名字。
这些名字只保留在训练侧 profile 中，用来表达“从全集中默认选择哪些列”。

## 2. 核心运行模式 (Core Workflow)

### 滚动训练与回测 (推荐实战模式)
当前推荐优先使用 native + LightGBM，先做稳健基线：
```bash
uv run python run_native_rolling.py --model lgbm --horizon 20 --run-tag core_v1
```

如果要直接从命令行切换因子 profile，不必改 `config.yaml`：
```bash
uv run python run_native_rolling.py --model lgbm --profile alpha158_full --horizon 20 --run-tag alpha158_full
uv run python run_native_rolling.py --model lgbm --profile lgbm_purified_v1 --horizon 20 --run-tag purified_v1
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

同样支持命令行覆写 profile：
```bash
uv run python main.py --model lgbm --profile alpha158_full --run-tag alpha158_full_single
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

- **股票池** (`universe`): 强烈建议使用 `csi300`（沪深300）。
- **回看天数** (`lookback`): 建议设为 `20`（抓取短期时序特征）。
- **特征 Profile** (`features.profile`): 默认使用 `core_v1`。这里的 profile 主要表示“从全集 factor store 中默认选择哪些列”，不再表示单独的 cache 家族。具体定义保存在 `configs/features/*.yaml`。
- **推荐候选**: `alpha158_compact_v1` 适合作为技术面基线，`lgbm_purified_v1` 适合作为 LightGBM 研究起点。
- **命令行覆写**: `main.py` 和 `run_native_rolling.py` 都支持 `--profile`，适合做 profile 对照实验。
- **模型 Preset** (`model.preset`): 主配置只引用模型预设，具体超参保存在 `configs/models/*.yaml`。
- **训练期选列** (`features.selected_columns`): 可以在不重建 cache 的前提下，只挑选全集中的部分因子参与训练。
- **训练历史**: 建议从 `2016-01-01` 开始，以适应当前的机构化行情。
- **交易约束**: 在 `src/backtest.py` 中已默认开启“涨跌停禁止交易”和“开盘价成交”。

示例：
```yaml
features:
  profile: core_v1
  selected_columns:
    - KMID
    - MA20
    - RSV20
    - LGBM_ret_20
    - TEMP_ret_20
```

改完 `selected_columns` 后，不需要重新执行 `gen_feature.py`；直接重新训练即可。
如果启用 `features.transforms.cross_sectional_rank: true`，同样不需要重建 cache；这是训练期动态变换。

为什么 `gen_feature.py` 仍然独立存在，而不是在主训练脚本里隐式生成：
- factor store 生成是一个重 I/O、重 CPU 的预处理步骤，耗时和训练完全不是一个量级。
- 训练入口保持“只消费已有 factor store”，复现性更强，也更容易比较不同模型、不同选列、不同策略。
- 同一个全量 factor store 可以被很多次训练复用，这正好符合“先生成最全，再按需挑选”的研究方式。

如果后续要进一步提效，推荐新增一个显式模式，例如 `main.py --build-cache-if-missing`，而不是让训练脚本默认偷偷重建 cache。

LightGBM 训练会自动输出特征重要性：
- 单次实验：`results/native/lgbm/feature_importance_gain.csv`
- 滚动实验：`results/native_rolling_lgbm/feature_importance_gain_mean.csv`

## 6. 结果分析 (Analysis)

当前推荐结果目录一般为 `results/native_rolling_lgbm/`：
- `native_monthly_heatmap.png`: 月度收益红绿矩阵图。
- `native_monthly_report.csv`: **数字化月度报表**，方便 AI 进一步分析。
- `native_cumulative_return.png`: 包含真实滑点与限制的收益曲线。
- `models/`: 存放各时间段的专家模型权重。
