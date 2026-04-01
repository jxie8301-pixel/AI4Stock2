# AI4Stock2 用户使用指南

## 1. 数据准备 (Data Preparation)

本项目当前同时保留三条数据源路径：

- `akshare` / 东财：历史主路径，当前仍可用于旧数据兼容
- `GM`：独立 raw 保留 + 二次规范化路径
- `Tushare`：新的主替代候选，当前已经可以独立抓取和增量更新

你需要手动执行数据同步。

当前支持两种互斥的网络后端：
- `cookie`: 默认方案，使用本地 `data/cookies.json` + `curl_cffi`
- `proxy_patch`: 使用 `akshare-proxy-patch` 的代理鉴权方案

两种方案不要混用；每次运行只选其一。

如果你要试验 GM 数据源，当前建议走“全量 raw 保留 + 二次规范化”的独立目录，不要和东财数据混写。GM 路径当前使用：
- raw: `data/gm/raw/...`
- normalized parquet: `data/gm/processed/combined/...`

如果你要试验或逐步切换到 Tushare，当前同样建议走独立目录，不要和东财 / GM 混写。Tushare 路径当前使用：
- raw: `data/tushare/raw/...`
- normalized parquet: `data/tushare/processed/combined/...`

### 同步旧项目数据
建议直接将旧 `AI4Stock` 项目的 `data/raw` 和 `data/processed` 文件夹复制到本项目的 `data/` 目录下，实现无缝衔接。

### 增量更新
运行以下命令，自动补全缺失数据并刷新本地 Parquet 数据：
```bash
# 更新数据
uv run python src/collector_akshare.py --update --workers 8
```

如果你要切到 `proxy_patch` 后端，需要显式传入：
```bash
uv run python src/collector_akshare.py --update --network-backend proxy_patch --proxy-auth-token <TOKEN> --workers 8
```

当前 collector 直接内置 README 示例里的固定网关 `101.201.173.125`，不再暴露 `--proxy-auth-ip` 这个选项。

当前 proxy 模式只 hook 我们实际用到的东财域名：
- `push2.eastmoney.com`
- `push2his.eastmoney.com`
- `82.push2.eastmoney.com`
- `datacenter-web.eastmoney.com`

`--update` 现在默认不会联网刷新股票代码列表。
它会只用：
- 本地已有 raw/processed symbol
- 本地缓存过的股票列表

做并集，也就是 `local + cache`。

如果你希望在增量更新时顺手把新上市股票也带进来，再显式加：
```bash
uv run python src/collector_akshare.py --update --refresh-stock-list --workers 8
```

这时才会联网刷新一份 live 股票列表，并用：
- 本地已有 raw/processed symbol
- 本地缓存过的股票列表
- 当前 live 股票列表

做并集，因此不会漏掉新上市股票，也不会因为退市/停牌把本地已有 symbol 丢掉。

`--all` 现在也优先使用本地股票列表缓存。
只有两种情况才会重新触发股票列表分页刷新：
- 本地 `data/raw/meta/stock_list.parquet` 不存在
- 你显式传入 `--refresh-stock-list`

股票列表刷新现在是按页落盘并支持续跑的，页缓存位于：
- `data/raw/meta/stock_list_pages/page_0001.parquet`
- `data/raw/meta/stock_list_manifest.json`

如果单个 cookie 无法跑完整个股票列表分页，可以先只刷新或续跑股票列表缓存：
```bash
uv run python src/collector_akshare.py --refresh-stock-list-only
```

切换 cookie 后重复执行这条命令即可从缺失页继续，不会丢掉已经抓到的页。

如果你刚切到新的 collector schema，并且已经把旧数据完整备份到别处，建议先清空以下目录再重抓：
- `data/raw/daily`
- `data/raw/valuation`
- `data/processed/combined`
- `data/factor_store`

这样新的 parquet 从第一天起就只使用新脚本自己的格式，不再混用旧 schema。

如果只是修复融合逻辑、列名或本地 processed 文件，而不想重新联网抓取，可以只用本地 raw 重建：
```bash
uv run python src/collector_akshare.py --rebuild-processed --workers 8
```

如需构建或刷新常用股票池文件：
```bash
uv run python src/build_universes.py
```

### GM 数据采集
GM 路径会先保留各个 endpoint 的完整原始字段，再输出一份给 native pipeline 使用的规范化 parquet。

运行前先在 shell 里注入 token：
```bash
export GM_TOKEN=<YOUR_TOKEN>
```

全量或缓存股票池更新：
```bash
uv run python src/collector_gm.py --all --workers 8 --end-date 2026-03-31
```

只刷新 GM 股票列表缓存：
```bash
uv run python src/collector_gm.py --refresh-symbols-only
```

只用本地 raw 重建 GM 规范化 parquet：
```bash
uv run python src/collector_gm.py --rebuild-processed --workers 8
```

GM raw 目录当前拆分为：
- `data/gm/raw/bars_raw/`
- `data/gm/raw/symbol_day/`
- `data/gm/raw/daily_basic/`
- `data/gm/raw/daily_mktvalue/`
- `data/gm/raw/daily_valuation/`

当前默认 GM 路径只依赖你已验证可用的免费接口。
`stk_get_adj_factor` 属于付费增值接口，因此当前 collector 不再把它作为必需步骤；免费 `get_history_symbol` 返回的 `adj_factor` 会保留在 `symbol_day` 和规范化 parquet 里。

如果你要基于 GM 的规范化 parquet 生成因子库，直接显式指定输入目录：
```bash
uv run python src/gen_feature.py --parquet-dir data/gm/processed/combined --output-dir data/factor_store/gm_full_factor_space --workers 8
```

### Tushare 数据采集

Tushare 路径当前已经支持：
- 股票列表缓存
- 交易日历缓存
- `daily`
- `daily_basic`
- `adj_factor`
- `stk_limit`
- 基于这些表输出一份 `hfq` 规范化 combined parquet

当前这条路径已经可以独立跑通，但还没有正式切换为默认研究数据源。
原因不是 collector 不可用，而是正式接入前还要完成：
- canonical schema 定稿
- 训练 / rolling 入口的正式切换
- 财务 / 事件表的系统接入

如果 shell 里已经保存过 Tushare token，通常不需要重复显式 `export`。
如需显式指定，也可以在运行前注入：
```bash
export TUSHARE_TOKEN=<YOUR_TOKEN>
```

刷新 Tushare 股票列表缓存：
```bash
uv run python src/collector_tushare.py --refresh-symbols-only
```

抓取全量或缓存股票池：
```bash
uv run python src/collector_tushare.py --all --workers 8 --end-date 2026-03-31
```

按本地已有 symbol + 缓存股票池做增量更新：
```bash
uv run python src/collector_tushare.py --update --workers 8 --end-date 2026-03-31
```

只跑少量 symbol 做验证：
```bash
uv run python src/collector_tushare.py --symbols 600000,000333 --workers 2 --end-date 2026-03-31
```

如果 raw 已经完整，只想从本地 raw 重建 `hfq` 规范化 parquet：
```bash
uv run python src/collector_tushare.py --rebuild-processed --workers 8
```

Tushare raw 目录当前拆分为：
- `data/tushare/raw/meta/`
- `data/tushare/raw/daily/`
- `data/tushare/raw/daily_basic/`
- `data/tushare/raw/adj_factor/`
- `data/tushare/raw/stk_limit/`

当前 Tushare collector 的几个关键行为：
- 以 `stock_basic(list_status=L/D)` 维护股票缓存，并为退市股票记录 `delist_date`
- 用交易日历推断“目标最新交易日”，而不是简单拿今天日期判断是否完整
- 对长历史表按时间块分段抓取，避免 `6000` 行截断导致“看似完成，实际缺早期历史”
- `processed/combined` 当前默认把 `open/high/low/close` 统一到 `hfq`，同时保留 `raw_open/raw_high/raw_low/raw_close/raw_pre_close`

当前限流处理已经改成 stage-level cooldown 调度，而不是固定 sleep：
- 某个接口如果明确返回“每分钟最多访问该接口...”，当前 stage 会进入 `60s cooldown`
- 调度器会先去跑其他 stage
- 只有当所有仍有 pending 的 stage 都处于 cooldown 时，才会等待最早恢复的那个

如果你想先看真实接口列名和速度，再决定是否接入新的财务表，可以用探针脚本：
```bash
uv run python src/probe_tushare.py --symbol 000333.SZ
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

之后无论是单次训练还是 rolling，都可以只在训练阶段通过 profile 或 `features.selected_columns` 挑选子集，不需要重复生成 cache。
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
当前推荐优先使用 native + LightGBM，并通过命名实验 profile 运行：
```bash
uv run python run_native_rolling.py --experiment-profile core_v4_lgbm_default_10x20x10
```

训练/回测入口现在不再依赖默认 experiment profile。
请显式传入 `--experiment-profile`。

如果要直接从命令行切换 feature / model profile，不必改 `config.yaml`：
```bash
uv run python run_native_rolling.py --experiment-profile core_v4_lgbm_default_10x20x10 --feature-profile alpha158_full --run-tag alpha158_full
uv run python run_native_rolling.py --experiment-profile core_v4_lgbm_default_10x20x10 --model-profile lgbm_fast --run-tag fast_profile
```

同一实验做策略参数对比时，可以在 experiment profile 基础上直接覆写：
```bash
uv run python run_native_rolling.py --experiment-profile core_v4_lgbm_default_10x20x10 --topk 20 --n-drop 4 --run-tag top20_drop4
uv run python run_native_rolling.py --experiment-profile core_v4_lgbm_default_10x20x10 --topk 30 --n-drop 5 --run-tag top30_drop5
```

如果你不想为了一个小参数改动再复制一份 experiment yaml，现在也可以直接用通用覆写：
```bash
uv run python run_native_rolling.py --experiment-profile core_v4_lgbm_default_10x20x10 --set strategy.topk=20 --set rolling.retrain_step=5
uv run python main.py --experiment-profile core_v4_lgbm_default_10x20x10 --set label.signal_horizon=10
```

### 批量参数扫描

如果你想批量扫描一组参数，不要再复制很多 experiment yaml。
现在可以直接用批量入口：
```bash
uv run python run_experiment_batch.py \
  --pipeline rolling \
  --experiment-profile core_v4_lgbm_default_10x20x10 \
  --sweep 'rolling.retrain_step=[5,10,15]' \
  --run-tag-prefix retrain_sweep
```

也可以同时扫多个维度，按笛卡尔积展开：
```bash
uv run python run_experiment_batch.py \
  --pipeline rolling \
  --experiment-profile core_v4_lgbm_default_10x20x10 \
  --sweep 'rolling.retrain_step=[5,10,15]' \
  --sweep 'strategy.topk=[20,30]'
```

固定参数可以继续用 `--set`，它会应用到每一个子运行：
```bash
uv run python run_experiment_batch.py \
  --pipeline rolling \
  --experiment-profile core_v4_lgbm_default_10x20x10 \
  --data-source tushare \
  --set strategy.n_drop=5 \
  --sweep 'rolling.retrain_step=[5,10,15]'
```

当前 batch runner 默认是串行顺序执行。
如果你只是想先确认会展开哪些命令，可以加：
```bash
uv run python run_experiment_batch.py --pipeline rolling --experiment-profile core_v4_lgbm_default_10x20x10 --sweep 'rolling.retrain_step=[5,10,15]' --dry-run
```

### 单次实验 (研究模式)
用于快速验证想法：
```bash
uv run python main.py --experiment-profile core_v4_lgbm_default_10x20x10 --save-model results/lgbm/model.pkl
```

同样支持命令行覆写 profile：
```bash
uv run python main.py --experiment-profile core_v4_lgbm_default_10x20x10 --feature-profile alpha158_full --run-tag alpha158_full_single
```

也可以让系统自动把模型和实验元数据归档到本地实验库：
```bash
uv run python main.py --experiment-profile core_v4_lgbm_default_10x20x10 --topk 25 --n-drop 5 --run-tag alpha25
```

## 3. 模型复用 (Save & Load)

利用之前保存的滚动专家模型，实现秒级快速回测：
```bash
# 加载 native rolling 模型库
uv run python run_native_rolling.py --experiment-profile core_v4_lgbm_default_10x20x10 --load-models
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

## 5. 配置分层

当前推荐按以下层级管理配置：

- `configs/config.yaml`: 运行时配置，只放路径、存储、本地环境默认值
- `configs/feature_profiles.yaml` + `configs/features/*.yaml`: feature profile，只定义训练选列
- `configs/model_profiles.yaml` + `configs/models/*.yaml`: model profile，只定义模型与训练超参
- `configs/experiment_profiles.yaml` + `configs/experiments/*.yaml`: experiment profile，定义完整实验语义

其中实验语义包括：

- `signal_horizon`
- `retrain_step`
- `rebalance_freq`
- `topk`
- `n_drop`
- 时间切分
- universe
- 实验级 transforms

术语约定：

- `signal_horizon`: 模型预测的前瞻周期
- `retrain_step`: rolling 多久重训一次
- `rebalance_freq`: 组合多久调仓一次

不要再使用孤立的 `horizon` 概念。

示例：
```yaml
features:
  profile: core_v4_techlite

model:
  profile: lgbm_default

label:
  signal_horizon: 20

rolling:
  retrain_step: 10

backtest:
  rebalance_freq: 10
```

改 feature profile、model profile、experiment profile，都不需要重新执行 `gen_feature.py`。
只有当 unified factor store 的生成空间本身变化时，才需要重建 cache。

为什么 `gen_feature.py` 仍然独立存在，而不是在主训练脚本里隐式生成：
- factor store 生成是一个重 I/O、重 CPU 的预处理步骤，耗时和训练完全不是一个量级。
- 训练入口保持“只消费已有 factor store”，复现性更强，也更容易比较不同模型、不同选列、不同策略。
- 同一个全量 factor store 可以被很多次训练复用，这正好符合“先生成最全，再按需挑选”的研究方式。

如果后续要进一步提效，推荐新增一个显式模式，例如 `main.py --build-cache-if-missing`，而不是让训练脚本默认偷偷重建 cache。

LightGBM 的训练参数应当优先写进 model profile，而不是硬编码在 Python 中。
例如：

- `num_boost_round`
- `early_stop`
- `early_stopping_min_delta`
- `learning_rate`
- `num_leaves`

如果只想先做配置审核，而不真正开始训练，可以单独运行：

```bash
uv run python -m src.config_validation --config configs/config.yaml --experiment-profile core_v4_lgbm_default_10x20x10
```

LightGBM 训练会自动输出特征重要性：
- 单次实验：`results/native/lgbm/feature_importance_gain.csv`
- 滚动实验：`results/native_rolling_lgbm/feature_importance_gain_mean.csv`

## 6. 结果分析 (Analysis)

当前推荐结果目录一般为 `results/native_rolling_lgbm/`：
- `native_monthly_heatmap.png`: 月度收益红绿矩阵图。
- `native_monthly_report.csv`: **数字化月度报表**，方便 AI 进一步分析。
- `native_cumulative_return.png`: 包含真实滑点与限制的收益曲线。
- `models/`: 存放各时间段的专家模型权重。
