# AI4Stock2 Project Context & Architecture Map

## 1. 项目愿景 (Project Philosophy)
这是一个基于 Microsoft Qlib 框架的 A 股量化研究项目。其核心目标是通过“由简入繁”的迭代路径，探索深度学习在 Alpha 预测中的实战能力。

**核心理念：**
- **模块化而非配置驱动**：虽然 Qlib 支持 YAML 驱动，但本项目选择了 **Python 模块化架构**（Plan B），旨在提高灵活性并方便 AI 助手理解底层的逻辑流动。
- **研究范式**：从预测绝对价格转向预测“截面收益率排名”（Learning to Rank），这是量化研究的核心范式。
- **渐进式演进**：模型路线设定为 `LSTM (时序特征)` -> `Transformer (全局注意) ` -> `GNN (行业/关联特征)`。

## 2. 核心思路与工作流 (Logic Flow)
整个项目遵循标准的量化研究闭环，各模块通过清晰的接口解耦：

1.  **Data Layer (src/data_setup.py)**: 处理 Qlib 环境初始化与原始二进制数据同步。
2.  **Feature Layer (src/features.py)**: 利用 Alpha158 因子库构建基础特征池。
3.  **Dataset Layer (src/dataset.py)**: 将特征转换为时序张量（TSDatasetH），为深度学习模型提供 `(batch, lookback, features)` 形状的输入。
4.  **Model Layer (src/models/)**: 
    - 统一采用 `fit` / `predict` 接口。
    - 关注点在于捕捉时序依赖（Time-series dependency）。
5.  **Strategy Layer (src/strategy.py & backtest.py)**: 
    - **TopK Dropout 逻辑**：根据预测分数选择前 N 只股票，并引入换手控制（n_drop）以平衡交易成本。
6.  **Evaluation Layer (src/evaluate.py)**: 
    - 信号层面：关注 IC (Information Coefficient) 和 ICIR。
    - 组合层面：关注夏普比率、最大回撤和超额收益（vs CSI300）。

## 3. 项目结构图 (Structure Map)
```text
AI4Stock2/
├── configs/            # 实验参数（时间范围、模型超参、交易成本）
├── src/                # 核心逻辑
│   ├── models/         # 深度学习模型实现（LSTM, Transformer...）
│   ├── data_setup.py   # 数据获取与初始化
│   ├── features.py     # 因子工程
│   ├── dataset.py      # 数据格式转换（面向神经网络）
│   ├── strategy.py     # 调仓逻辑
│   ├── backtest.py     # Qlib 回测引擎封装
│   └── evaluate.py     # 指标计算与可视化中心
├── main.py             # 全流程串联入口（Pipeline Orchestrator）
└── results/            # 实验产出（模型权重、预测分数、回测图表）
```

## 4. 关键决策记录 (Key Decisions)
- **为什么选 Alpha158?**：作为 baseline，它涵盖了 A 股常用的技术指标，能有效验证 pipeline 是否打通。
- **为什么选 TSDatasetH?**：Qlib 原生支持将非对齐的时序数据高效切片，是进行深度学习训练的最佳中间层。
- **换手控制策略**：在策略模块中强制限制单次调仓比例，这是因为 A 股的印花税和佣金对高频换手极其敏感。

## 5. 跨 AI 协作指南 (AI Interaction Protocol)
当后续 AI 接入此项目时，应遵循以下指引：
- **新增模型**：在 `src/models/` 下新建类，并保持与 `main.py` 中 `_build_model` 的适配。
- **修改因子**：在 `src/features.py` 中调整 DataHandler，确保返回的数据列名能被模型正确识别。
- **环境隔离**：所有 Qlib 二进制数据默认存放在根目录 `data/` 下，该目录已被 `.gitignore` 忽略。

## 6. 运行指令速查 (Usage at a Glance)
- **下载数据**: `python main.py --download-only`
- **基础运行**: `python main.py`
- **切换模型**: `python main.py --model transformer`
- **性能调试**: `python main.py --gpu -1 --skip-backtest` (CPU + 仅信号测试)

详细操作请参考 [USER_GUIDE.md](./USER_GUIDE.md)。
