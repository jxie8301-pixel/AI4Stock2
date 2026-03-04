# AI4Stock2 Project Context & Architecture Map

## 1. 项目愿景 (Project Philosophy)
这是一个基于 Microsoft Qlib 框架的高级 A 股量化研究项目。其核心目标是通过深度学习模型（LSTM/Transformer）捕捉市场非线性 Alpha，并实现绝对真实、可落地的实战回测。

**核心理念：**
- **由简入繁**：从 `LSTM (时序特征)` 向 `Transformer` 和 `GNN` 演进。
- **真实为王**：拒绝一切“回测幻觉”，严格遵循 T+1 开盘价成交、涨跌停限制和滑点模型。
- **与时俱进**：通过“滚动训练”机制（Rolling Retrain）动态对抗 A 股的市场风格漂移。

## 2. 核心思路与工作流 (Logic Flow)
1.  **Data Layer (src/collector_akshare.py)**: 继承旧项目资产，利用 Cookie 劫持增量同步量价与估值数据。
2.  **Feature Layer (src/features.py)**: 扩展 Alpha158 算子，动态融合 `PE_TTM`、`PB`、`市值` 等 166 维特征。
3.  **Model Layer (src/models/)**: 
    - **内核**：AMP 加速 + AdamW 优化器。
    - **目标**：Pearson 相关性损失（直接优化 IC）。
4.  **Rolling Layer (run_rolling.py)**: 实现每半年为一个单位的滚动训练循环，拼接长周期预测信号。
5.  **Backtest Layer (src/backtest.py)**: 引入 `LimitRules` 和 `OpenPrice` 交易逻辑，确保收益真实。

## 3. 项目结构图 (Structure Map)
```text
AI4Stock2/
├── data/               # 核心数据库（raw/processed/qlib_bin）
├── configs/            # 全局实验配置中心
├── src/
│   ├── models/         # 模型架构与自定义 Loss (Pearson/CCC)
│   ├── collector_akshare.py # 带 Cookie 的全自动数据管线
│   ├── backtest.py     # 真实 A 股约束回测引擎
│   └── evaluate.py     # 月度报表与热力图分析
├── main.py             # 单次实验入口
└── run_rolling.py      # 滚动训练主控制塔
```

## 4. 关键决策记录 (Key Decisions)
- **为什么选 Open-to-Open Label?**：为了解决“信号发生在 T 日收盘，但交易发生在 T+1 开盘”的时间差，保证回测无偏。
- **为什么选 AdamW?**：相比 Adam，AdamW 能更有效地控制 L2 正则化，大幅缓解深度模型在金融噪声数据上的过拟合。
- **为什么选 csi300_real?**：通过清洗官方列表并去除前缀，构建了既符合 Qlib 规范又贴合 A 股真实成分的股票池。

## 5. 跨 AI 协作指南 (AI Interaction Protocol)
当后续 AI 接入此项目时，请优先阅读 `docs/PROGRESS_AND_TODO.md` 了解 V2.2 稳定版的具体参数与战绩。
- **新增模型**：需继承 `AMPLSTM` 或保持兼容的 `fit`/`predict` 接口。
- **分析历史**：直接读取 `results/rolling_lstm/monthly_report.csv` 即可获取最详细的数字版战报。
```
