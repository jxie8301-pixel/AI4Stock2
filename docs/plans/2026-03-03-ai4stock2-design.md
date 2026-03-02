# AI4Stock2 设计文档

## 概述

基于 Qlib 框架的 A 股量化投资项目，采用模块化 Python 代码架构，由简到繁探索深度学习模型在量化选股中的应用。

**核心定位**：学习 + 逐步实战
**目标市场**：A 股（先 CSI300，后续扩展）
**模型路线**：LSTM → Transformer → GNN

## 项目结构

```
AI4Stock2/
├── configs/
│   └── config.yaml              # 全局配置
├── data/                         # Qlib 二进制数据（gitignore）
├── src/
│   ├── __init__.py
│   ├── data_setup.py             # Qlib 初始化 + 数据下载
│   ├── features.py               # 因子构建（Alpha158）
│   ├── dataset.py                # 数据集构建（TSDatasetH）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py         # Phase 1
│   │   ├── transformer_model.py  # Phase 2
│   │   └── gnn_model.py          # Phase 3
│   ├── strategy.py               # TopK 选股策略
│   ├── backtest.py               # 回测封装
│   └── evaluate.py               # 指标计算 + 可视化
├── results/                      # 回测输出（gitignore）
├── notebooks/                    # 探索性分析
├── main.py                       # 主入口
└── pyproject.toml
```

## 数据层

### 数据源
- Qlib 内置 A 股日频数据（通过 `qlib dump` 下载）
- 存储格式：Qlib 二进制格式，存入 `data/qlib_data_cn`

### 因子
- 使用 Qlib Alpha158 DataHandler，生成 158 个技术因子
- 包含均线、动量、波动率、成交量等维度

### 数据集
- 格式：TSDatasetH（时序格式），每只股票回看 60 个交易日
- 输入形状：`(batch, 60, 158)`

### 时间划分
| 集合 | 时间范围 |
|------|----------|
| 训练集 | 2008-01-01 ~ 2018-12-31 |
| 验证集 | 2019-01-01 ~ 2020-12-31 |
| 测试集 | 2021-01-01 ~ 2023-12-31 |

### 预测目标
未来 5 日收益率：`Ref($close, -5) / $close - 1`

## 模型层

### 统一接口

三个模型继承统一基类：

```python
class BaseModel:
    def fit(self, dataset) -> None
    def predict(self, dataset) -> Series
    def save(self, path) -> None
    def load(self, path) -> None
```

### Phase 1：LSTM

```
输入: (batch, 60, 158)
  → LSTM(input=158, hidden=64, layers=2, dropout=0.1)
  → 取最后时刻隐藏状态: (batch, 64)
  → Linear(64, 1)
  → 预测分数
```

训练：MSE Loss，Adam 优化器，lr=0.001，early stop patience=20

### Phase 2：Transformer

```
输入: (batch, 60, 158)
  → Linear(158, 64) 维度映射
  → PositionalEncoding(64)
  → TransformerEncoder(nhead=4, layers=2, dim_ff=256)
  → 均值池化: (batch, 64)
  → Linear(64, 1)
  → 预测分数
```

### Phase 3：GNN

```
构建股票关系图（节点=股票，边=行业/相关性）
  → 每只股票用 LSTM/Transformer 编码时序: (N_stocks, 64)
  → GAT(in=64, hidden=64, heads=4, layers=2)
  → Linear(64, 1)
  → 每只股票预测分数
```

## 策略层

### TopK 选股策略
- 每 5 个交易日调仓一次
- 按模型分数降序选 Top 30 只
- 等权分配资金
- 限制换手：每期最多换 5 只（TopkDropoutStrategy）

### 交易成本
| 项目 | 费率 |
|------|------|
| 买入佣金 | 0.03% |
| 卖出佣金 | 0.03% |
| 印花税（卖出）| 0.1% |

## 评估层

### 信号质量指标
| 指标 | 含义 | 良好标准 |
|------|------|----------|
| IC | 预测与收益的 Rank 相关 | > 0.05 |
| ICIR | IC均值 / IC标准差 | > 0.5 |
| Rank IC | Spearman 相关 | > 0.05 |

### 组合收益指标
- 年化收益率
- 夏普比率（> 1.5 优秀）
- 最大回撤
- 超额收益（vs CSI300）
- 换手率

### 可视化
- 累计收益曲线（策略 vs 基准）
- 月度收益热力图
- IC 时间序列图
- 回撤曲线

输出保存至 `results/`（图表 PNG + 指标 CSV）。

## 配置

全局配置 `configs/config.yaml`：

```yaml
qlib:
  provider_uri: ./data/qlib_data_cn
  region: cn

universe: csi300

time:
  train: ["2008-01-01", "2018-12-31"]
  valid: ["2019-01-01", "2020-12-31"]
  test:  ["2021-01-01", "2023-12-31"]

features:
  handler: Alpha158
  lookback: 60

model:
  name: lstm
  hidden_size: 64
  num_layers: 2
  dropout: 0.1
  lr: 0.001
  epochs: 200
  early_stop: 20

strategy:
  topk: 30
  n_drop: 5
```

## 运行方式

```bash
python main.py                        # 默认 LSTM
python main.py model.name=transformer # 切换 Transformer
```
