# AI4Stock2 去 Qlib 迁移执行计划

## 1. 目标与范围

### 目标
- 在不牺牲可复现性与核心指标的前提下，移除运行时对 `qlib` 的依赖。
- 保留现有策略逻辑（Alpha 特征、Open-to-Open 标签、TopK 组合思想）。
- 最终使主流程、滚动训练、回测评估都可以在无 `pyqlib` 环境运行。

### 非目标
- 本计划不追求一次性重写所有历史脚本。
- 不要求第一版就达到与 qlib 完全一致的回测结果（先保证流程完整，再做指标逼近）。

---

## 2. 当前状态（截至本计划编写时）
- 已有原生特征生成：`src/gen_feature.py`（`panel_2d` 缓存）。
- 已有训练侧动态切窗：`src/models/pure_pytorch_lstm.py`。
- 仍存在 qlib 耦合：
  - 训练入口：`main.py`, `run_rolling.py`
  - 特征/数据集旧链路：`src/features.py`, `src/dataset.py`
  - 回测与评估：`src/backtest.py`, `src/evaluate.py`, `src/strategy.py`
  - 依赖声明：`pyproject.toml` 中 `pyqlib`

---

## 3. 执行原则
- 单阶段可回滚：每阶段完成后都可独立运行并验收。
- 双轨验证：迁移期间保留 `qlib` 旧路径，只做对照，不继续扩展功能。
- 指标守门：每阶段有明确 DoD（Definition of Done）。

---

## 4. 分阶段执行清单

## Phase 0 - 基线冻结（1 天）
- [ ] 固定一版对照配置与时间区间（建议 `2024-01-01` 到 `2025-12-31`）。
- [ ] 导出旧流程关键指标作为 baseline：
  - Signal: `IC_mean`, `Rank_IC_mean`, `ICIR`
  - Portfolio: 年化、Sharpe、最大回撤、换手
- [ ] 固定随机种子与版本信息，写入 `results/baseline/`.

### 验收
- [ ] baseline 指标可复跑且结果稳定（容忍轻微浮动）。

---

## Phase 1 - 主训练入口切换到原生缓存（2-3 天）
- [ ] 在 `main.py` 增加 `native` 路径，默认读取 `panel_2d` 缓存。
- [ ] 去掉 `main.py` 中对 `init_qlib`, `build_alpha158_handler`, `TSDatasetH` 的强耦合。
- [ ] 统一使用：
  - `src/gen_feature.py`（离线生成）
  - `src/models/pure_pytorch_lstm.py`（训练侧切窗）
- [ ] 在配置中保留 `lookback`，但由训练阶段使用，不在生成阶段使用。

### 验收
- [ ] `python main.py` 在无 qlib import 的情况下可完成：训练 + 预测 + 指标输出。

### 回滚点
- [ ] 保留旧入口分支（例如 `pipeline.backend: qlib|native`）。

---

## Phase 2 - 模型层去 Qlib（2-4 天）
- [ ] LSTM：使用 `pure_pytorch_lstm.py` 替代 `src/models/lstm_model.py` 的 qlib 版本。
- [ ] Transformer：新增纯 PyTorch 版本（替代 `src/models/transformer_model.py` qlib 版本）。
- [ ] LightGBM：改为直接 `lightgbm` 原生训练接口，替代 `qlib.contrib.model.gbdt`.
- [ ] 统一模型接口：`fit/predict/save/load`。

### 验收
- [ ] `lstm/transformer/lgbm` 三模型都可在 native 管线跑通。

---

## Phase 3 - 回测与评估去 Qlib（3-5 天）
- [ ] 替换 `risk_analysis`（`src/evaluate.py`）为本地实现：
  - 年化收益、波动、Sharpe、最大回撤、Calmar、月度收益。
- [ ] 替换 `backtest_daily + TopkDropoutStrategy`（`src/backtest.py`）为本地引擎：
  - TopK 持仓
  - `n_drop` 换仓约束
  - 开盘成交、买卖成本、最小费用
- [ ] 保持现有可视化输出接口不变（图和 JSON 字段尽量兼容）。

### 验收
- [ ] `main.py` 的“训练->预测->回测->报告”全流程不再 import qlib。

---

## Phase 4 - 滚动训练去 Qlib（2-4 天）
- [ ] 重写 `run_rolling.py` 的交易日历来源（不再使用 `qlib.data.D.calendar`）。
- [ ] 用本地交易日历（来自 panel 缓存日期）生成滚动窗口。
- [ ] 复用 native 训练/评估/回测组件。

### 验收
- [ ] `run_rolling.py` 在 native 模式跑通并产出完整结果文件。

---

## Phase 5 - 清理依赖与遗留代码（1-2 天）
- [ ] 将以下模块转为归档或删除：
  - `src/features.py`
  - `src/dataset.py`
  - `src/data_setup.py`（若仅服务 qlib）
  - qlib 相关 strategy/backtest 封装
- [ ] 从 `pyproject.toml` 移除 `pyqlib`.
- [ ] 更新 `README` 与 `docs/USER_GUIDE.md`，改为 native 使用说明。
- [ ] 新增 smoke tests（至少 3 个）：
  - 生成缓存
  - 训练一个 epoch
  - 回测输出关键字段

### 验收
- [ ] 全项目 `rg -n "from qlib|import qlib|pyqlib"` 无运行链路引用。

---

## Phase 6 - 指标对齐与性能优化（持续）
- [ ] 与 baseline 做差异分析：
  - IC/RankIC 偏差
  - 收益与回撤偏差
  - 换手偏差
- [ ] 性能优化优先级：
  - 特征计算并行与缓存复用
  - 训练侧 dataloader 吞吐
  - 回测循环向量化
- [ ] 设立发布门槛（建议）：
  - Rank_IC 偏差绝对值 <= 0.01
  - 年化偏差 <= 15%
  - 最大回撤偏差 <= 20%

---

## 5. 推荐执行顺序（最小风险）
1. Phase 0 基线冻结  
2. Phase 1 主入口 native 化  
3. Phase 3 回测评估替换（先打通闭环）  
4. Phase 2 模型全替换  
5. Phase 4 rolling 改造  
6. Phase 5 清理依赖  
7. Phase 6 对齐与提速

---

## 6. 每周交付建议
- Week 1：Phase 0 + Phase 1  
- Week 2：Phase 3（先回测）+ Phase 2（先 LSTM）  
- Week 3：Phase 2（Transformer/LGBM）+ Phase 4  
- Week 4：Phase 5 + Phase 6 初轮对齐

---

## 7. 风险清单
- 标签/执行价错位导致回测虚高。
- 时间切分泄漏（标准化必须仅用 train 拟合）。
- rolling 窗口边界处理不一致导致信号偏差。
- 多进程写缓存时的并发冲突（需固定切片写入，禁止重叠）。

---

## 8. 立即可执行的下一步
- [ ] 先完成 Phase 0：输出一份稳定 baseline 报告。
- [ ] 然后在 `main.py` 加 `native` 主路径，确保端到端跑通。
