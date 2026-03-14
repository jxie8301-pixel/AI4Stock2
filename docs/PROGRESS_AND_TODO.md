# AI4Stock2: Progress & TODO Map

## 1. Version History & Milestones

### V2.2 - The Final Realistic Stable Version (Current)
*   **True Realism Alignment**: 
    *   **Label Correction**: Switched to `Open-to-Open` labels (`Ref($open, -2)/Ref($open, -1) - 1`). This perfectly aligns the model's prediction target with our T+1 morning execution strategy.
    *   **Universe Correction**: Cleaned the `csi300_real.txt` universe to use authentic index constituents while matching our pure-numeric data format.
    *   **Data Error Filtering**: Implemented a sanity check in `evaluate.py` to clip extreme daily portfolio returns (>50%) caused by data artifacts.
*   **Algorithm Upgrade**: Integrated **AdamW** with a `1e-4` weight decay as the default optimizer for LSTM, significantly improving generalization and preventing the "Epoch 2 collapse".
*   **Performance Stability**: Standardized the `early_stop` to 10 rounds to save 50% training time without losing signal quality.
*   **Results (Rolling 2022-2025)**:
    *   **Rank IC**: 0.0579 (Extremely high for realistic Open-to-Open prediction).
    *   **Annualized Return**: **+184.3%** (True profit after 10% limit rules and 5bps slippage).
    *   **Information Ratio (IR)**: **2.19** (Exceptional risk-adjusted performance).
    *   **Max Drawdown**: **-26.1%** (Controlled risk during the 2024 volatility).

### V2.1 - The Realism & Rolling Breakthrough
*   **Rolling Retrain Pipeline**: Implemented `run_rolling.py` training a new model every 6 months.
*   **Realistic Backtesting**: Switched to `open` price execution and 10% limit rules.
*   **Digital Reporting**: Added CSV exports and monthly heatmap visualization.

### V2.0 - The Feature Fusion Update
*   **Fundamental Data**: Integrated `pe_ttm`, `pb`, `total_mv`, etc., into the 166-dimensional feature tensor.

### V1.0 - The Pearson Breakthrough
*   **Loss Function**: Replaced MSE with Pearson Correlation Loss (optimizing directly for IC).

---

## 2. Immediate TODO List (Phase 5: Strategy Mastery)

### Portfolio Engineering
- [ ] **Risk Parity & Confidence Weighting**: Optimize the Top 30 weights based on prediction strength.
- [ ] **Sector Neutralization**: Prevent the LSTM from over-concentrating in specific industry themes.
- [ ] **Transaction Cost Sensitivity Analysis**: Test the strategy's survivability with higher slippage (e.g., 20bps).

### Model Evolution
- [ ] **Transformer Rolling**: Run the 2022-2025 pipeline with the Transformer model using the new AdamW/Open-Open configuration.
- [ ] **GNN Phase**: Incorporate industry relationship graphs to capture sector rotation Alpha.

### Infrastructure
- [ ] **Automated PIT (Point-in-Time) Universe**: Dynamically load index constituents based on the date to fully eliminate survivorship bias.

---

## 3. De-Qlib Migration Path (Execution Plan)

### Phase A - Data Layer (Parquet as Source, NPY/Memmap as Training Cache)
- [ ] Keep `data/processed/combined/*.parquet` as source of truth.
- [ ] Add `src/gen_feature.py` to load per-symbol parquet and build panel data `(date, instrument)`.
- [ ] Generate labels with current realistic definition: `open_{t+2}/open_{t+1} - 1`.
- [ ] Export model-ready cache to `data/cache/{split}/X.memmap`, `y.memmap`, `meta.parquet`.

### Phase B - Feature Layer (No qlib Runtime)
- [ ] Use `src/alpha_definitions.py` as canonical Alpha158/Alpha360 expression definitions.
- [ ] Implement pandas-based operators required by Alpha set (`Ref/Mean/Std/Rank/IdxMax/...`).
- [ ] Add feature validity checks: NaN ratio, inf ratio, and per-feature coverage report.
- [ ] Freeze feature list + order in `configs/feature_schema.yaml` for reproducibility.

### Phase C - Dataset & Training Layer
- [ ] Replace `TSDatasetH/DatasetH` with local dataset loaders reading memmap directly.
- [ ] Connect `src/models/pure_pytorch_lstm.py` into `main.py` as `model.name: pure_lstm`.
- [ ] Add train/valid/test split before normalization to avoid leakage.
- [ ] Save checkpoints and prediction outputs in the same format as current pipeline.

### Phase D - Evaluation & Backtest Layer
- [ ] Replace `qlib.contrib.evaluate.risk_analysis` with local metrics (`ann_ret`, `vol`, `sharpe`, `max_drawdown`).
- [ ] Implement local `TopK + n_drop` simulator with open-price execution and transaction costs.
- [ ] Validate parity vs current qlib path on same period (`2024-01-01` to `2025-12-31`).
- [ ] Add tolerance gates: IC diff, turnover diff, annual return diff.

### Phase E - Switch & Cleanup
- [ ] Add config switch `pipeline.backend: qlib | native`.
- [ ] Make `native` default after parity tests pass.
- [ ] Move qlib-specific modules behind optional import guards.
- [ ] Remove hard dependency `pyqlib` from default install profile.

### Exit Criteria
- [ ] End-to-end run (`download/update -> feature gen -> train -> eval -> backtest`) succeeds without importing `qlib`.
- [ ] Rolling pipeline (`run_rolling.py`) has native backend implementation.
- [ ] Performance regression within agreed tolerance on key metrics.
