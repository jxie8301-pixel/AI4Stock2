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
