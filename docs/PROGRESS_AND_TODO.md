# AI4Stock2: Progress & TODO Map

## 1. Current State (V1.0 - The Pearson Breakthrough)
We have successfully established a robust, modular quantitative research pipeline using Qlib, AkShare, and PyTorch. The system has evolved from a basic demonstration into a highly competitive "Learning to Rank" engine.

### Major Achievements
- **Data Pipeline Rebuilt**: 
  - Restored the old `AI4Stock` directory structure (`data/raw/daily`, `data/raw/valuation`, `data/processed/combined`).
  - Added Cookie/TLS impersonation via `curl_cffi` to reliably fetch HFQ data from EastMoney without getting blocked.
  - Successfully exported all 5,188 A-share stocks with both price/volume and valuation data (PE, PB, MV) into Qlib's binary format up to 2026.
- **Model Architecture Upgrade**:
  - Replaced the naive MSE (Mean Squared Error) loss with a custom **Pearson Correlation Loss**.
  - The model now directly optimizes for the Information Coefficient (IC), shifting the paradigm from regression to "Learning to Rank".
- **Hyperparameter Optimization**:
  - Shortened the LSTM lookback window from `60` days to `20` days to reduce noise and adapt to rapid style rotations in the A-share market.
  - Set the universe to all A-shares with infinite lifespan bounds (`2000-2099`) to prevent empty dataset crashes during extreme market drops.
- **Workflow Enhancements**:
  - Added model serialization (`--save-model` and `--load-model`) allowing instant backtests without retraining.

---

## 2. Immediate TODO List (The Valuation Evolution)

### Phase 2: Feature Fusion (Valuation + Price/Volume)
*Currently, the model only sees the 158 price/volume factors. We have the valuation data in Qlib, but the model doesn't know how to read it.*
- [ ] **Custom DataHandler**: Rewrite `src/features.py` to create a `ValuationAlpha` handler.
- [ ] **Feature Selection**: Inject `pe_ttm`, `pb`, `circ_mv` (Circulating Market Value), and `turnover` into the feature tensor.
- [ ] **Cross-Sectional Normalization**: Ensure valuation factors are properly neutralized (e.g., Z-Score across the universe per day) so large-cap stocks don't overwhelm the network weights.

### Phase 3: Backtest Realism (Curing the Illusion)
*The current 225% annualized return is partly an illusion caused by ideal trading conditions.*
- [ ] **Limit Rules**: Implement Qlib's `LimitRules` to prevent the backtest engine from buying stocks that open at the +10% limit (一字涨停) or selling stocks at the -10% limit (一字跌停).
- [ ] **Volume/Liquidity Constraints**: Restrict trade volume to a maximum of e.g., 5% of a stock's daily turnover.
- [ ] **Slippage Modeling**: Increase the effective cost to simulate slippage, especially for small-cap stocks.

### Phase 4: Rolling Retrain (Walk-forward Analysis)
- [ ] **Concept Drift Mitigation**: Implement Qlib's `RollingGen` or write a custom script to train a model every 6 months (e.g., train on 2015-2020 to predict 2021H1, then train on 2015-2020H2 to predict 2021H2) to test true out-of-sample robustness over long periods.