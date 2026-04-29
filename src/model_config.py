"""Helpers for resolving model-specific config blocks."""


def get_lgbm_config(cfg: dict) -> dict:
    """Return LightGBM-specific config without leaking LSTM-only fields."""
    lgbm_cfg = dict(cfg.get("lgbm", {}))
    model_cfg = cfg.get("model", {})
    strategy_cfg = cfg.get("strategy", {})

    if "early_stop" not in lgbm_cfg and "early_stop" in model_cfg:
        lgbm_cfg["early_stop"] = model_cfg["early_stop"]
    if "validation_topk" not in lgbm_cfg and "topk" in strategy_cfg:
        lgbm_cfg["validation_topk"] = int(strategy_cfg["topk"])

    return lgbm_cfg
