"""Helpers for resolving model-specific config blocks."""


def get_lgbm_config(cfg: dict) -> dict:
    """Return LightGBM-specific config without leaking LSTM-only fields."""
    lgbm_cfg = dict(cfg.get("lgbm", {}))
    model_cfg = cfg.get("model", {})

    if "early_stop" not in lgbm_cfg and "early_stop" in model_cfg:
        lgbm_cfg["early_stop"] = model_cfg["early_stop"]
    if "num_threads" not in lgbm_cfg and "n_jobs" in model_cfg:
        lgbm_cfg["num_threads"] = model_cfg["n_jobs"]

    return lgbm_cfg
