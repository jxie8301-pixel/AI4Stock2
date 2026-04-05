"""Feature family definitions and canonical naming helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from src.data_source import normalize_data_source_name
except ModuleNotFoundError:
    from data_source import normalize_data_source_name  # type: ignore


EPS = 1e-12
FULL_FACTOR_SPACE_NAME = "full_factor_space"
DEFAULT_FULL_FACTOR_STORE_DIR = "data/factor_store/full_factor_space"


DEFAULT_ALPHA158_CONFIG: dict[str, Any] = {
    "kbar": {},
    "price": {
        "windows": [0],
        "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
    },
    "rolling": {},
}


DEFAULT_LGBM_PURIFIED_CONFIG: dict[str, Any] = {
    "momentum_windows": [20, 60],
    "ma_windows": [20, 60, 120],
    "vol_window": 60,
    "atr_window": 14,
    "amihud_window": 20,
    "volume_window": 20,
    "corr_window": 20,
    "turnover_window": 20,
    "extreme_window": 20,
}

DEFAULT_TEMPORAL_FACTOR_CONFIG: dict[str, Any] = {
    "windows": [1, 5, 10, 20, 30, 60, 120],
    "groups": [
        "ret",
        "ma_gap",
        "std",
        "rsv",
        "price_rank",
        "volume_ratio",
        "turnover_mean",
        "amihud",
        "high_gap",
        "low_gap",
        "corr_cv",
    ],
}

DEFAULT_TECHNICAL_FACTOR_CONFIG: dict[str, Any] = {
    "macd_setups": [
        {"fast": 12, "slow": 26, "signal": 9},
    ],
    "rsi_windows": [6, 14, 24],
    "boll_windows": [20, 60],
    "boll_num_std": 2,
    "adx_windows": [14],
    "mfi_windows": [14],
    "cci_windows": [20],
    "willr_windows": [14],
    "aroon_windows": [25],
    "trix_windows": [15],
    "trix_signal": 9,
    "obv_windows": [20, 60],
}

DEFAULT_TUSHARE_FACTOR_CONFIG: dict[str, Any] = {
    "free_turnover_windows": [5, 20],
    "limit_stat_windows": [5, 20],
    "amplitude_windows": [5, 20],
    "pct_chg_windows": [5, 20],
    "ratio_change_windows": [20, 60],
    "valuation_change_windows": [20, 60],
    "zscore_window": 20,
}


ALL_FACTORS_ALPHA360_PREFIX = "A360_"
ALL_FACTORS_LGBM_PREFIX = "LGBM_"
TEMPORAL_FACTOR_PREFIX = "TEMP_"
TECHNICAL_FACTOR_PREFIX = "TECH_"
TUSHARE_FACTOR_PREFIX = "TS_"


def get_alpha158_feature_config(config: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """Return Alpha158 expression fields and names (qlib-compatible)."""
    cfg = deepcopy(DEFAULT_ALPHA158_CONFIG)
    if config is not None:
        cfg.update(config)

    fields: list[str] = []
    names: list[str] = []

    if "kbar" in cfg:
        fields += [
            "($close-$open)/$open",
            "($high-$low)/$open",
            "($close-$open)/($high-$low+1e-12)",
            "($high-Greater($open, $close))/$open",
            "($high-Greater($open, $close))/($high-$low+1e-12)",
            "(Less($open, $close)-$low)/$open",
            "(Less($open, $close)-$low)/($high-$low+1e-12)",
            "(2*$close-$high-$low)/$open",
            "(2*$close-$high-$low)/($high-$low+1e-12)",
        ]
        names += ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]

    if "price" in cfg:
        windows = cfg["price"].get("windows", range(5))
        feature = cfg["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
        for field in feature:
            field = field.lower()
            fields += [f"Ref(${field}, {d})/$close" if d != 0 else f"${field}/$close" for d in windows]
            names += [field.upper() + str(d) for d in windows]

    if "volume" in cfg:
        windows = cfg["volume"].get("windows", range(5))
        fields += [f"Ref($volume, {d})/($volume+1e-12)" if d != 0 else "$volume/($volume+1e-12)" for d in windows]
        names += [f"VOLUME{d}" for d in windows]

    if "rolling" in cfg:
        windows = cfg["rolling"].get("windows", [5, 10, 20, 30, 60])
        include = cfg["rolling"].get("include", None)
        exclude = cfg["rolling"].get("exclude", [])

        def use(op: str) -> bool:
            return op not in exclude and (include is None or op in include)

        if use("ROC"):
            fields += [f"Ref($close, {d})/$close" for d in windows]
            names += [f"ROC{d}" for d in windows]
        if use("MA"):
            fields += [f"Mean($close, {d})/$close" for d in windows]
            names += [f"MA{d}" for d in windows]
        if use("STD"):
            fields += [f"Std($close, {d})/$close" for d in windows]
            names += [f"STD{d}" for d in windows]
        if use("BETA"):
            fields += [f"Slope($close, {d})/$close" for d in windows]
            names += [f"BETA{d}" for d in windows]
        if use("RSQR"):
            fields += [f"Rsquare($close, {d})" for d in windows]
            names += [f"RSQR{d}" for d in windows]
        if use("RESI"):
            fields += [f"Resi($close, {d})/$close" for d in windows]
            names += [f"RESI{d}" for d in windows]
        if use("MAX"):
            fields += [f"Max($high, {d})/$close" for d in windows]
            names += [f"MAX{d}" for d in windows]
        if use("LOW"):
            fields += [f"Min($low, {d})/$close" for d in windows]
            names += [f"MIN{d}" for d in windows]
        if use("QTLU"):
            fields += [f"Quantile($close, {d}, 0.8)/$close" for d in windows]
            names += [f"QTLU{d}" for d in windows]
        if use("QTLD"):
            fields += [f"Quantile($close, {d}, 0.2)/$close" for d in windows]
            names += [f"QTLD{d}" for d in windows]
        if use("RANK"):
            fields += [f"Rank($close, {d})" for d in windows]
            names += [f"RANK{d}" for d in windows]
        if use("RSV"):
            fields += [f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)" for d in windows]
            names += [f"RSV{d}" for d in windows]
        if use("IMAX"):
            fields += [f"IdxMax($high, {d})/{d}" for d in windows]
            names += [f"IMAX{d}" for d in windows]
        if use("IMIN"):
            fields += [f"IdxMin($low, {d})/{d}" for d in windows]
            names += [f"IMIN{d}" for d in windows]
        if use("IMXD"):
            fields += [f"(IdxMax($high, {d})-IdxMin($low, {d}))/{d}" for d in windows]
            names += [f"IMXD{d}" for d in windows]
        if use("CORR"):
            fields += [f"Corr($close, Log($volume+1), {d})" for d in windows]
            names += [f"CORR{d}" for d in windows]
        if use("CORD"):
            fields += [f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})" for d in windows]
            names += [f"CORD{d}" for d in windows]
        if use("CNTP"):
            fields += [f"Mean($close>Ref($close, 1), {d})" for d in windows]
            names += [f"CNTP{d}" for d in windows]
        if use("CNTN"):
            fields += [f"Mean($close<Ref($close, 1), {d})" for d in windows]
            names += [f"CNTN{d}" for d in windows]
        if use("CNTD"):
            fields += [f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})" for d in windows]
            names += [f"CNTD{d}" for d in windows]
        if use("SUMP"):
            fields += [
                f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"SUMP{d}" for d in windows]
        if use("SUMN"):
            fields += [
                f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"SUMN{d}" for d in windows]
        if use("SUMD"):
            fields += [
                f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))"
                f"/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"SUMD{d}" for d in windows]
        if use("VMA"):
            fields += [f"Mean($volume, {d})/($volume+1e-12)" for d in windows]
            names += [f"VMA{d}" for d in windows]
        if use("VSTD"):
            fields += [f"Std($volume, {d})/($volume+1e-12)" for d in windows]
            names += [f"VSTD{d}" for d in windows]
        if use("WVMA"):
            fields += [
                f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"
                for d in windows
            ]
            names += [f"WVMA{d}" for d in windows]
        if use("VSUMP"):
            fields += [
                f"Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"VSUMP{d}" for d in windows]
        if use("VSUMN"):
            fields += [
                f"Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"VSUMN{d}" for d in windows]
        if use("VSUMD"):
            fields += [
                f"(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))"
                f"/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"VSUMD{d}" for d in windows]

    return fields, names


def get_alpha360_feature_config() -> tuple[list[str], list[str]]:
    """Return Alpha360 expression fields and names (qlib-compatible)."""
    fields: list[str] = []
    names: list[str] = []

    for i in range(59, 0, -1):
        fields.append(f"Ref($close, {i})/$close")
        names.append(f"CLOSE{i}")
    fields.append("$close/$close")
    names.append("CLOSE0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($open, {i})/$close")
        names.append(f"OPEN{i}")
    fields.append("$open/$close")
    names.append("OPEN0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($high, {i})/$close")
        names.append(f"HIGH{i}")
    fields.append("$high/$close")
    names.append("HIGH0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($low, {i})/$close")
        names.append(f"LOW{i}")
    fields.append("$low/$close")
    names.append("LOW0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($vwap, {i})/$close")
        names.append(f"VWAP{i}")
    fields.append("$vwap/$close")
    names.append("VWAP0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($volume, {i})/($volume+1e-12)")
        names.append(f"VOLUME{i}")
    fields.append("$volume/($volume+1e-12)")
    names.append("VOLUME0")

    return fields, names


def get_lgbm_purified_feature_names(config: dict[str, Any] | None = None) -> list[str]:
    cfg = deepcopy(DEFAULT_LGBM_PURIFIED_CONFIG)
    if config is not None:
        cfg.update(config)

    names: list[str] = []
    names += [f"ret_{d}" for d in cfg["momentum_windows"]]
    names += [f"dist_ma{d}" for d in cfg["ma_windows"]]
    names += ["std_60", "atr_14", "amihud_20", "vol_ratio_20", "corr_cv_20", "vwap_ratio"]
    names += ["log_mcap", "ep_ttm", "is_loss", "bp", "turnover_20", "dist_high_20", "dist_low_20"]
    return names


def get_temporal_factor_feature_names(config: dict[str, Any] | None = None) -> list[str]:
    cfg = deepcopy(DEFAULT_TEMPORAL_FACTOR_CONFIG)
    if config is not None:
        cfg.update(config)

    names: list[str] = []
    groups = list(cfg["groups"])
    windows = list(cfg["windows"])
    for window in windows:
        if "ret" in groups:
            names.append(f"ret_{window}")
        if "ma_gap" in groups:
            names.append(f"ma_gap_{window}")
        if "std" in groups:
            names.append(f"std_{window}")
        if "rsv" in groups:
            names.append(f"rsv_{window}")
        if "price_rank" in groups:
            names.append(f"price_rank_{window}")
        if "volume_ratio" in groups:
            names.append(f"volume_ratio_{window}")
        if "turnover_mean" in groups:
            names.append(f"turnover_mean_{window}")
        if "amihud" in groups:
            names.append(f"amihud_{window}")
        if "high_gap" in groups:
            names.append(f"high_gap_{window}")
        if "low_gap" in groups:
            names.append(f"low_gap_{window}")
        if "corr_cv" in groups:
            names.append(f"corr_cv_{window}")
    return names


def get_technical_factor_feature_names(config: dict[str, Any] | None = None) -> list[str]:
    cfg = deepcopy(DEFAULT_TECHNICAL_FACTOR_CONFIG)
    if config is not None:
        cfg.update(config)

    names: list[str] = []
    for setup in cfg["macd_setups"]:
        fast = int(setup["fast"])
        slow = int(setup["slow"])
        signal = int(setup["signal"])
        names += [
            f"macd_line_{fast}_{slow}_{signal}",
            f"macd_signal_{fast}_{slow}_{signal}",
            f"macd_hist_{fast}_{slow}_{signal}",
        ]
    for window in cfg["rsi_windows"]:
        names.append(f"rsi_{int(window)}")
    num_std = int(cfg["boll_num_std"])
    for window in cfg["boll_windows"]:
        window = int(window)
        names += [
            f"boll_pos_{window}_{num_std}",
            f"boll_width_{window}_{num_std}",
            f"boll_zscore_{window}_{num_std}",
        ]
    for window in cfg["adx_windows"]:
        window = int(window)
        names += [f"adx_{window}", f"plus_di_{window}", f"minus_di_{window}"]
    for window in cfg["mfi_windows"]:
        names.append(f"mfi_{int(window)}")
    for window in cfg["cci_windows"]:
        names.append(f"cci_{int(window)}")
    for window in cfg["willr_windows"]:
        names.append(f"willr_{int(window)}")
    for window in cfg["aroon_windows"]:
        window = int(window)
        names += [f"aroon_up_{window}", f"aroon_down_{window}", f"aroon_osc_{window}"]
    trix_signal = int(cfg["trix_signal"])
    for window in cfg["trix_windows"]:
        window = int(window)
        names += [f"trix_{window}", f"trix_signal_{window}_{trix_signal}", f"trix_hist_{window}_{trix_signal}"]
    for window in cfg["obv_windows"]:
        names.append(f"obv_flow_{int(window)}")
    return names


def get_tushare_factor_feature_names(config: dict[str, Any] | None = None) -> list[str]:
    cfg = deepcopy(DEFAULT_TUSHARE_FACTOR_CONFIG)
    if config is not None:
        cfg.update(config)

    names = [
        "gap_up_limit",
        "gap_down_limit",
        "limit_band_pct",
        "limit_band_pos",
        "hit_up_limit",
        "hit_down_limit",
        "free_float_ratio",
        "circ_float_ratio",
        "free_to_circ_ratio",
        "free_turnover_ratio",
        "free_turnover_spread",
        "volume_ratio_raw",
        "float_mv_ratio",
        "ep",
        "sp",
        "sp_ttm",
        "ep_ttm_gap",
        "dividend_yield",
        "dividend_yield_ttm",
        "has_dividend",
    ]
    names += [f"free_turnover_mean_{int(window)}" for window in cfg["free_turnover_windows"]]
    names += [f"limit_band_pct_mean_{int(window)}" for window in cfg["limit_stat_windows"]]
    names += [f"limit_band_pos_mean_{int(window)}" for window in cfg["limit_stat_windows"]]
    names += [f"gap_up_limit_mean_{int(window)}" for window in cfg["limit_stat_windows"]]
    names += [f"gap_down_limit_mean_{int(window)}" for window in cfg["limit_stat_windows"]]
    names += [f"hit_up_limit_count_{int(window)}" for window in cfg["limit_stat_windows"]]
    names += [f"hit_down_limit_count_{int(window)}" for window in cfg["limit_stat_windows"]]
    names += [f"amplitude_mean_{int(window)}" for window in cfg["amplitude_windows"]]
    names += [f"pct_chg_mean_{int(window)}" for window in cfg["pct_chg_windows"]]
    names += [f"free_float_ratio_change_{int(window)}" for window in cfg["ratio_change_windows"]]
    names += [f"free_to_circ_ratio_change_{int(window)}" for window in cfg["ratio_change_windows"]]
    names += [f"float_mv_ratio_change_{int(window)}" for window in cfg["ratio_change_windows"]]
    names += [f"sp_ttm_change_{int(window)}" for window in cfg["valuation_change_windows"]]
    names += [f"dividend_yield_ttm_change_{int(window)}" for window in cfg["valuation_change_windows"]]
    names += [
        "latest_eps",
        "latest_dt_eps",
        "latest_bps",
        "latest_ocfps",
        "latest_roe",
        "latest_roe_dt",
        "latest_roa",
        "latest_grossprofit_margin",
        "latest_netprofit_margin",
        "latest_debt_to_assets",
        "latest_q_eps",
        "latest_q_dtprofit",
        "latest_q_roe",
        "latest_q_dt_roe",
        "latest_tr_yoy",
        "latest_or_yoy",
        "latest_op_yoy",
        "latest_netprofit_yoy",
        "latest_ocf_yoy",
        "latest_div_cash",
        "latest_div_cash_tax",
        "latest_div_stock",
        "latest_div_bo_rate",
        "latest_div_co_rate",
        "latest_div_base_share",
        "latest_div_cash_yield_proxy",
        "latest_div_stock_ratio",
        "has_stock_dividend",
        "latest_fc_p_change_min",
        "latest_fc_p_change_max",
        "latest_fc_net_profit_min",
        "latest_fc_net_profit_max",
        "latest_fc_last_parent_net",
        "latest_exp_revenue",
        "latest_exp_operate_profit",
        "latest_exp_total_profit",
        "latest_exp_n_income",
        "latest_exp_total_assets",
        "latest_exp_diluted_eps",
        "latest_exp_diluted_roe",
        "latest_exp_yoy_sales",
        "latest_exp_yoy_op",
        "latest_exp_yoy_tp",
        "latest_exp_yoy_dedu_np",
        "latest_exp_yoy_eps",
        "latest_exp_yoy_roe",
        "latest_exp_growth_assets",
        "latest_exp_yoy_assets",
    ]
    zscore_window = int(cfg["zscore_window"])
    names += [
        f"amplitude_zscore_{zscore_window}",
        f"pct_chg_zscore_{zscore_window}",
        f"free_turnover_ratio_zscore_{zscore_window}",
        f"volume_ratio_raw_zscore_{zscore_window}",
    ]
    return names


def get_known_exact_duplicate_feature_groups(
    alpha158_config: dict[str, Any] | None = None,
    lgbm_purified_config: dict[str, Any] | None = None,
    temporal_config: dict[str, Any] | None = None,
) -> list[tuple[str, ...]]:
    alpha_cfg = deepcopy(DEFAULT_ALPHA158_CONFIG)
    if alpha158_config is not None:
        alpha_cfg.update(alpha158_config)
    lgbm_cfg = deepcopy(DEFAULT_LGBM_PURIFIED_CONFIG)
    if lgbm_purified_config is not None:
        lgbm_cfg.update(lgbm_purified_config)
    temporal_cfg = deepcopy(DEFAULT_TEMPORAL_FACTOR_CONFIG)
    if temporal_config is not None:
        temporal_cfg.update(temporal_config)

    alpha_rolling_cfg = alpha_cfg.get("rolling", {})
    alpha_windows = {int(value) for value in alpha_rolling_cfg.get("windows", [5, 10, 20, 30, 60])}
    alpha_include = alpha_rolling_cfg.get("include")
    alpha_exclude = set(alpha_rolling_cfg.get("exclude", []))

    def alpha_uses(op: str) -> bool:
        return op not in alpha_exclude and (alpha_include is None or op in alpha_include)

    temporal_windows = {int(value) for value in temporal_cfg.get("windows", [])}
    temporal_groups = set(temporal_cfg.get("groups", []))

    groups: list[tuple[str, ...]] = []
    if alpha_uses("RSV") and "rsv" in temporal_groups:
        for window in sorted(alpha_windows & temporal_windows):
            groups.append((f"RSV{window}", f"{TEMPORAL_FACTOR_PREFIX}rsv_{window}"))
    if alpha_uses("CORR") and "corr_cv" in temporal_groups:
        for window in sorted(alpha_windows & temporal_windows):
            groups.append((f"CORR{window}", f"{TEMPORAL_FACTOR_PREFIX}corr_cv_{window}"))
    if "ret" in temporal_groups:
        for window in sorted({int(value) for value in lgbm_cfg.get("momentum_windows", [])} & temporal_windows):
            groups.append((f"{ALL_FACTORS_LGBM_PREFIX}ret_{window}", f"{TEMPORAL_FACTOR_PREFIX}ret_{window}"))
    if "ma_gap" in temporal_groups:
        for window in sorted({int(value) for value in lgbm_cfg.get("ma_windows", [])} & temporal_windows):
            groups.append((f"{ALL_FACTORS_LGBM_PREFIX}dist_ma{window}", f"{TEMPORAL_FACTOR_PREFIX}ma_gap_{window}"))
    if "std" in temporal_groups:
        vol_window = int(lgbm_cfg.get("vol_window", 60))
        if vol_window in temporal_windows:
            groups.append((f"{ALL_FACTORS_LGBM_PREFIX}std_60", f"{TEMPORAL_FACTOR_PREFIX}std_{vol_window}"))
    if "amihud" in temporal_groups:
        amihud_window = int(lgbm_cfg.get("amihud_window", 20))
        if amihud_window in temporal_windows:
            groups.append((f"{ALL_FACTORS_LGBM_PREFIX}amihud_20", f"{TEMPORAL_FACTOR_PREFIX}amihud_{amihud_window}"))
    if "turnover_mean" in temporal_groups:
        turnover_window = int(lgbm_cfg.get("turnover_window", 20))
        if turnover_window in temporal_windows:
            groups.append((f"{ALL_FACTORS_LGBM_PREFIX}turnover_20", f"{TEMPORAL_FACTOR_PREFIX}turnover_mean_{turnover_window}"))
    extreme_window = int(lgbm_cfg.get("extreme_window", 20))
    if "high_gap" in temporal_groups and extreme_window in temporal_windows:
        groups.append((f"{ALL_FACTORS_LGBM_PREFIX}dist_high_20", f"{TEMPORAL_FACTOR_PREFIX}high_gap_{extreme_window}"))
    if "low_gap" in temporal_groups and extreme_window in temporal_windows:
        groups.append((f"{ALL_FACTORS_LGBM_PREFIX}dist_low_20", f"{TEMPORAL_FACTOR_PREFIX}low_gap_{extreme_window}"))
    return groups


def get_exact_duplicate_feature_source_map(
    alpha158_config: dict[str, Any] | None = None,
    lgbm_purified_config: dict[str, Any] | None = None,
    temporal_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    source_map: dict[str, str] = {}
    for group in get_known_exact_duplicate_feature_groups(
        alpha158_config=alpha158_config,
        lgbm_purified_config=lgbm_purified_config,
        temporal_config=temporal_config,
    ):
        canonical = str(group[0])
        source_map[canonical] = canonical
        for feature_name in group[1:]:
            source_map[str(feature_name)] = canonical
    return source_map


def deduplicate_exact_feature_names(
    feature_names: list[str],
    *,
    alpha158_config: dict[str, Any] | None = None,
    lgbm_purified_config: dict[str, Any] | None = None,
    temporal_config: dict[str, Any] | None = None,
) -> list[str]:
    source_map = get_exact_duplicate_feature_source_map(
        alpha158_config=alpha158_config,
        lgbm_purified_config=lgbm_purified_config,
        temporal_config=temporal_config,
    )
    deduped: list[str] = []
    seen_sources: set[str] = set()
    for feature_name in feature_names:
        source_name = source_map.get(feature_name, feature_name)
        if source_name in seen_sources:
            continue
        deduped.append(feature_name)
        seen_sources.add(source_name)
    return deduped


def get_all_factor_feature_names(
    alpha158_config: dict[str, Any] | None = None,
    lgbm_purified_config: dict[str, Any] | None = None,
    technical_config: dict[str, Any] | None = None,
    data_source: str | None = None,
    tushare_config: dict[str, Any] | None = None,
) -> list[str]:
    alpha158_names = get_alpha158_feature_config(alpha158_config)[1]
    lgbm_names = [f"{ALL_FACTORS_LGBM_PREFIX}{name}" for name in get_lgbm_purified_feature_names(lgbm_purified_config)]
    temporal_names = [f"{TEMPORAL_FACTOR_PREFIX}{name}" for name in get_temporal_factor_feature_names()]
    technical_names = [f"{TECHNICAL_FACTOR_PREFIX}{name}" for name in get_technical_factor_feature_names(technical_config)]
    feature_names = alpha158_names + lgbm_names + temporal_names + technical_names
    if data_source is not None and normalize_data_source_name(data_source) == "tushare":
        feature_names += [f"{TUSHARE_FACTOR_PREFIX}{name}" for name in get_tushare_factor_feature_names(tushare_config)]
    return deduplicate_exact_feature_names(
        feature_names,
        alpha158_config=alpha158_config,
        lgbm_purified_config=lgbm_purified_config,
    )


def get_full_factor_space_feature_names(data_source: str | None = None) -> list[str]:
    return get_all_factor_feature_names(data_source=data_source)


def validate_default_dimensions() -> dict[str, int]:
    f158, n158 = get_alpha158_feature_config()
    f360, n360 = get_alpha360_feature_config()
    if len(f158) != 158 or len(n158) != 158:
        raise ValueError(f"Alpha158 mismatch: {len(f158)}, {len(n158)}")
    if len(f360) != 360 or len(n360) != 360:
        raise ValueError(f"Alpha360 mismatch: {len(f360)}, {len(n360)}")
    return {"alpha158": 158, "alpha360": 360}
