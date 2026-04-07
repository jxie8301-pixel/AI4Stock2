"""Score transforms, trade selection, and target weights for native backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WEIGHTING = "equal"
SUPPORTED_WEIGHTING_MODES = ("equal", "rank", "score_softmax")
DEFAULT_SCORE_TRANSFORM = "none"
SUPPORTED_SCORE_TRANSFORMS = ("none", "rank_pct", "zscore_clip")


def normalize_weighting_mode(weighting: str | None) -> str:
    mode = str(weighting or DEFAULT_WEIGHTING).strip().lower()
    return mode or DEFAULT_WEIGHTING


def normalize_score_transform(score_transform: str | None) -> str:
    mode = str(score_transform or DEFAULT_SCORE_TRANSFORM).strip().lower()
    return mode or DEFAULT_SCORE_TRANSFORM


def normalize_keep_top_n(keep_top_n: int | None, topk: int) -> int | None:
    if keep_top_n is None:
        return None
    keep_top_n = max(int(keep_top_n), int(topk))
    return keep_top_n


def normalize_min_score(min_score: float | None) -> float | None:
    if min_score is None:
        return None
    return float(min_score)


def transform_scores(
    scores: pd.Series,
    *,
    score_transform: str,
    zscore_clip: float,
) -> pd.Series:
    transformed = pd.to_numeric(scores, errors="coerce").astype(float)
    mode = normalize_score_transform(score_transform)

    if transformed.empty or mode == "none":
        return transformed
    if mode == "rank_pct":
        out = transformed.rank(method="average", pct=True)
        out[transformed.isna()] = np.nan
        return out.astype(float)
    if mode == "zscore_clip":
        finite = transformed.dropna()
        if finite.empty:
            return transformed
        std = float(finite.std(ddof=0))
        if np.isfinite(std) and not np.isclose(std, 0.0):
            out = (transformed - float(finite.mean())) / std
        else:
            out = pd.Series(0.0, index=transformed.index, dtype=float)
            out[transformed.isna()] = np.nan
            return out
        clip_value = max(float(zscore_clip), 0.0)
        if clip_value > 0:
            out = out.clip(lower=-clip_value, upper=clip_value)
        return out.astype(float)
    raise ValueError(
        f"Unsupported score transform: {score_transform}. Supported: {', '.join(SUPPORTED_SCORE_TRANSFORMS)}"
    )


def transform_score_matrix(
    score_matrix: pd.DataFrame,
    *,
    score_transform: str,
    zscore_clip: float,
) -> pd.DataFrame:
    mode = normalize_score_transform(score_transform)
    matrix = score_matrix.apply(pd.to_numeric, errors="coerce").astype(float)
    if matrix.empty or mode == "none":
        return matrix
    if mode == "rank_pct":
        return matrix.rank(axis=1, method="average", pct=True)
    if mode != "zscore_clip":
        raise ValueError(
            f"Unsupported score transform: {score_transform}. Supported: {', '.join(SUPPORTED_SCORE_TRANSFORMS)}"
        )

    values = matrix.to_numpy(dtype=float, copy=False)
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=1).astype(float)
    sums = np.where(finite_mask, values, 0.0).sum(axis=1)
    means = np.divide(sums, counts, out=np.full(values.shape[0], np.nan, dtype=float), where=counts > 0)
    centered = values - means[:, None]
    centered[~finite_mask] = np.nan
    sq_sums = np.where(finite_mask, centered * centered, 0.0).sum(axis=1)
    stds = np.sqrt(np.divide(sq_sums, counts, out=np.full(values.shape[0], np.nan, dtype=float), where=counts > 0))

    out = np.full(values.shape, np.nan, dtype=float)
    zero_std_rows = finite_mask & (~np.isfinite(stds)[:, None] | np.isclose(stds, 0.0)[:, None])
    out[zero_std_rows] = 0.0
    valid_std_rows = np.isfinite(stds) & ~np.isclose(stds, 0.0)
    if valid_std_rows.any():
        out[valid_std_rows] = centered[valid_std_rows] / stds[valid_std_rows, None]

    clip_value = max(float(zscore_clip), 0.0)
    if clip_value > 0:
        out = np.clip(out, -clip_value, clip_value)
        out[~finite_mask] = np.nan
    return pd.DataFrame(out, index=matrix.index, columns=matrix.columns, dtype=float)


def ordered_unique_symbols(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(str(symbol) for symbol in symbols))


def cap_target_weights(weights: pd.Series, max_weight: float | None) -> pd.Series:
    if weights.empty:
        return weights.astype(float)

    total = float(weights.sum())
    if total <= 0:
        return pd.Series(0.0, index=weights.index, dtype=float)

    normalized = weights.astype(float) / total
    if max_weight is None:
        return normalized

    cap = float(max_weight)
    final = pd.Series(0.0, index=normalized.index, dtype=float)
    remaining_idx = normalized.index
    remaining_budget = 1.0

    while len(remaining_idx) > 0 and remaining_budget > 1e-12:
        active = normalized.loc[remaining_idx]
        active_total = float(active.sum())
        if active_total <= 0:
            break
        proposal = active / active_total * remaining_budget
        over_idx = proposal[proposal > cap + 1e-12].index
        if len(over_idx) == 0:
            final.loc[remaining_idx] = proposal
            remaining_budget = 0.0
            break
        final.loc[over_idx] = cap
        remaining_budget = max(remaining_budget - cap * len(over_idx), 0.0)
        remaining_idx = remaining_idx.difference(over_idx, sort=False)

    return final


def cap_group_weights(
    weights: pd.Series,
    *,
    group_labels: pd.Series | None,
    max_group_weight: float | None,
) -> pd.Series:
    if weights.empty:
        return weights.astype(float)
    if max_group_weight is None or group_labels is None:
        return weights.astype(float)

    normalized = weights.astype(float).copy()
    total = float(normalized.sum())
    if total <= 0:
        return pd.Series(0.0, index=normalized.index, dtype=float)
    normalized /= total

    cap = float(max_group_weight)
    final = pd.Series(0.0, index=normalized.index, dtype=float)
    remaining_idx = normalized.index
    remaining_budget = 1.0

    while len(remaining_idx) > 0 and remaining_budget > 1e-12:
        active = normalized.loc[remaining_idx]
        active_total = float(active.sum())
        if active_total <= 0:
            break
        active = active / active_total * remaining_budget
        active_groups = group_labels.reindex(active.index)
        group_totals = active.groupby(active_groups, sort=False).sum()
        over_groups = group_totals[group_totals > cap + 1e-12]
        if over_groups.empty:
            final.loc[active.index] = active
            break

        consumed_budget = 0.0
        locked_members: list[str] = []
        for group_name in over_groups.index.tolist():
            member_idx = active_groups[active_groups == group_name].index
            member_weights = active.loc[member_idx]
            member_total = float(member_weights.sum())
            if member_total <= 0:
                locked_members.extend(member_idx.tolist())
                continue
            scaled = member_weights / member_total * cap
            final.loc[member_idx] = scaled
            consumed_budget += float(scaled.sum())
            locked_members.extend(member_idx.tolist())

        remaining_budget = max(remaining_budget - consumed_budget, 0.0)
        remaining_idx = remaining_idx.difference(pd.Index(locked_members, dtype=object), sort=False)

    return final


def compute_target_weights(
    transformed_scores: pd.Series,
    target_holdings: list[str],
    *,
    weighting: str,
    max_weight: float | None,
    group_labels: pd.Series | None = None,
    max_group_weight: float | None = None,
) -> pd.Series:
    target_index = pd.Index(ordered_unique_symbols(target_holdings), dtype=object)
    if target_index.empty:
        return pd.Series(dtype=float)

    mode = normalize_weighting_mode(weighting)
    target_scores = pd.to_numeric(transformed_scores.reindex(target_index), errors="coerce").astype(float)
    if target_scores.notna().any():
        fill_value = float(target_scores.min(skipna=True)) - 1.0
        target_scores = target_scores.fillna(fill_value)
    else:
        target_scores = pd.Series(0.0, index=target_index, dtype=float)

    if mode == "equal":
        raw = pd.Series(1.0, index=target_index, dtype=float)
    elif mode == "rank":
        ranks = target_scores.rank(ascending=False, method="average")
        raw = float(len(target_scores)) - ranks + 1.0
    elif mode == "score_softmax":
        std = float(target_scores.std(ddof=0))
        if np.isfinite(std) and not np.isclose(std, 0.0):
            scaled = (target_scores - target_scores.mean()) / std
        else:
            scaled = pd.Series(0.0, index=target_index, dtype=float)
        scaled = scaled.clip(lower=-20.0, upper=20.0)
        raw = pd.Series(np.exp((scaled - scaled.max()).to_numpy()), index=target_index, dtype=float)
    else:
        raise ValueError(
            f"Unsupported weighting mode: {weighting}. Supported: {', '.join(SUPPORTED_WEIGHTING_MODES)}"
        )

    capped = cap_target_weights(raw, max_weight)
    return cap_group_weights(
        capped,
        group_labels=group_labels.reindex(target_index) if group_labels is not None else None,
        max_group_weight=max_group_weight,
    )


def select_topk_dropout_trades(
    transformed_scores: pd.Series,
    current_holdings: list[str],
    topk: int,
    n_drop: int,
    locked_holdings: set[str] | None = None,
    keep_top_n: int | None = None,
    min_score: float | None = None,
) -> tuple[list[str], list[str]]:
    ranked_scores = pd.to_numeric(transformed_scores, errors="coerce").dropna().sort_values(ascending=False)
    min_score_value = normalize_min_score(min_score)
    eligible_scores = ranked_scores if min_score_value is None else ranked_scores[ranked_scores > min_score_value]
    locked_set = set() if locked_holdings is None else set(locked_holdings)

    if eligible_scores.empty:
        if min_score_value is None:
            return [], []
        return [stock for stock in current_holdings if stock not in locked_set], []

    current_index = pd.Index(current_holdings, dtype=object)
    ranked_current = ranked_scores.reindex(current_index).sort_values(ascending=False, na_position="last").index
    keep_top_n_value = normalize_keep_top_n(keep_top_n, topk)
    eligible_ranks = {stock: rank for rank, stock in enumerate(eligible_scores.index.tolist(), start=1)}
    forced_sell = [stock for stock in current_holdings if stock not in locked_set and stock not in eligible_ranks]

    buffer_protected = set()
    if keep_top_n_value is not None:
        buffer_protected = {
            stock
            for stock in current_holdings
            if stock not in locked_set
            and stock not in forced_sell
            and eligible_ranks.get(stock, keep_top_n_value + 1) <= keep_top_n_value
        }

    sellable_current = pd.Index(
        [
            stock
            for stock in ranked_current
            if stock not in locked_set and stock not in forced_sell and stock not in buffer_protected
        ],
        dtype=object,
    )

    n_drop = max(0, int(n_drop))
    topk = max(0, int(topk))
    candidate_count = len(forced_sell) + n_drop + max(topk - len(ranked_current), 0)
    today = eligible_scores[~eligible_scores.index.isin(ranked_current)].index[:candidate_count]

    comb = eligible_scores.reindex(ranked_current.union(today)).sort_values(ascending=False, na_position="last").index
    sellable_comb = pd.Index(
        [stock for stock in comb if stock not in locked_set and stock not in forced_sell and stock not in buffer_protected],
        dtype=object,
    )
    effective_drop = min(n_drop, len(sellable_current))
    drop_set = set(sellable_comb[-effective_drop:]) if effective_drop > 0 else set()

    sell = forced_sell + [stock for stock in sellable_current if stock in drop_set]
    buy_count = len(sell) + max(topk - len(ranked_current), 0)
    buy = list(today[:buy_count])
    return sell, buy


def trade_cost(trade_value: float, rate: float, min_cost: float) -> float:
    if trade_value <= 0:
        return 0.0
    return max(trade_value * rate, min_cost)


def max_affordable_trade_value(cash: float, rate: float, min_cost: float) -> float:
    if cash <= 0:
        return 0.0
    if rate <= 0:
        return max(cash - min_cost, 0.0) if min_cost > 0 else cash

    proportional_limit = cash / (1.0 + rate)
    threshold = min_cost / rate
    if proportional_limit >= threshold:
        return max(proportional_limit, 0.0)
    return max(cash - min_cost, 0.0)


def snapshot_holdings(holdings: dict[str, float]) -> dict[str, float]:
    return {stock: float(value) for stock, value in sorted(holdings.items())}


def compute_signal_strength_value(
    transformed_scores: pd.Series,
    *,
    topk: int,
    min_score: float | None,
    signal_metric: str,
) -> float:
    ranked = pd.to_numeric(transformed_scores, errors="coerce").dropna().sort_values(ascending=False)
    min_score_value = normalize_min_score(min_score)
    if min_score_value is not None:
        ranked = ranked[ranked > min_score_value]
    if ranked.empty:
        return float("nan")
    top_values = ranked.iloc[: max(int(topk), 1)]
    if signal_metric == "top1":
        return float(top_values.iloc[0])
    if signal_metric == "topk_sum":
        return float(top_values.sum())
    return float(top_values.mean())
