use std::cmp::Ordering;

pub(crate) const OUT_COLS: usize = 30;
const OUT_GROSS_RETURN: usize = 0;
const OUT_NET_RETURN: usize = 1;
const OUT_TURNOVER: usize = 2;
const OUT_COST: usize = 3;
const OUT_BENCH: usize = 4;
const OUT_BUY_COUNT: usize = 5;
const OUT_SELL_COUNT: usize = 6;
const OUT_HOLDINGS: usize = 7;
const OUT_FROZEN_HOLDINGS: usize = 8;
const OUT_ACCOUNT_VALUE: usize = 9;
const OUT_RISK_DEGREE: usize = 10;
const OUT_EFFECTIVE_N_DROP: usize = 11;
const OUT_RISK_SIGNAL: usize = 12;
const OUT_DESTICKY_ACTIVE: usize = 13;
const OUT_INTRAPERIOD_EXIT_COUNT: usize = 14;
const OUT_INTRAPERIOD_SCORE_CANDIDATE_COUNT: usize = 15;
const OUT_INTRAPERIOD_PRICE_CONFIRM_REQUIRED_COUNT: usize = 16;
const OUT_INTRAPERIOD_PRICE_CONFIRM_BLOCKED_COUNT: usize = 17;
const OUT_INTRAPERIOD_PRICE_CONFIRM_BYPASSED_REMAINING_STEPS_COUNT: usize = 18;
const OUT_INTRAPERIOD_PRICE_CONFIRM_BYPASSED_FORCE_EXIT_COUNT: usize = 19;
const OUT_INTRAPERIOD_REMAINING_STEPS: usize = 20;
const OUT_INTRAPERIOD_SIGNAL_MEAN: usize = 21;
const OUT_INTRAPERIOD_SIGNAL_MIN: usize = 22;
const OUT_INTRAPERIOD_RESIDUAL_MEAN: usize = 23;
const OUT_INTRAPERIOD_RESIDUAL_MIN: usize = 24;
const OUT_INTRAPERIOD_RESIDUAL_MAX: usize = 25;
const OUT_INTRAPERIOD_SAVED_RETURN: usize = 26;
const OUT_INTRAPERIOD_MISSED_RETURN: usize = 27;
const OUT_INTRAPERIOD_BENEFICIAL_COUNT: usize = 28;
const OUT_INTRAPERIOD_HARMFUL_COUNT: usize = 29;
const EPS: f64 = 1e-12;
const WEIGHT_EQUAL: u8 = 0;
const WEIGHT_RANK: u8 = 1;
const WEIGHT_SCORE_SOFTMAX: u8 = 2;
const SCORE_TRANSFORM_DISABLED: u8 = 255;
const SCORE_TRANSFORM_STRATEGY: u8 = 3;
const SCORE_TRANSFORM_NONE: u8 = 0;
const SCORE_TRANSFORM_RANK_PCT: u8 = 1;
const SCORE_TRANSFORM_ZSCORE_CLIP: u8 = 2;
const INTRAPERIOD_EXIT_SCORE_THRESHOLD: u8 = 1;
const INTRAPERIOD_EXIT_EXPECTED_RETURN: u8 = 2;

#[inline]
fn finite(value: f64) -> bool {
    value.is_finite()
}

#[inline]
fn almost_zero(value: f64) -> bool {
    value.abs() <= 1e-8
}

#[inline]
fn trade_cost(trade_value: f64, rate: f64, min_cost: f64) -> f64 {
    if trade_value <= 0.0 {
        0.0
    } else {
        (trade_value * rate).max(min_cost)
    }
}

#[inline]
fn max_affordable_trade_value(cash: f64, rate: f64, min_cost: f64) -> f64 {
    if cash <= 0.0 {
        return 0.0;
    }
    if rate <= 0.0 {
        return if min_cost > 0.0 {
            (cash - min_cost).max(0.0)
        } else {
            cash
        };
    }
    let proportional_limit = cash / (1.0 + rate);
    let threshold = min_cost / rate;
    if proportional_limit >= threshold {
        proportional_limit.max(0.0)
    } else {
        (cash - min_cost).max(0.0)
    }
}

#[inline]
fn cmp_desc_with_index(scores: &[f64], left: usize, right: usize) -> Ordering {
    match scores[right]
        .partial_cmp(&scores[left])
        .unwrap_or(Ordering::Equal)
    {
        Ordering::Equal => left.cmp(&right),
        ordering => ordering,
    }
}

fn ranked_finite_indices(scores: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len())
        .filter(|&idx| finite(scores[idx]))
        .collect();
    indices.sort_by(|&left, &right| cmp_desc_with_index(scores, left, right));
    indices
}

fn current_holdings_order(holdings: &[f64], order: &[usize]) -> Vec<usize> {
    let mut current = Vec::new();
    for &idx in order {
        if holdings[idx] > EPS {
            current.push(idx);
        }
    }
    current
}

fn ranked_current(current: &[usize], rank_positions: &[usize], missing_rank: usize) -> Vec<usize> {
    let mut ranked = current.to_vec();
    ranked.sort_by_key(|&idx| rank_positions.get(idx).copied().unwrap_or(missing_rank));
    ranked
}

fn select_topk_dropout_trades(
    scores: &[f64],
    current: &[usize],
    locked: &[bool],
    topk: usize,
    n_drop: usize,
    keep_top_n: Option<usize>,
    min_score: Option<f64>,
) -> (Vec<usize>, Vec<usize>) {
    let ranked_scores = ranked_finite_indices(scores);
    let eligible: Vec<usize> = match min_score {
        Some(threshold) => ranked_scores
            .iter()
            .copied()
            .filter(|&idx| scores[idx] > threshold)
            .collect(),
        None => ranked_scores.clone(),
    };

    if eligible.is_empty() {
        if min_score.is_none() {
            return (Vec::new(), Vec::new());
        }
        let sell = current
            .iter()
            .copied()
            .filter(|&idx| !locked[idx])
            .collect();
        return (sell, Vec::new());
    }

    let missing_rank = scores.len() + 1;
    let mut rank_positions = vec![missing_rank; scores.len()];
    for (rank, &idx) in ranked_scores.iter().enumerate() {
        rank_positions[idx] = rank;
    }

    let mut current_set = vec![false; scores.len()];
    for &idx in current {
        current_set[idx] = true;
    }

    let mut eligible_rank = vec![0usize; scores.len()];
    for (rank, &idx) in eligible.iter().enumerate() {
        eligible_rank[idx] = rank + 1;
    }

    let ranked_current = ranked_current(current, &rank_positions, missing_rank);
    let mut forced_sell = Vec::new();
    let mut forced_sell_set = vec![false; scores.len()];
    for &idx in current {
        if !locked[idx] && eligible_rank[idx] == 0 {
            forced_sell.push(idx);
            forced_sell_set[idx] = true;
        }
    }

    let mut buffer_protected = vec![false; scores.len()];
    if let Some(limit) = keep_top_n {
        for &idx in current {
            let rank = eligible_rank[idx];
            if !locked[idx] && !forced_sell_set[idx] && rank > 0 && rank <= limit {
                buffer_protected[idx] = true;
            }
        }
    }

    let sellable_current: Vec<usize> = ranked_current
        .iter()
        .copied()
        .filter(|&idx| !locked[idx] && !forced_sell_set[idx] && !buffer_protected[idx])
        .collect();

    let current_shortfall = topk.saturating_sub(ranked_current.len());
    let candidate_count = forced_sell.len() + n_drop + current_shortfall;
    let mut today = Vec::new();
    for &idx in &eligible {
        if !current_set[idx] {
            today.push(idx);
            if today.len() >= candidate_count {
                break;
            }
        }
    }

    let mut combined_set = vec![false; scores.len()];
    let mut combined = Vec::new();
    for &idx in ranked_current.iter().chain(today.iter()) {
        if !combined_set[idx] {
            combined.push(idx);
            combined_set[idx] = true;
        }
    }
    combined.sort_by_key(|&idx| rank_positions.get(idx).copied().unwrap_or(missing_rank));

    let sellable_comb: Vec<usize> = combined
        .into_iter()
        .filter(|&idx| !locked[idx] && !forced_sell_set[idx] && !buffer_protected[idx])
        .collect();
    let effective_drop = n_drop.min(sellable_current.len());
    let mut drop_set = vec![false; scores.len()];
    if effective_drop > 0 {
        for &idx in sellable_comb.iter().rev().take(effective_drop) {
            drop_set[idx] = true;
        }
    }

    let mut sell = forced_sell;
    for &idx in &sellable_current {
        if drop_set[idx] {
            sell.push(idx);
        }
    }

    let buy_count = sell.len() + current_shortfall;
    let buy = today.into_iter().take(buy_count).collect();
    (sell, buy)
}

fn compact_order(order: &mut Vec<usize>, holdings: &[f64]) {
    order.retain(|&idx| holdings[idx] > EPS);
}

fn cap_weights(raw_weights: &[f64], max_weight: Option<f64>) -> Vec<f64> {
    if raw_weights.is_empty() {
        return Vec::new();
    }
    let total: f64 = raw_weights.iter().sum();
    if total <= 0.0 {
        return vec![0.0; raw_weights.len()];
    }

    let normalized: Vec<f64> = raw_weights.iter().map(|value| value / total).collect();
    let Some(cap) = max_weight else {
        return normalized;
    };

    let mut final_weights = vec![0.0; raw_weights.len()];
    let mut remaining: Vec<usize> = (0..raw_weights.len()).collect();
    let mut remaining_budget = 1.0;

    while !remaining.is_empty() && remaining_budget > EPS {
        let active_total: f64 = remaining.iter().map(|&idx| normalized[idx]).sum();
        if active_total <= 0.0 {
            break;
        }

        let mut over = Vec::new();
        let mut proposal = Vec::with_capacity(remaining.len());
        for &idx in &remaining {
            let value = normalized[idx] / active_total * remaining_budget;
            if value > cap + EPS {
                over.push(idx);
            }
            proposal.push((idx, value));
        }

        if over.is_empty() {
            for (idx, value) in proposal {
                final_weights[idx] = value;
            }
            break;
        }

        let mut over_set = vec![false; raw_weights.len()];
        for &idx in &over {
            final_weights[idx] = cap;
            over_set[idx] = true;
        }
        remaining_budget = (remaining_budget - cap * over.len() as f64).max(0.0);
        remaining.retain(|&idx| !over_set[idx]);
    }

    final_weights
}

fn cap_group_weights(
    weights: &[f64],
    target_holdings: &[usize],
    group_ids: &[i32],
    max_group_weight: Option<f64>,
) -> Vec<f64> {
    if weights.is_empty() {
        return Vec::new();
    }
    let Some(cap) = max_group_weight else {
        return weights.to_vec();
    };
    if group_ids.is_empty() {
        return weights.to_vec();
    }

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return vec![0.0; weights.len()];
    }

    let normalized: Vec<f64> = weights.iter().map(|value| value / total).collect();
    let mut final_weights = vec![0.0; weights.len()];
    let mut remaining: Vec<usize> = (0..weights.len()).collect();
    let mut remaining_budget = 1.0;

    while !remaining.is_empty() && remaining_budget > EPS {
        let active_total: f64 = remaining.iter().map(|&idx| normalized[idx]).sum();
        if active_total <= 0.0 {
            break;
        }

        let mut active_weights = vec![0.0; weights.len()];
        for &idx in &remaining {
            active_weights[idx] = normalized[idx] / active_total * remaining_budget;
        }

        let mut group_order: Vec<i32> = Vec::new();
        let mut group_totals: Vec<(i32, f64)> = Vec::new();
        for &idx in &remaining {
            let stock = target_holdings[idx];
            let group = group_ids.get(stock).copied().unwrap_or(stock as i32);
            if let Some(pos) = group_order.iter().position(|&value| value == group) {
                group_totals[pos].1 += active_weights[idx];
            } else {
                group_order.push(group);
                group_totals.push((group, active_weights[idx]));
            }
        }

        let over_groups: Vec<i32> = group_totals
            .iter()
            .filter_map(|(group, total)| {
                if *total > cap + EPS {
                    Some(*group)
                } else {
                    None
                }
            })
            .collect();
        if over_groups.is_empty() {
            for &idx in &remaining {
                final_weights[idx] = active_weights[idx];
            }
            break;
        }

        let mut locked_members = vec![false; weights.len()];
        let mut consumed_budget = 0.0;
        for group in over_groups {
            let members: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&idx| {
                    let stock = target_holdings[idx];
                    group_ids.get(stock).copied().unwrap_or(stock as i32) == group
                })
                .collect();
            let member_total: f64 = members.iter().map(|&idx| active_weights[idx]).sum();
            if member_total <= 0.0 {
                for idx in members {
                    locked_members[idx] = true;
                }
                continue;
            }
            for idx in members {
                let scaled = active_weights[idx] / member_total * cap;
                final_weights[idx] = scaled;
                consumed_budget += scaled;
                locked_members[idx] = true;
            }
        }

        remaining_budget = (remaining_budget - consumed_budget).max(0.0);
        remaining.retain(|&idx| !locked_members[idx]);
    }

    final_weights
}

fn rank_weight_raw(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| cmp_desc_with_index(scores, left, right));

    let mut ranks = vec![0.0; n];
    let mut pos = 0usize;
    while pos < n {
        let mut end = pos + 1;
        while end < n && scores[order[end]] == scores[order[pos]] {
            end += 1;
        }
        let avg_rank = (pos + 1 + end) as f64 / 2.0;
        for idx in pos..end {
            ranks[order[idx]] = avg_rank;
        }
        pos = end;
    }

    ranks
        .into_iter()
        .map(|rank| n as f64 - rank + 1.0)
        .collect()
}

fn softmax_weight_raw(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / scores.len() as f64;
    let std = variance.sqrt();

    let mut scaled = Vec::with_capacity(scores.len());
    if finite(std) && !almost_zero(std) {
        for &score in scores {
            scaled.push(((score - mean) / std).clamp(-20.0, 20.0));
        }
    } else {
        scaled.resize(scores.len(), 0.0);
    }
    let max_scaled = scaled
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));
    scaled
        .into_iter()
        .map(|value| (value - max_scaled).exp())
        .collect()
}

fn rank_pct_transform_row(input: &[f64], out: &mut [f64]) {
    out.fill(f64::NAN);
    let mut order: Vec<usize> = (0..input.len()).filter(|&idx| finite(input[idx])).collect();
    let n = order.len();
    if n == 0 {
        return;
    }

    order.sort_by(|&left, &right| {
        match input[left]
            .partial_cmp(&input[right])
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => left.cmp(&right),
            ordering => ordering,
        }
    });

    let mut pos = 0usize;
    while pos < n {
        let mut end = pos + 1;
        while end < n && input[order[end]] == input[order[pos]] {
            end += 1;
        }
        let avg_rank = (pos + 1 + end) as f64 / 2.0;
        let pct_rank = avg_rank / n as f64;
        for idx in pos..end {
            out[order[idx]] = pct_rank;
        }
        pos = end;
    }
}

fn zscore_clip_transform_row(input: &[f64], clip: f64, out: &mut [f64]) {
    out.fill(f64::NAN);
    let mut count = 0usize;
    let mut sum = 0.0;
    for &value in input {
        if finite(value) {
            count += 1;
            sum += value;
        }
    }
    if count == 0 {
        return;
    }

    let mean = sum / count as f64;
    let mut sq_sum = 0.0;
    for &value in input {
        if finite(value) {
            let centered = value - mean;
            sq_sum += centered * centered;
        }
    }
    let std = (sq_sum / count as f64).sqrt();
    if !finite(std) || almost_zero(std) {
        for (idx, &value) in input.iter().enumerate() {
            if finite(value) {
                out[idx] = 0.0;
            }
        }
        return;
    }

    let clip_value = clip.max(0.0);
    for (idx, &value) in input.iter().enumerate() {
        if finite(value) {
            let mut transformed = (value - mean) / std;
            if clip_value > 0.0 {
                transformed = transformed.clamp(-clip_value, clip_value);
            }
            out[idx] = transformed;
        }
    }
}

fn transform_score_row(
    input: &[f64],
    mode: u8,
    zscore_clip: f64,
    scratch: &mut Vec<f64>,
) -> Result<(), String> {
    scratch.resize(input.len(), f64::NAN);
    match mode {
        SCORE_TRANSFORM_NONE => scratch.copy_from_slice(input),
        SCORE_TRANSFORM_RANK_PCT => rank_pct_transform_row(input, scratch),
        SCORE_TRANSFORM_ZSCORE_CLIP => zscore_clip_transform_row(input, zscore_clip, scratch),
        _ => return Err(format!("unsupported Rust score transform code: {mode}")),
    }
    Ok(())
}

fn prepare_exit_score_row(
    raw_scores: &[f64],
    strategy_scores: &[f64],
    mode: u8,
    zscore_clip: f64,
    scratch: &mut Vec<f64>,
) -> Result<(), String> {
    match mode {
        SCORE_TRANSFORM_DISABLED => {
            scratch.clear();
            Ok(())
        }
        SCORE_TRANSFORM_STRATEGY => {
            scratch.resize(strategy_scores.len(), f64::NAN);
            scratch.copy_from_slice(strategy_scores);
            Ok(())
        }
        SCORE_TRANSFORM_NONE | SCORE_TRANSFORM_RANK_PCT | SCORE_TRANSFORM_ZSCORE_CLIP => {
            transform_score_row(raw_scores, mode, zscore_clip, scratch)
        }
        _ => Err(format!(
            "unsupported Rust intraperiod score transform code: {mode}"
        )),
    }
}

fn residual_return(
    labels: &[f64],
    pos: usize,
    steps: usize,
    stock: usize,
    n_instruments: usize,
) -> Option<f64> {
    if steps == 0 {
        return None;
    }
    let mut product = 1.0;
    for offset in 0..steps {
        let idx = (pos + offset) * n_instruments + stock;
        let value = labels[idx];
        if !finite(value) {
            return None;
        }
        product *= 1.0 + value;
    }
    Some(product - 1.0)
}

#[inline]
fn intraperiod_remaining_steps_for_pos(pos: usize, rebalance_freq: usize, n_dates: usize) -> usize {
    if pos.is_multiple_of(rebalance_freq) {
        0
    } else {
        (((pos / rebalance_freq) + 1) * rebalance_freq)
            .min(n_dates)
            .saturating_sub(pos)
    }
}

#[inline]
fn isclose_like_numpy(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-8 + 1e-5 * right.abs()
}

fn searchsorted_right(sorted_values: &[f64], value: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted_values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if value < sorted_values[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

#[derive(Default)]
struct ExpectedReturnHistory {
    pairs: Vec<(f64, f64)>,
    return_sum: f64,
}

impl ExpectedReturnHistory {
    fn append_from_row(
        &mut self,
        scores: &[f64],
        labels: &[f64],
        pos: usize,
        steps: usize,
        n_instruments: usize,
    ) {
        if steps == 0 {
            return;
        }

        let mut chunk = Vec::new();
        for (stock, &score) in scores.iter().enumerate() {
            if !finite(score) {
                continue;
            }
            if let Some(return_value) = residual_return(labels, pos, steps, stock, n_instruments) {
                chunk.push((score, return_value));
            }
        }
        if chunk.is_empty() {
            return;
        }

        self.return_sum += chunk
            .iter()
            .map(|(_, return_value)| *return_value)
            .sum::<f64>();
        chunk.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));

        if self.pairs.is_empty() {
            self.pairs = chunk;
            return;
        }

        let existing = std::mem::take(&mut self.pairs);
        let mut merged = Vec::with_capacity(existing.len() + chunk.len());
        let mut left = 0usize;
        let mut right = 0usize;
        while left < existing.len() && right < chunk.len() {
            if existing[left].0 <= chunk[right].0 {
                merged.push(existing[left]);
                left += 1;
            } else {
                merged.push(chunk[right]);
                right += 1;
            }
        }
        merged.extend_from_slice(&existing[left..]);
        merged.extend_from_slice(&chunk[right..]);
        self.pairs = merged;
    }

    fn estimate(
        &self,
        current_scores: &[f64],
        n_bins: usize,
        min_history: usize,
        out: &mut Vec<f64>,
    ) {
        out.resize(current_scores.len(), f64::NAN);
        out.fill(f64::NAN);
        let history_len = self.pairs.len();
        if history_len == 0 || history_len < min_history {
            return;
        }

        let global_mean = self.return_sum / history_len as f64;
        let bin_count = n_bins.max(1).min(history_len);
        if bin_count == 1 {
            for (idx, &score) in current_scores.iter().enumerate() {
                if !score.is_nan() {
                    out[idx] = global_mean;
                }
            }
            return;
        }

        let mut edges = Vec::with_capacity(bin_count + 1);
        let quantile_step = 1.0 / bin_count as f64;
        for idx in 0..=bin_count {
            let quantile = if idx == bin_count {
                1.0
            } else {
                idx as f64 * quantile_step
            };
            edges.push(self.quantile(quantile));
        }
        if !edges.iter().all(|value| finite(*value))
            || isclose_like_numpy(edges[0], *edges.last().unwrap_or(&edges[0]))
        {
            for (idx, &score) in current_scores.iter().enumerate() {
                if !score.is_nan() {
                    out[idx] = global_mean;
                }
            }
            return;
        }

        let inner_edges = &edges[1..bin_count];
        let mut bucket_counts = vec![0usize; bin_count];
        let mut bucket_sums = vec![0.0_f64; bin_count];
        let mut bucket = 0usize;
        for &(score, return_value) in &self.pairs {
            while bucket < inner_edges.len() && score >= inner_edges[bucket] {
                bucket += 1;
            }
            bucket_counts[bucket] += 1;
            bucket_sums[bucket] += return_value;
        }

        let mut bucket_means = vec![global_mean; bin_count];
        for idx in 0..bin_count {
            if bucket_counts[idx] > 0 {
                bucket_means[idx] = bucket_sums[idx] / bucket_counts[idx] as f64;
            }
        }

        for (idx, &score) in current_scores.iter().enumerate() {
            if finite(score) {
                out[idx] = bucket_means[searchsorted_right(inner_edges, score)];
            }
        }
    }

    fn quantile(&self, q: f64) -> f64 {
        let n = self.pairs.len();
        if n == 1 {
            return self.pairs[0].0;
        }
        let pos = q * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        if lower == upper {
            self.pairs[lower].0
        } else {
            let weight = pos - lower as f64;
            self.pairs[lower].0 + (self.pairs[upper].0 - self.pairs[lower].0) * weight
        }
    }
}

fn compute_target_weights(
    scores: &[f64],
    target_holdings: &[usize],
    weighting_mode: u8,
    max_weight: Option<f64>,
    group_ids: &[i32],
    max_group_weight: Option<f64>,
) -> Result<Vec<f64>, String> {
    if target_holdings.is_empty() {
        return Ok(Vec::new());
    }

    let mut target_scores: Vec<f64> = target_holdings.iter().map(|&idx| scores[idx]).collect();
    let finite_values: Vec<f64> = target_scores
        .iter()
        .copied()
        .filter(|value| finite(*value))
        .collect();
    if finite_values.is_empty() {
        target_scores.fill(0.0);
    } else {
        let fill_value = finite_values
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, value| acc.min(value))
            - 1.0;
        for score in &mut target_scores {
            if !finite(*score) {
                *score = fill_value;
            }
        }
    }

    let raw = match weighting_mode {
        WEIGHT_EQUAL => vec![1.0; target_holdings.len()],
        WEIGHT_RANK => rank_weight_raw(&target_scores),
        WEIGHT_SCORE_SOFTMAX => softmax_weight_raw(&target_scores),
        _ => {
            return Err(format!(
                "unsupported Rust backtest weighting mode code: {weighting_mode}"
            ))
        }
    };

    let capped = cap_weights(&raw, max_weight);
    Ok(cap_group_weights(
        &capped,
        target_holdings,
        group_ids,
        max_group_weight,
    ))
}

pub(crate) struct CoreInputs<'a> {
    pub(crate) scores: &'a [f64],
    pub(crate) labels: &'a [f64],
    pub(crate) bench: &'a [f64],
    pub(crate) group_ids: &'a [i32],
    pub(crate) risk_values: &'a [f64],
    pub(crate) risk_signal_values: &'a [f64],
    pub(crate) price_confirm: &'a [u8],
    pub(crate) intraperiod_scores: &'a [f64],
    pub(crate) n_dates: usize,
    pub(crate) n_instruments: usize,
}

impl CoreInputs<'_> {
    fn validate(&self, out_len: usize) -> Result<(), String> {
        if self.n_dates == 0 || self.n_instruments == 0 {
            return Err("scores must be non-empty".to_string());
        }
        if self.scores.len() != self.n_dates * self.n_instruments
            || self.labels.len() != self.scores.len()
        {
            return Err("score and label arrays must have matching 2D shapes".to_string());
        }
        if !self.bench.is_empty() && self.bench.len() != self.n_dates {
            return Err("bench must be empty or have one value per date".to_string());
        }
        if !self.group_ids.is_empty() && self.group_ids.len() != self.n_instruments {
            return Err("group_ids must be empty or have one value per instrument".to_string());
        }
        if !self.risk_values.is_empty() && self.risk_values.len() != self.n_dates {
            return Err("risk_values must be empty or have one value per date".to_string());
        }
        if !self.risk_signal_values.is_empty() && self.risk_signal_values.len() != self.n_dates {
            return Err("risk_signal_values must be empty or have one value per date".to_string());
        }
        if !self.price_confirm.is_empty()
            && self.price_confirm.len() != self.n_dates * self.n_instruments
        {
            return Err("price_confirm must be empty or have the same shape as scores".to_string());
        }
        if !self.intraperiod_scores.is_empty()
            && self.intraperiod_scores.len() != self.n_dates * self.n_instruments
        {
            return Err(
                "intraperiod_scores must be empty or have the same shape as scores".to_string(),
            );
        }
        if out_len != self.n_dates * OUT_COLS {
            return Err(format!("out must have shape (n_dates, {OUT_COLS})"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BacktestParams {
    pub(crate) topk: usize,
    pub(crate) n_drop: usize,
    pub(crate) rebalance_freq: usize,
    pub(crate) account: f64,
    pub(crate) default_risk_degree: f64,
    pub(crate) open_rate: f64,
    pub(crate) close_rate: f64,
    pub(crate) min_cost: f64,
    pub(crate) weighting_mode: u8,
    pub(crate) score_transform_mode: u8,
    pub(crate) zscore_clip: f64,
    pub(crate) intraperiod_exit_mode: u8,
    pub(crate) intraperiod_score_transform_mode: u8,
    pub(crate) intraperiod_exit_threshold: Option<f64>,
    pub(crate) intraperiod_expected_return_n_bins: usize,
    pub(crate) intraperiod_expected_return_min_history: usize,
    pub(crate) price_confirm_min_remaining_steps: usize,
    pub(crate) price_confirm_force_exit_threshold: Option<f64>,
    pub(crate) max_weight: Option<f64>,
    pub(crate) max_group_weight: Option<f64>,
    pub(crate) keep_top_n: Option<usize>,
    pub(crate) min_score: Option<f64>,
    pub(crate) desticky_threshold: Option<f64>,
    pub(crate) desticky_n_drop: Option<usize>,
}

impl BacktestParams {
    fn validate(&self) -> Result<(), String> {
        if self.topk == 0 || self.rebalance_freq == 0 {
            return Err("topk and rebalance_freq must be positive".to_string());
        }
        if self.intraperiod_exit_threshold.is_some()
            && self.intraperiod_exit_mode != INTRAPERIOD_EXIT_SCORE_THRESHOLD
            && self.intraperiod_exit_mode != INTRAPERIOD_EXIT_EXPECTED_RETURN
        {
            return Err(format!(
                "unsupported Rust intraperiod exit mode code: {}",
                self.intraperiod_exit_mode
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct TradeStats {
    cost: f64,
    buy_value: f64,
    sell_value: f64,
    buy_count: usize,
    sell_count: usize,
}

impl TradeStats {
    fn record_buy(&mut self, trade_value: f64, cost: f64) {
        self.buy_value += trade_value;
        self.cost += cost;
        self.buy_count += 1;
    }

    fn record_sell(&mut self, trade_value: f64, cost: f64) {
        self.sell_value += trade_value;
        self.cost += cost;
        self.sell_count += 1;
    }
}

pub(crate) struct IntraperiodExitStats {
    pub(crate) exit_count: usize,
    pub(crate) score_candidate_count: usize,
    pub(crate) price_confirm_required_count: usize,
    pub(crate) price_confirm_blocked_count: usize,
    pub(crate) price_confirm_bypassed_remaining_steps_count: usize,
    pub(crate) price_confirm_bypassed_force_exit_count: usize,
    pub(crate) remaining_steps: usize,
    signal_sum: f64,
    pub(crate) signal_min: f64,
    signal_count: usize,
    residual_sum: f64,
    pub(crate) residual_min: f64,
    pub(crate) residual_max: f64,
    residual_count: usize,
    pub(crate) saved_return: f64,
    pub(crate) missed_return: f64,
    pub(crate) beneficial_count: usize,
    pub(crate) harmful_count: usize,
}

pub(crate) struct TraceExitEvent {
    pub(crate) stock: usize,
    pub(crate) score_value: f64,
    pub(crate) residual_return_if_held: f64,
    pub(crate) position_value: f64,
    pub(crate) saved_return_contribution: f64,
    pub(crate) missed_return_contribution: f64,
    pub(crate) remaining_steps: usize,
    pub(crate) price_confirm_required: bool,
    pub(crate) price_confirm_passed: bool,
    pub(crate) price_confirm_bypass_reason: u8,
}

pub(crate) struct TraceRecord {
    pub(crate) pos: usize,
    pub(crate) start_value: f64,
    pub(crate) end_value: f64,
    pub(crate) cash_before: f64,
    pub(crate) cash_after: f64,
    pub(crate) holdings_before: Vec<(usize, f64)>,
    pub(crate) holdings_after: Vec<(usize, f64)>,
    pub(crate) locked_holdings: Vec<usize>,
    pub(crate) sell_list: Vec<usize>,
    pub(crate) buy_list: Vec<usize>,
    pub(crate) trade_sell_list: Vec<usize>,
    pub(crate) trade_buy_list: Vec<usize>,
    pub(crate) risk_degree: f64,
    pub(crate) risk_control_signal: f64,
    pub(crate) effective_n_drop: usize,
    pub(crate) desticky_active: bool,
    pub(crate) intraperiod: IntraperiodExitStats,
    pub(crate) intraperiod_signal_values: Vec<(usize, f64)>,
    pub(crate) intraperiod_residual_values: Vec<(usize, f64)>,
    pub(crate) intraperiod_events: Vec<TraceExitEvent>,
    pub(crate) target_weights: Vec<(usize, f64)>,
    pub(crate) target_values: Vec<(usize, f64)>,
    pub(crate) buy_count: usize,
    pub(crate) sell_count: usize,
    pub(crate) buy_value: f64,
    pub(crate) sell_value: f64,
    pub(crate) trade_cost_value: f64,
    pub(crate) gross_return: f64,
    pub(crate) net_return: f64,
    pub(crate) frozen_holdings: usize,
}

impl Default for IntraperiodExitStats {
    fn default() -> Self {
        Self {
            exit_count: 0,
            score_candidate_count: 0,
            price_confirm_required_count: 0,
            price_confirm_blocked_count: 0,
            price_confirm_bypassed_remaining_steps_count: 0,
            price_confirm_bypassed_force_exit_count: 0,
            remaining_steps: 0,
            signal_sum: 0.0,
            signal_min: f64::INFINITY,
            signal_count: 0,
            residual_sum: 0.0,
            residual_min: f64::INFINITY,
            residual_max: f64::NEG_INFINITY,
            residual_count: 0,
            saved_return: 0.0,
            missed_return: 0.0,
            beneficial_count: 0,
            harmful_count: 0,
        }
    }
}

impl IntraperiodExitStats {
    fn observe_signal(&mut self, value: f64) {
        if finite(value) {
            self.signal_sum += value;
            self.signal_count += 1;
            self.signal_min = self.signal_min.min(value);
        }
    }

    fn observe_residual(&mut self, residual: f64, position_value: f64, denom: f64) {
        self.residual_sum += residual;
        self.residual_count += 1;
        self.residual_min = self.residual_min.min(residual);
        self.residual_max = self.residual_max.max(residual);
        self.saved_return += (-residual).max(0.0) * position_value / denom;
        self.missed_return += residual.max(0.0) * position_value / denom;
        self.beneficial_count += usize::from(residual < 0.0);
        self.harmful_count += usize::from(residual > 0.0);
    }

    pub(crate) fn signal_mean(&self) -> f64 {
        if self.signal_count > 0 {
            self.signal_sum / self.signal_count as f64
        } else {
            f64::NAN
        }
    }

    pub(crate) fn residual_mean(&self) -> f64 {
        if self.residual_count > 0 {
            self.residual_sum / self.residual_count as f64
        } else {
            f64::NAN
        }
    }

    fn write_to(&self, row: &mut [f64]) {
        row[OUT_INTRAPERIOD_EXIT_COUNT] = self.exit_count as f64;
        row[OUT_INTRAPERIOD_SCORE_CANDIDATE_COUNT] = self.score_candidate_count as f64;
        row[OUT_INTRAPERIOD_PRICE_CONFIRM_REQUIRED_COUNT] =
            self.price_confirm_required_count as f64;
        row[OUT_INTRAPERIOD_PRICE_CONFIRM_BLOCKED_COUNT] = self.price_confirm_blocked_count as f64;
        row[OUT_INTRAPERIOD_PRICE_CONFIRM_BYPASSED_REMAINING_STEPS_COUNT] =
            self.price_confirm_bypassed_remaining_steps_count as f64;
        row[OUT_INTRAPERIOD_PRICE_CONFIRM_BYPASSED_FORCE_EXIT_COUNT] =
            self.price_confirm_bypassed_force_exit_count as f64;
        row[OUT_INTRAPERIOD_REMAINING_STEPS] = self.remaining_steps as f64;
        row[OUT_INTRAPERIOD_SIGNAL_MEAN] = self.signal_mean();
        row[OUT_INTRAPERIOD_SIGNAL_MIN] = if self.signal_count > 0 {
            self.signal_min
        } else {
            f64::NAN
        };
        row[OUT_INTRAPERIOD_RESIDUAL_MEAN] = self.residual_mean();
        row[OUT_INTRAPERIOD_RESIDUAL_MIN] = if self.residual_count > 0 {
            self.residual_min
        } else {
            f64::NAN
        };
        row[OUT_INTRAPERIOD_RESIDUAL_MAX] = if self.residual_count > 0 {
            self.residual_max
        } else {
            f64::NAN
        };
        row[OUT_INTRAPERIOD_SAVED_RETURN] = self.saved_return;
        row[OUT_INTRAPERIOD_MISSED_RETURN] = self.missed_return;
        row[OUT_INTRAPERIOD_BENEFICIAL_COUNT] = self.beneficial_count as f64;
        row[OUT_INTRAPERIOD_HARMFUL_COUNT] = self.harmful_count as f64;
    }
}

fn snapshot_holding_pairs(holdings: &[f64]) -> Vec<(usize, f64)> {
    holdings
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| {
            if value > EPS {
                Some((idx, value))
            } else {
                None
            }
        })
        .collect()
}

fn locked_indices(locked: &[bool]) -> Vec<usize> {
    locked
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value { Some(idx) } else { None })
        .collect()
}

pub(crate) fn run_backtest_core_impl(
    inputs: CoreInputs<'_>,
    params: BacktestParams,
    out: &mut [f64],
) -> Result<(), String> {
    run_backtest_core_impl_with_trace(inputs, params, out, None).map(|_| ())
}

pub(crate) fn run_backtest_core_impl_with_trace(
    inputs: CoreInputs<'_>,
    params: BacktestParams,
    out: &mut [f64],
    trace_mask: Option<&[u8]>,
) -> Result<Vec<TraceRecord>, String> {
    inputs.validate(out.len())?;
    params.validate()?;
    if trace_mask.is_some_and(|mask| mask.len() != inputs.n_dates) {
        return Err("trace_mask must be empty or have one value per date".to_string());
    }
    let CoreInputs {
        scores,
        labels,
        bench,
        group_ids,
        risk_values,
        risk_signal_values,
        price_confirm,
        intraperiod_scores: input_intraperiod_scores,
        n_dates,
        n_instruments,
    } = inputs;
    let BacktestParams {
        topk,
        n_drop,
        rebalance_freq,
        account,
        default_risk_degree,
        open_rate,
        close_rate,
        min_cost,
        weighting_mode,
        score_transform_mode,
        zscore_clip,
        intraperiod_exit_mode,
        intraperiod_score_transform_mode,
        intraperiod_exit_threshold,
        intraperiod_expected_return_n_bins,
        intraperiod_expected_return_min_history,
        price_confirm_min_remaining_steps,
        price_confirm_force_exit_threshold,
        max_weight,
        max_group_weight,
        keep_top_n,
        min_score,
        desticky_threshold,
        desticky_n_drop,
    } = params;

    let mut cash = account;
    let mut holdings = vec![0.0_f64; n_instruments];
    let mut order: Vec<usize> = Vec::new();
    let mut locked = vec![false; n_instruments];
    let mut target_by_stock = vec![0.0_f64; n_instruments];
    let mut transformed_scores = vec![f64::NAN; n_instruments];
    let mut intraperiod_scores = Vec::new();
    let mut intraperiod_expected_scores = Vec::new();
    let mut expected_histories: Vec<ExpectedReturnHistory> =
        std::iter::repeat_with(ExpectedReturnHistory::default)
            .take(rebalance_freq + 1)
            .collect();
    let mut trace_records = Vec::new();

    for pos in 0..n_dates {
        let trace_this_date = trace_mask.is_some_and(|mask| mask[pos] != 0);
        let row_start = pos * n_instruments;
        let score_row = &scores[row_start..row_start + n_instruments];
        let intraperiod_score_row = if input_intraperiod_scores.is_empty() {
            score_row
        } else {
            &input_intraperiod_scores[row_start..row_start + n_instruments]
        };
        let label_row = &labels[row_start..row_start + n_instruments];
        transform_score_row(
            score_row,
            score_transform_mode,
            zscore_clip,
            &mut transformed_scores,
        )?;
        let cash_before = cash;
        let holdings_before = if trace_this_date {
            snapshot_holding_pairs(&holdings)
        } else {
            Vec::new()
        };
        let start_value = cash + holdings.iter().sum::<f64>();
        let current_risk_degree = if risk_values.is_empty() {
            default_risk_degree
        } else {
            risk_values[pos]
        };
        let current_risk_signal =
            if !risk_signal_values.is_empty() && finite(risk_signal_values[pos]) {
                risk_signal_values[pos]
            } else {
                f64::NAN
            };

        let mut bench_sum = 0.0;
        let mut bench_count = 0usize;
        for idx in 0..n_instruments {
            let stock_return = label_row[idx];
            if finite(stock_return) {
                bench_sum += stock_return;
                bench_count += 1;
            }
            locked[idx] = holdings[idx] > EPS && !finite(stock_return);
        }

        let mut trades = TradeStats::default();
        let mut intraperiod = IntraperiodExitStats::default();
        let is_rebalance = pos % rebalance_freq == 0;
        let mut effective_n_drop = n_drop;
        let mut desticky_active = false;
        let mut exit_score_prepared = false;
        let mut trace_sell_list = Vec::new();
        let mut trace_buy_list = Vec::new();
        let mut trace_trade_sell_list = Vec::new();
        let mut trace_trade_buy_list = Vec::new();
        let mut trace_target_weights = Vec::new();
        let mut trace_target_values = Vec::new();
        let mut trace_intraperiod_signal_values = Vec::new();
        let mut trace_intraperiod_residual_values = Vec::new();
        let mut trace_intraperiod_events = Vec::new();
        let trace_locked_holdings = if trace_this_date {
            locked_indices(&locked)
        } else {
            Vec::new()
        };

        if is_rebalance {
            if let (Some(threshold), Some(drop_count)) = (desticky_threshold, desticky_n_drop) {
                if finite(current_risk_signal) && current_risk_signal <= threshold {
                    effective_n_drop = drop_count;
                    desticky_active = effective_n_drop > n_drop;
                }
            }

            let current = current_holdings_order(&holdings, &order);
            let (sell_list, buy_list) = select_topk_dropout_trades(
                &transformed_scores,
                &current,
                &locked,
                topk,
                effective_n_drop,
                keep_top_n,
                min_score,
            );
            if trace_this_date {
                trace_sell_list = sell_list.clone();
                trace_buy_list = buy_list.clone();
            }

            for stock in sell_list {
                let position_value = holdings[stock];
                holdings[stock] = 0.0;
                if position_value <= 0.0 {
                    continue;
                }
                let cost_value = trade_cost(position_value, close_rate, min_cost);
                cash += position_value - cost_value;
                trades.record_sell(position_value, cost_value);
                if trace_this_date && !trace_trade_sell_list.contains(&stock) {
                    trace_trade_sell_list.push(stock);
                }
            }
            compact_order(&mut order, &holdings);

            let tradable_holdings: Vec<usize> = current_holdings_order(&holdings, &order)
                .into_iter()
                .filter(|&idx| !locked[idx])
                .collect();

            let eligible_current_holdings: Vec<usize> = match min_score {
                Some(threshold) => tradable_holdings
                    .iter()
                    .copied()
                    .filter(|&idx| {
                        finite(transformed_scores[idx]) && transformed_scores[idx] > threshold
                    })
                    .collect(),
                None => tradable_holdings.clone(),
            };

            let mut target_seen = vec![false; n_instruments];
            let mut target_holdings = Vec::new();
            for stock in eligible_current_holdings
                .into_iter()
                .chain(buy_list.into_iter())
            {
                if !target_seen[stock] {
                    target_seen[stock] = true;
                    target_holdings.push(stock);
                }
            }

            target_by_stock.fill(0.0);
            if !target_holdings.is_empty() {
                let target_weights = compute_target_weights(
                    &transformed_scores,
                    &target_holdings,
                    weighting_mode,
                    max_weight,
                    group_ids,
                    max_group_weight,
                )?;
                let locked_value: f64 = holdings
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, value)| if locked[idx] { Some(*value) } else { None })
                    .sum();
                let tradable_budget = (start_value * current_risk_degree - locked_value).max(0.0);
                let target_values: Vec<f64> = target_weights
                    .iter()
                    .map(|weight| weight * tradable_budget)
                    .collect();
                for (&stock, &target_value) in target_holdings.iter().zip(target_values.iter()) {
                    target_by_stock[stock] = target_value;
                }
                if trace_this_date {
                    trace_target_weights = target_holdings
                        .iter()
                        .copied()
                        .zip(target_weights.iter().copied())
                        .collect();
                    trace_target_values = target_holdings
                        .iter()
                        .copied()
                        .zip(target_values.iter().copied())
                        .collect();
                }

                for &stock in &tradable_holdings {
                    let current_value = holdings[stock];
                    let target_value = target_by_stock[stock];
                    let trade_value = current_value - target_value;
                    if trade_value <= EPS {
                        continue;
                    }
                    let cost_value = trade_cost(trade_value, close_rate, min_cost);
                    cash += trade_value - cost_value;
                    holdings[stock] = target_value;
                    if holdings[stock] <= EPS {
                        holdings[stock] = 0.0;
                    }
                    trades.record_sell(trade_value, cost_value);
                    if trace_this_date && !trace_trade_sell_list.contains(&stock) {
                        trace_trade_sell_list.push(stock);
                    }
                }
                compact_order(&mut order, &holdings);

                let mut buy_order: Vec<usize> = (0..target_holdings.len()).collect();
                buy_order.sort_by(|&left, &right| {
                    match target_values[right]
                        .partial_cmp(&target_values[left])
                        .unwrap_or(Ordering::Equal)
                    {
                        Ordering::Equal => left.cmp(&right),
                        ordering => ordering,
                    }
                });

                for target_idx in buy_order {
                    let stock = target_holdings[target_idx];
                    let target_value = target_values[target_idx];
                    let current_value = holdings[stock];
                    let deficit_value = target_value - current_value;
                    if deficit_value <= EPS {
                        continue;
                    }
                    let mut trade_value =
                        deficit_value.min(max_affordable_trade_value(cash, open_rate, min_cost));
                    if trade_value <= 0.0 {
                        continue;
                    }
                    let mut cost_value = trade_cost(trade_value, open_rate, min_cost);
                    if trade_value + cost_value > cash {
                        trade_value = max_affordable_trade_value(cash, open_rate, min_cost);
                        cost_value = if trade_value > 0.0 {
                            trade_cost(trade_value, open_rate, min_cost)
                        } else {
                            0.0
                        };
                    }
                    if trade_value <= 0.0 || trade_value + cost_value > cash {
                        continue;
                    }
                    if holdings[stock] <= EPS && !order.contains(&stock) {
                        order.push(stock);
                    }
                    holdings[stock] += trade_value;
                    cash -= trade_value + cost_value;
                    trades.record_buy(trade_value, cost_value);
                    if trace_this_date && !trace_trade_buy_list.contains(&stock) {
                        trace_trade_buy_list.push(stock);
                    }
                }
            }
        } else if let Some(threshold) = intraperiod_exit_threshold {
            if intraperiod_score_transform_mode != SCORE_TRANSFORM_DISABLED {
                prepare_exit_score_row(
                    intraperiod_score_row,
                    &transformed_scores,
                    intraperiod_score_transform_mode,
                    zscore_clip,
                    &mut intraperiod_scores,
                )?;
                exit_score_prepared = true;
                intraperiod.remaining_steps =
                    intraperiod_remaining_steps_for_pos(pos, rebalance_freq, n_dates);

                let signal_scores = if intraperiod_exit_mode == INTRAPERIOD_EXIT_EXPECTED_RETURN {
                    expected_histories[intraperiod.remaining_steps].estimate(
                        &intraperiod_scores,
                        intraperiod_expected_return_n_bins,
                        intraperiod_expected_return_min_history,
                        &mut intraperiod_expected_scores,
                    );
                    intraperiod_expected_scores.as_slice()
                } else {
                    intraperiod_scores.as_slice()
                };

                for stock in current_holdings_order(&holdings, &order) {
                    let signal_value = signal_scores[stock];
                    intraperiod.observe_signal(signal_value);
                    if trace_this_date && finite(signal_value) {
                        trace_intraperiod_signal_values.push((stock, signal_value));
                    }
                }

                let exit_denom = if start_value > 0.0 { start_value } else { 1.0 };
                for stock in current_holdings_order(&holdings, &order) {
                    if locked[stock] {
                        continue;
                    }
                    let score_value = signal_scores[stock];
                    if !finite(score_value) || score_value > threshold {
                        continue;
                    }
                    intraperiod.score_candidate_count += 1;
                    let mut price_confirm_required = false;
                    let price_confirm_passed = true;
                    let mut price_confirm_bypass_reason = 0u8;
                    if !price_confirm.is_empty() {
                        if price_confirm_force_exit_threshold
                            .is_some_and(|force_threshold| score_value <= force_threshold)
                        {
                            intraperiod.price_confirm_bypassed_force_exit_count += 1;
                            price_confirm_bypass_reason = 1;
                        } else if intraperiod.remaining_steps < price_confirm_min_remaining_steps {
                            intraperiod.price_confirm_bypassed_remaining_steps_count += 1;
                            price_confirm_bypass_reason = 2;
                        } else {
                            intraperiod.price_confirm_required_count += 1;
                            price_confirm_required = true;
                            if price_confirm[row_start + stock] == 0 {
                                intraperiod.price_confirm_blocked_count += 1;
                                continue;
                            }
                        }
                    }
                    let position_value = holdings[stock];
                    holdings[stock] = 0.0;
                    if position_value <= 0.0 {
                        continue;
                    }

                    let residual_value = residual_return(
                        labels,
                        pos,
                        intraperiod.remaining_steps,
                        stock,
                        n_instruments,
                    );
                    let mut saved_return = 0.0;
                    let mut missed_return = 0.0;
                    let residual_for_trace = residual_value.unwrap_or(f64::NAN);
                    if let Some(residual_value) = residual_value {
                        saved_return = (-residual_value).max(0.0) * position_value / exit_denom;
                        missed_return = residual_value.max(0.0) * position_value / exit_denom;
                        intraperiod.observe_residual(residual_value, position_value, exit_denom);
                        if trace_this_date {
                            trace_intraperiod_residual_values.push((stock, residual_value));
                        }
                    }

                    let cost_value = trade_cost(position_value, close_rate, min_cost);
                    cash += position_value - cost_value;
                    trades.record_sell(position_value, cost_value);
                    intraperiod.exit_count += 1;
                    if trace_this_date {
                        if !trace_trade_sell_list.contains(&stock) {
                            trace_trade_sell_list.push(stock);
                        }
                        trace_sell_list.push(stock);
                        trace_intraperiod_events.push(TraceExitEvent {
                            stock,
                            score_value,
                            residual_return_if_held: residual_for_trace,
                            position_value,
                            saved_return_contribution: saved_return,
                            missed_return_contribution: missed_return,
                            remaining_steps: intraperiod.remaining_steps,
                            price_confirm_required,
                            price_confirm_passed,
                            price_confirm_bypass_reason,
                        });
                    }
                }
                compact_order(&mut order, &holdings);
            }
        }

        let mut gross_pnl = 0.0;
        let mut frozen_holdings = 0usize;
        for idx in 0..n_instruments {
            let position_value = holdings[idx];
            if position_value <= EPS {
                continue;
            }
            let mut stock_return = label_row[idx];
            if !finite(stock_return) {
                frozen_holdings += 1;
                stock_return = 0.0;
            }
            let new_value = position_value * (1.0 + stock_return);
            gross_pnl += new_value - position_value;
            holdings[idx] = new_value;
        }
        compact_order(&mut order, &holdings);

        let end_value = cash + holdings.iter().sum::<f64>();
        let denom = if start_value > 0.0 { start_value } else { 1.0 };
        let out_start = pos * OUT_COLS;
        let out_row = &mut out[out_start..out_start + OUT_COLS];
        out_row[OUT_GROSS_RETURN] = gross_pnl / denom;
        out_row[OUT_NET_RETURN] = (end_value - start_value) / denom;
        out_row[OUT_TURNOVER] = (trades.buy_value + trades.sell_value) / (2.0 * denom);
        out_row[OUT_COST] = trades.cost / denom;
        out_row[OUT_BENCH] = if bench.is_empty() {
            if bench_count > 0 {
                bench_sum / bench_count as f64
            } else {
                0.0
            }
        } else {
            bench[pos]
        };
        out_row[OUT_BUY_COUNT] = trades.buy_count as f64;
        out_row[OUT_SELL_COUNT] = trades.sell_count as f64;
        out_row[OUT_HOLDINGS] = holdings.iter().filter(|value| **value > EPS).count() as f64;
        out_row[OUT_FROZEN_HOLDINGS] = frozen_holdings as f64;
        out_row[OUT_ACCOUNT_VALUE] = end_value;
        out_row[OUT_RISK_DEGREE] = current_risk_degree;
        out_row[OUT_EFFECTIVE_N_DROP] = effective_n_drop as f64;
        out_row[OUT_RISK_SIGNAL] = current_risk_signal;
        out_row[OUT_DESTICKY_ACTIVE] = if desticky_active { 1.0 } else { 0.0 };
        intraperiod.write_to(out_row);

        if trace_this_date {
            trace_records.push(TraceRecord {
                pos,
                start_value,
                end_value,
                cash_before,
                cash_after: cash,
                holdings_before,
                holdings_after: snapshot_holding_pairs(&holdings),
                locked_holdings: trace_locked_holdings,
                sell_list: trace_sell_list,
                buy_list: trace_buy_list,
                trade_sell_list: trace_trade_sell_list,
                trade_buy_list: trace_trade_buy_list,
                risk_degree: current_risk_degree,
                risk_control_signal: current_risk_signal,
                effective_n_drop,
                desticky_active,
                intraperiod,
                intraperiod_signal_values: trace_intraperiod_signal_values,
                intraperiod_residual_values: trace_intraperiod_residual_values,
                intraperiod_events: trace_intraperiod_events,
                target_weights: trace_target_weights,
                target_values: trace_target_values,
                buy_count: trades.buy_count,
                sell_count: trades.sell_count,
                buy_value: trades.buy_value,
                sell_value: trades.sell_value,
                trade_cost_value: trades.cost,
                gross_return: out_row[OUT_GROSS_RETURN],
                net_return: out_row[OUT_NET_RETURN],
                frozen_holdings,
            });
        }

        if intraperiod_exit_mode == INTRAPERIOD_EXIT_EXPECTED_RETURN
            && intraperiod_exit_threshold.is_some()
            && intraperiod_score_transform_mode != SCORE_TRANSFORM_DISABLED
        {
            let remaining_steps = intraperiod_remaining_steps_for_pos(pos, rebalance_freq, n_dates);
            if remaining_steps > 0 {
                if !exit_score_prepared {
                    prepare_exit_score_row(
                        intraperiod_score_row,
                        &transformed_scores,
                        intraperiod_score_transform_mode,
                        zscore_clip,
                        &mut intraperiod_scores,
                    )?;
                }
                expected_histories[remaining_steps].append_from_row(
                    &intraperiod_scores,
                    labels,
                    pos,
                    remaining_steps,
                    n_instruments,
                );
            }
        }
    }

    Ok(trace_records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trade_cost_uses_minimum_only_for_positive_trades() {
        assert_eq!(trade_cost(0.0, 0.001, 5.0), 0.0);
        assert_eq!(trade_cost(-1.0, 0.001, 5.0), 0.0);
        assert_eq!(trade_cost(100.0, 0.001, 5.0), 5.0);
        assert_eq!(trade_cost(10_000.0, 0.001, 5.0), 10.0);
    }

    #[test]
    fn select_topk_dropout_replaces_only_the_dropped_holding() {
        let scores = [3.0, 2.0, 1.0, 4.0];
        let current = vec![0, 1];
        let locked = vec![false; 4];

        let (sell, buy) = select_topk_dropout_trades(&scores, &current, &locked, 2, 1, None, None);

        assert_eq!(sell, vec![1]);
        assert_eq!(buy, vec![3]);
    }

    #[test]
    fn min_score_empty_pool_sells_unlocked_holdings() {
        let scores = [-1.0, -2.0, f64::NAN];
        let current = vec![0, 1, 2];
        let locked = vec![false, true, false];

        let (sell, buy) =
            select_topk_dropout_trades(&scores, &current, &locked, 2, 1, None, Some(0.0));

        assert_eq!(sell, vec![0, 2]);
        assert!(buy.is_empty());
    }

    #[test]
    fn rank_weights_allocate_more_to_better_scores() {
        let scores = [1.0, 3.0, 2.0];
        let target = vec![0, 1, 2];
        let weights =
            compute_target_weights(&scores, &target, WEIGHT_RANK, None, &[], None).unwrap();

        assert!(weights[1] > weights[2]);
        assert!(weights[2] > weights[0]);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rank_pct_transform_matches_average_percentile_ranks() {
        let scores = [3.0, 1.0, 3.0, f64::NAN, 2.0];
        let mut out = vec![0.0; scores.len()];

        rank_pct_transform_row(&scores, &mut out);

        assert!((out[0] - 0.875).abs() < 1e-12);
        assert!((out[1] - 0.25).abs() < 1e-12);
        assert!((out[2] - 0.875).abs() < 1e-12);
        assert!(out[3].is_nan());
        assert!((out[4] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn zscore_clip_transform_preserves_nan_and_zeroes_constant_rows() {
        let scores = [2.0, 2.0, f64::NAN];
        let mut out = vec![1.0; scores.len()];

        zscore_clip_transform_row(&scores, 3.0, &mut out);

        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert!(out[2].is_nan());
    }

    #[test]
    fn simple_equal_core_produces_finite_daily_records() {
        let scores = [3.0, 2.0, 1.0, 1.0, 3.0, 2.0];
        let labels = [0.10, 0.00, -0.05, 0.00, 0.20, 0.05];
        let bench = [0.016666666666666666, 0.08333333333333333];
        let mut out = [0.0_f64; OUT_COLS * 2];

        let inputs = CoreInputs {
            scores: &scores,
            labels: &labels,
            bench: &bench,
            group_ids: &[],
            risk_values: &[],
            risk_signal_values: &[],
            price_confirm: &[],
            intraperiod_scores: &[],
            n_dates: 2,
            n_instruments: 3,
        };
        let params = BacktestParams {
            topk: 2,
            n_drop: 1,
            rebalance_freq: 1,
            account: 1_000.0,
            default_risk_degree: 1.0,
            open_rate: 0.0,
            close_rate: 0.0,
            min_cost: 0.0,
            weighting_mode: WEIGHT_EQUAL,
            score_transform_mode: SCORE_TRANSFORM_NONE,
            zscore_clip: 3.0,
            intraperiod_exit_mode: INTRAPERIOD_EXIT_SCORE_THRESHOLD,
            intraperiod_score_transform_mode: SCORE_TRANSFORM_DISABLED,
            intraperiod_exit_threshold: None,
            intraperiod_expected_return_n_bins: 20,
            intraperiod_expected_return_min_history: 200,
            price_confirm_min_remaining_steps: 0,
            price_confirm_force_exit_threshold: None,
            max_weight: None,
            max_group_weight: None,
            keep_top_n: None,
            min_score: None,
            desticky_threshold: None,
            desticky_n_drop: None,
        };

        run_backtest_core_impl(inputs, params, &mut out).unwrap();

        assert_eq!(out[OUT_BUY_COUNT], 2.0);
        assert_eq!(out[OUT_COLS + OUT_BUY_COUNT], 2.0);
        assert_eq!(out[OUT_COLS + OUT_SELL_COUNT], 1.0);
        assert!(out.iter().all(|value| value.is_finite() || value.is_nan()));
    }
}
