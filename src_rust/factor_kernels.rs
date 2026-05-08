use std::collections::HashMap;

const EPS: f64 = 1e-12;

#[allow(clippy::too_many_arguments)]
pub fn alpha158_kbar_price(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    vwap: &[f64],
    include_kbar: bool,
    price_features: &[String],
    price_windows: &[usize],
) -> Result<Vec<(String, Vec<f64>)>, String> {
    let len = open.len();
    for (name, values) in [
        ("high", high),
        ("low", low),
        ("close", close),
        ("vwap", vwap),
    ] {
        if values.len() != len {
            return Err(format!(
                "alpha158_kbar_price length mismatch: open has {}, {name} has {}",
                len,
                values.len()
            ));
        }
    }

    let mut out = Vec::new();
    if include_kbar {
        let mut kmid = Vec::with_capacity(len);
        let mut klen = Vec::with_capacity(len);
        let mut kmid2 = Vec::with_capacity(len);
        let mut kup = Vec::with_capacity(len);
        let mut kup2 = Vec::with_capacity(len);
        let mut klow = Vec::with_capacity(len);
        let mut klow2 = Vec::with_capacity(len);
        let mut ksft = Vec::with_capacity(len);
        let mut ksft2 = Vec::with_capacity(len);
        for idx in 0..len {
            let open_value = open[idx];
            let high_value = high[idx];
            let low_value = low[idx];
            let close_value = close[idx];
            let hl_range = high_value - low_value + EPS;
            let max_open_close = numpy_max(open_value, close_value);
            let min_open_close = numpy_min(open_value, close_value);
            kmid.push((close_value - open_value) / open_value);
            klen.push((high_value - low_value) / open_value);
            kmid2.push((close_value - open_value) / hl_range);
            kup.push((high_value - max_open_close) / open_value);
            kup2.push((high_value - max_open_close) / hl_range);
            klow.push((min_open_close - low_value) / open_value);
            klow2.push((min_open_close - low_value) / hl_range);
            ksft.push((2.0 * close_value - high_value - low_value) / open_value);
            ksft2.push((2.0 * close_value - high_value - low_value) / hl_range);
        }
        out.extend([
            ("KMID".to_owned(), kmid),
            ("KLEN".to_owned(), klen),
            ("KMID2".to_owned(), kmid2),
            ("KUP".to_owned(), kup),
            ("KUP2".to_owned(), kup2),
            ("KLOW".to_owned(), klow),
            ("KLOW2".to_owned(), klow2),
            ("KSFT".to_owned(), ksft),
            ("KSFT2".to_owned(), ksft2),
        ]);
    }

    for raw_field in price_features {
        let field = raw_field.to_ascii_uppercase();
        let series = match field.as_str() {
            "OPEN" => open,
            "HIGH" => high,
            "LOW" => low,
            "CLOSE" => close,
            "VWAP" => vwap,
            other => {
                return Err(format!(
                    "alpha158_kbar_price unsupported price feature: {other}"
                ))
            }
        };
        for &window in price_windows {
            out.push((
                format!("{field}{window}"),
                shifted_ratio(series, close, window),
            ));
        }
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn alpha158_features(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    vwap: &[f64],
    volume: &[f64],
    include_kbar: bool,
    price_features: &[String],
    price_windows: &[usize],
    volume_windows: &[usize],
    rolling_windows: &[usize],
    rolling_ops: &[String],
) -> Result<Vec<(String, Vec<f64>)>, String> {
    let len = close.len();
    for (name, values) in [
        ("open", open),
        ("high", high),
        ("low", low),
        ("vwap", vwap),
        ("volume", volume),
    ] {
        if values.len() != len {
            return Err(format!(
                "alpha158_features length mismatch: close has {}, {name} has {}",
                len,
                values.len()
            ));
        }
    }
    let mut out = alpha158_kbar_price(
        open,
        high,
        low,
        close,
        vwap,
        include_kbar,
        price_features,
        price_windows,
    )?;
    for &window in volume_windows {
        out.push((
            format!("VOLUME{window}"),
            shifted_ratio_eps(volume, volume, window),
        ));
    }
    if rolling_windows.is_empty() {
        return Ok(out);
    }

    let ref_close1 = shift(close, 1);
    let ref_vol1 = shift(volume, 1);
    let close_ret = safe_divide(close, &ref_close1);
    let vol_ret_log = log_plus_one(&safe_divide(volume, &ref_vol1));
    let close_delta = subtract(close, &ref_close1);
    let vol_delta = subtract(volume, &ref_vol1);
    let close_delta_pos = clip_lower(&close_delta, 0.0);
    let close_delta_neg = clip_lower(
        &close_delta.iter().map(|value| -*value).collect::<Vec<_>>(),
        0.0,
    );
    let vol_delta_pos = clip_lower(&vol_delta, 0.0);
    let vol_delta_neg = clip_lower(
        &vol_delta.iter().map(|value| -*value).collect::<Vec<_>>(),
        0.0,
    );
    let close_abs_delta = abs_values(&close_delta);
    let vol_abs_delta = abs_values(&vol_delta);
    let log_volume = log_plus_one(volume);

    for &window in rolling_windows {
        let close_mean = rolling_mean(close, window);
        let close_std = rolling_std(close, window);
        let high_max = rolling_max(high, window);
        let low_min = rolling_min(low, window);
        let volume_mean = rolling_mean(volume, window);
        let volume_std = rolling_std(volume, window);
        let close_abs_delta_sum = rolling_sum_min_count(&close_abs_delta, window, 1);
        let vol_abs_delta_sum = rolling_sum_min_count(&vol_abs_delta, window, 1);
        let mut regression_stats = None;

        if has_group(rolling_ops, "ROC") {
            out.push((format!("ROC{window}"), shifted_ratio(close, close, window)));
        }
        if has_group(rolling_ops, "MA") {
            out.push((format!("MA{window}"), safe_divide(&close_mean, close)));
        }
        if has_group(rolling_ops, "STD") {
            out.push((format!("STD{window}"), safe_divide(&close_std, close)));
        }
        if has_group(rolling_ops, "BETA")
            || has_group(rolling_ops, "RSQR")
            || has_group(rolling_ops, "RESI")
        {
            regression_stats = Some(rolling_regression_stats(close, window));
        }
        if has_group(rolling_ops, "BETA") {
            let (slope, _, _) = regression_stats
                .as_ref()
                .expect("regression stats initialized");
            out.push((format!("BETA{window}"), safe_divide(slope, close)));
        }
        if has_group(rolling_ops, "RSQR") {
            let (_, rsquare, _) = regression_stats
                .as_ref()
                .expect("regression stats initialized");
            out.push((format!("RSQR{window}"), rsquare.clone()));
        }
        if has_group(rolling_ops, "RESI") {
            let (_, _, residual) = regression_stats
                .as_ref()
                .expect("regression stats initialized");
            out.push((format!("RESI{window}"), safe_divide(residual, close)));
        }
        if has_group(rolling_ops, "MAX") {
            out.push((format!("MAX{window}"), safe_divide(&high_max, close)));
        }
        if has_group(rolling_ops, "LOW") {
            out.push((format!("MIN{window}"), safe_divide(&low_min, close)));
        }
        if has_group(rolling_ops, "QTLU") {
            out.push((
                format!("QTLU{window}"),
                safe_divide(&rolling_quantile(close, window, 0.8), close),
            ));
        }
        if has_group(rolling_ops, "QTLD") {
            out.push((
                format!("QTLD{window}"),
                safe_divide(&rolling_quantile(close, window, 0.2), close),
            ));
        }
        if has_group(rolling_ops, "RANK") {
            out.push((
                format!("RANK{window}"),
                crate::feature_kernels::rolling_rank_pct(close, window),
            ));
        }
        if has_group(rolling_ops, "RSV") {
            out.push((
                format!("RSV{window}"),
                close
                    .iter()
                    .zip(&low_min)
                    .zip(&high_max)
                    .map(|((close_value, low_value), high_value)| {
                        (*close_value - *low_value) / (*high_value - *low_value + EPS)
                    })
                    .collect(),
            ));
        }
        let mut imax = None;
        let mut imin = None;
        if has_group(rolling_ops, "IMAX") || has_group(rolling_ops, "IMXD") {
            imax = Some(crate::feature_kernels::rolling_arg_extreme_position(
                high, window, true,
            ));
        }
        if has_group(rolling_ops, "IMIN") || has_group(rolling_ops, "IMXD") {
            imin = Some(crate::feature_kernels::rolling_arg_extreme_position(
                low, window, false,
            ));
        }
        if has_group(rolling_ops, "IMAX") {
            out.push((
                format!("IMAX{window}"),
                scale_values(
                    imax.as_ref().expect("imax initialized"),
                    1.0 / window as f64,
                ),
            ));
        }
        if has_group(rolling_ops, "IMIN") {
            out.push((
                format!("IMIN{window}"),
                scale_values(
                    imin.as_ref().expect("imin initialized"),
                    1.0 / window as f64,
                ),
            ));
        }
        if has_group(rolling_ops, "IMXD") {
            out.push((
                format!("IMXD{window}"),
                imax.as_ref()
                    .expect("imax initialized")
                    .iter()
                    .zip(imin.as_ref().expect("imin initialized"))
                    .map(|(max_idx, min_idx)| (*max_idx - *min_idx) / window as f64)
                    .collect(),
            ));
        }
        if has_group(rolling_ops, "CORR") {
            out.push((
                format!("CORR{window}"),
                rolling_corr_with_std_guard(close, &log_volume, window),
            ));
        }
        if has_group(rolling_ops, "CORD") {
            out.push((
                format!("CORD{window}"),
                rolling_corr_with_std_guard(&close_ret, &vol_ret_log, window),
            ));
        }
        if has_group(rolling_ops, "CNTP") {
            out.push((
                format!("CNTP{window}"),
                rolling_mean(&greater_than(close, &ref_close1), window),
            ));
        }
        if has_group(rolling_ops, "CNTN") {
            out.push((
                format!("CNTN{window}"),
                rolling_mean(&less_than(close, &ref_close1), window),
            ));
        }
        if has_group(rolling_ops, "CNTD") {
            let cntp = rolling_mean(&greater_than(close, &ref_close1), window);
            let cntn = rolling_mean(&less_than(close, &ref_close1), window);
            out.push((format!("CNTD{window}"), subtract(&cntp, &cntn)));
        }
        let mut close_pos_sum = None;
        let mut close_neg_sum = None;
        if has_group(rolling_ops, "SUMP")
            || has_group(rolling_ops, "SUMN")
            || has_group(rolling_ops, "SUMD")
        {
            close_pos_sum = Some(rolling_sum_min_count(&close_delta_pos, window, 1));
            close_neg_sum = Some(rolling_sum_min_count(&close_delta_neg, window, 1));
        }
        if has_group(rolling_ops, "SUMP") {
            out.push((
                format!("SUMP{window}"),
                safe_divide(
                    close_pos_sum.as_ref().expect("close pos initialized"),
                    &add_scalar(&close_abs_delta_sum, EPS),
                ),
            ));
        }
        if has_group(rolling_ops, "SUMN") {
            out.push((
                format!("SUMN{window}"),
                safe_divide(
                    close_neg_sum.as_ref().expect("close neg initialized"),
                    &add_scalar(&close_abs_delta_sum, EPS),
                ),
            ));
        }
        if has_group(rolling_ops, "SUMD") {
            out.push((
                format!("SUMD{window}"),
                safe_divide(
                    &subtract(
                        close_pos_sum.as_ref().expect("close pos initialized"),
                        close_neg_sum.as_ref().expect("close neg initialized"),
                    ),
                    &add_scalar(&close_abs_delta_sum, EPS),
                ),
            ));
        }
        if has_group(rolling_ops, "VMA") {
            out.push((
                format!("VMA{window}"),
                safe_divide_eps(&volume_mean, volume),
            ));
        }
        if has_group(rolling_ops, "VSTD") {
            out.push((
                format!("VSTD{window}"),
                safe_divide_eps(&volume_std, volume),
            ));
        }
        if has_group(rolling_ops, "WVMA") {
            let w = close_ret
                .iter()
                .zip(volume)
                .map(|(ret, volume_value)| (*ret - 1.0).abs() * *volume_value)
                .collect::<Vec<_>>();
            out.push((
                format!("WVMA{window}"),
                safe_divide(
                    &rolling_std(&w, window),
                    &add_scalar(&rolling_mean(&w, window), EPS),
                ),
            ));
        }
        let mut vol_pos_sum = None;
        let mut vol_neg_sum = None;
        if has_group(rolling_ops, "VSUMP")
            || has_group(rolling_ops, "VSUMN")
            || has_group(rolling_ops, "VSUMD")
        {
            vol_pos_sum = Some(rolling_sum_min_count(&vol_delta_pos, window, 1));
            vol_neg_sum = Some(rolling_sum_min_count(&vol_delta_neg, window, 1));
        }
        if has_group(rolling_ops, "VSUMP") {
            out.push((
                format!("VSUMP{window}"),
                safe_divide(
                    vol_pos_sum.as_ref().expect("vol pos initialized"),
                    &add_scalar(&vol_abs_delta_sum, EPS),
                ),
            ));
        }
        if has_group(rolling_ops, "VSUMN") {
            out.push((
                format!("VSUMN{window}"),
                safe_divide(
                    vol_neg_sum.as_ref().expect("vol neg initialized"),
                    &add_scalar(&vol_abs_delta_sum, EPS),
                ),
            ));
        }
        if has_group(rolling_ops, "VSUMD") {
            out.push((
                format!("VSUMD{window}"),
                safe_divide(
                    &subtract(
                        vol_pos_sum.as_ref().expect("vol pos initialized"),
                        vol_neg_sum.as_ref().expect("vol neg initialized"),
                    ),
                    &add_scalar(&vol_abs_delta_sum, EPS),
                ),
            ));
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn lgbm_purified_features(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    amount: &[f64],
    vwap: &[f64],
    circ_mv: &[f64],
    pe_ttm: &[f64],
    pb: &[f64],
    turnover: &[f64],
    has_pe_ttm: bool,
    has_pb: bool,
    momentum_windows: &[usize],
    ma_windows: &[usize],
    vol_window: usize,
    atr_window: usize,
    amihud_window: usize,
    volume_window: usize,
    corr_window: usize,
    turnover_window: usize,
    extreme_window: usize,
) -> Result<Vec<(String, Vec<f64>)>, String> {
    let len = close.len();
    for (name, values) in [
        ("high", high),
        ("low", low),
        ("volume", volume),
        ("amount", amount),
        ("vwap", vwap),
        ("circ_mv", circ_mv),
        ("pe_ttm", pe_ttm),
        ("pb", pb),
        ("turnover", turnover),
    ] {
        if values.len() != len {
            return Err(format!(
                "lgbm_purified_features length mismatch: close has {}, {name} has {}",
                len,
                values.len()
            ));
        }
    }

    let close_nonzero = replace_zero_with_nan(close);
    let volume_nonzero = replace_zero_with_nan(volume);
    let close_ret = pct_change(&close_nonzero, 1);
    let close_ret_abs = abs_values(&close_ret);
    let amount_abs = abs_values(amount);

    let mut out = Vec::new();
    for &window in momentum_windows {
        out.push((format!("ret_{window}"), pct_change(&close_nonzero, window)));
    }
    for &window in ma_windows {
        let ma = replace_zero_with_nan_owned(rolling_mean(&close_nonzero, window));
        out.push((
            format!("dist_ma{window}"),
            ratio_minus_one(&close_nonzero, &ma),
        ));
    }
    out.push(("std_60".to_owned(), rolling_std(&close_ret, vol_window)));
    out.push((
        "atr_14".to_owned(),
        calculate_atr(high, low, close, &close_nonzero, atr_window),
    ));
    out.push((
        "amihud_20".to_owned(),
        rolling_mean(
            &safe_divide(&close_ret_abs, &add_scalar(&amount_abs, EPS)),
            amihud_window,
        ),
    ));
    let vol_ma = replace_zero_with_nan_owned(rolling_mean(&volume_nonzero, volume_window));
    out.push((
        "vol_ratio_20".to_owned(),
        safe_divide(&volume_nonzero, &vol_ma),
    ));
    out.push((
        "corr_cv_20".to_owned(),
        rolling_corr(&close_nonzero, volume, corr_window),
    ));
    out.push((
        "vwap_ratio".to_owned(),
        ratio_minus_one(&close_nonzero, &replace_zero_with_nan(vwap)),
    ));
    out.push(("log_mcap".to_owned(), log1p_clipped_nonnegative(circ_mv)));
    if has_pe_ttm {
        let ep_ttm_clean = positive_inverse(pe_ttm);
        out.push(("ep_ttm".to_owned(), fill_nan(&ep_ttm_clean, -1.0)));
        out.push(("is_loss".to_owned(), nonpositive_or_zero_flag(pe_ttm)));
        out.push(("ep_ttm_clean".to_owned(), ep_ttm_clean));
        out.push((
            "ep_ttm_invalid".to_owned(),
            nonpositive_invalid_flag(pe_ttm),
        ));
    } else {
        push_nan_columns(
            &mut out,
            len,
            &["ep_ttm", "is_loss", "ep_ttm_clean", "ep_ttm_invalid"],
        );
    }
    if has_pb {
        let bp_clean = positive_inverse(pb);
        out.push(("bp".to_owned(), fill_nan(&bp_clean, -1.0)));
        out.push(("bp_clean".to_owned(), bp_clean));
        out.push(("bp_invalid".to_owned(), nonpositive_invalid_flag(pb)));
    } else {
        push_nan_columns(&mut out, len, &["bp", "bp_clean", "bp_invalid"]);
    }
    out.push((
        "turnover_20".to_owned(),
        rolling_mean(turnover, turnover_window),
    ));
    let rolling_high = replace_zero_with_nan_owned(rolling_max(high, extreme_window));
    let rolling_low = replace_zero_with_nan_owned(rolling_min(low, extreme_window));
    out.push((
        "dist_high_20".to_owned(),
        ratio_minus_one(&close_nonzero, &rolling_high),
    ));
    out.push((
        "dist_low_20".to_owned(),
        ratio_minus_one(&close_nonzero, &rolling_low),
    ));

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn temporal_factor_features(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    amount: &[f64],
    turnover: &[f64],
    windows: &[usize],
    groups: &[String],
) -> Result<Vec<(String, Vec<f64>)>, String> {
    let len = close.len();
    for (name, values) in [
        ("high", high),
        ("low", low),
        ("volume", volume),
        ("amount", amount),
        ("turnover", turnover),
    ] {
        if values.len() != len {
            return Err(format!(
                "temporal_factor_features length mismatch: close has {}, {name} has {}",
                len,
                values.len()
            ));
        }
    }

    let close_nonzero = replace_zero_with_nan(close);
    let high_nonzero = replace_zero_with_nan(high);
    let low_nonzero = replace_zero_with_nan(low);
    let volume_nonzero = replace_zero_with_nan(volume);
    let close_ret = pct_change(&close_nonzero, 1);
    let close_ret_abs = abs_values(&close_ret);
    let log_volume = log_plus_one(&volume_nonzero);
    let amount_abs = abs_values(amount);
    let mut out = Vec::new();

    for &window in windows {
        let mut rolling_low = None;
        let mut rolling_high = None;
        if has_group(groups, "ret") {
            out.push((format!("ret_{window}"), pct_change(&close_nonzero, window)));
        }
        if has_group(groups, "ma_gap") {
            let ma = replace_zero_with_nan_owned(rolling_mean(&close_nonzero, window));
            out.push((
                format!("ma_gap_{window}"),
                ratio_minus_one(&close_nonzero, &ma),
            ));
        }
        if has_group(groups, "std") {
            out.push((format!("std_{window}"), rolling_std(&close_ret, window)));
        }
        if has_group(groups, "rsv") || has_group(groups, "high_gap") || has_group(groups, "low_gap")
        {
            rolling_low = Some(rolling_min(&low_nonzero, window));
            rolling_high = Some(rolling_max(&high_nonzero, window));
        }
        if has_group(groups, "rsv") {
            let low_values = rolling_low.as_ref().expect("rolling low initialized");
            let high_values = rolling_high.as_ref().expect("rolling high initialized");
            out.push((
                format!("rsv_{window}"),
                close_nonzero
                    .iter()
                    .zip(low_values)
                    .zip(high_values)
                    .map(|((close_value, low_value), high_value)| {
                        (*close_value - *low_value) / (*high_value - *low_value + EPS)
                    })
                    .collect(),
            ));
        }
        if has_group(groups, "price_rank") {
            out.push((
                format!("price_rank_{window}"),
                crate::feature_kernels::rolling_rank_pct(&close_nonzero, window),
            ));
        }
        if has_group(groups, "volume_ratio") {
            let volume_ma = replace_zero_with_nan_owned(rolling_mean(&volume_nonzero, window));
            out.push((
                format!("volume_ratio_{window}"),
                ratio_minus_one(&volume_nonzero, &volume_ma),
            ));
        }
        if has_group(groups, "turnover_mean") {
            out.push((
                format!("turnover_mean_{window}"),
                rolling_mean(turnover, window),
            ));
        }
        if has_group(groups, "amihud") {
            out.push((
                format!("amihud_{window}"),
                rolling_mean(
                    &safe_divide(&close_ret_abs, &add_scalar(&amount_abs, EPS)),
                    window,
                ),
            ));
        }
        if has_group(groups, "high_gap") {
            let high_values = replace_zero_with_nan_owned(
                rolling_high
                    .as_ref()
                    .expect("rolling high initialized")
                    .clone(),
            );
            out.push((
                format!("high_gap_{window}"),
                ratio_minus_one(&close_nonzero, &high_values),
            ));
        }
        if has_group(groups, "low_gap") {
            let low_values = replace_zero_with_nan_owned(
                rolling_low
                    .as_ref()
                    .expect("rolling low initialized")
                    .clone(),
            );
            out.push((
                format!("low_gap_{window}"),
                ratio_minus_one(&close_nonzero, &low_values),
            ));
        }
        if has_group(groups, "corr_cv") {
            out.push((
                format!("corr_cv_{window}"),
                rolling_corr(&close_nonzero, &log_volume, window),
            ));
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn technical_factor_features(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    macd_fast: &[usize],
    macd_slow: &[usize],
    macd_signal: &[usize],
    rsi_windows: &[usize],
    boll_windows: &[usize],
    boll_num_std: f64,
    adx_windows: &[usize],
    mfi_windows: &[usize],
    cci_windows: &[usize],
    willr_windows: &[usize],
    aroon_windows: &[usize],
    trix_windows: &[usize],
    trix_signal: usize,
    obv_windows: &[usize],
) -> Result<Vec<(String, Vec<f64>)>, String> {
    let len = close.len();
    for (name, values) in [("high", high), ("low", low), ("volume", volume)] {
        if values.len() != len {
            return Err(format!(
                "technical_factor_features length mismatch: close has {}, {name} has {}",
                len,
                values.len()
            ));
        }
    }
    if macd_fast.len() != macd_slow.len() || macd_fast.len() != macd_signal.len() {
        return Err(
            "technical_factor_features MACD setup arrays have different lengths".to_owned(),
        );
    }

    let close_nonzero = replace_zero_with_nan(close);
    let mut out = Vec::new();

    for idx in 0..macd_fast.len() {
        let fast = macd_fast[idx];
        let slow = macd_slow[idx];
        let signal = macd_signal[idx];
        let (macd_line, signal_line, hist) = calculate_macd(&close_nonzero, fast, slow, signal);
        out.push((format!("macd_line_{fast}_{slow}_{signal}"), macd_line));
        out.push((format!("macd_signal_{fast}_{slow}_{signal}"), signal_line));
        out.push((format!("macd_hist_{fast}_{slow}_{signal}"), hist));
    }
    for &window in rsi_windows {
        out.push((
            format!("rsi_{window}"),
            calculate_rsi(&close_nonzero, window),
        ));
    }
    let num_std_label = boll_num_std as i64;
    for &window in boll_windows {
        let (pos, width, zscore) = calculate_bollinger(&close_nonzero, window, boll_num_std);
        out.push((format!("boll_pos_{window}_{num_std_label}"), pos));
        out.push((format!("boll_width_{window}_{num_std_label}"), width));
        out.push((format!("boll_zscore_{window}_{num_std_label}"), zscore));
    }
    for &window in adx_windows {
        let (adx, plus_di, minus_di) = calculate_adx(high, low, &close_nonzero, window);
        out.push((format!("adx_{window}"), adx));
        out.push((format!("plus_di_{window}"), plus_di));
        out.push((format!("minus_di_{window}"), minus_di));
    }
    for &window in mfi_windows {
        out.push((
            format!("mfi_{window}"),
            calculate_mfi(high, low, &close_nonzero, volume, window),
        ));
    }
    for &window in cci_windows {
        out.push((
            format!("cci_{window}"),
            calculate_cci(high, low, &close_nonzero, window),
        ));
    }
    for &window in willr_windows {
        out.push((
            format!("willr_{window}"),
            calculate_willr(high, low, &close_nonzero, window),
        ));
    }
    for &window in aroon_windows {
        let (aroon_up, aroon_down, aroon_osc) = calculate_aroon(high, low, window);
        out.push((format!("aroon_up_{window}"), aroon_up));
        out.push((format!("aroon_down_{window}"), aroon_down));
        out.push((format!("aroon_osc_{window}"), aroon_osc));
    }
    for &window in trix_windows {
        let (trix, signal, hist) = calculate_trix(&close_nonzero, window, trix_signal);
        out.push((format!("trix_{window}"), trix));
        out.push((format!("trix_signal_{window}_{trix_signal}"), signal));
        out.push((format!("trix_hist_{window}_{trix_signal}"), hist));
    }
    for &window in obv_windows {
        out.push((
            format!("obv_flow_{window}"),
            calculate_obv_flow(&close_nonzero, volume, window),
        ));
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn tushare_factor_features(
    columns: &HashMap<String, Vec<f64>>,
    free_turnover_windows: &[usize],
    limit_stat_windows: &[usize],
    amplitude_windows: &[usize],
    pct_chg_windows: &[usize],
    ratio_change_windows: &[usize],
    valuation_change_windows: &[usize],
    industry_windows: &[usize],
    relative_industry_windows: &[usize],
    zscore_window: usize,
    pandas_float32_mode: bool,
) -> Result<Vec<(String, Vec<f64>)>, String> {
    let close_source = columns
        .get("close")
        .ok_or_else(|| "tushare_factor_features requires close column".to_owned())?;
    let len = close_source.len();
    for (name, values) in columns {
        if values.len() != len {
            return Err(format!(
                "tushare_factor_features length mismatch: close has {}, {name} has {}",
                len,
                values.len()
            ));
        }
    }

    let flow_windows = sorted_unique_windows(free_turnover_windows);
    let valuation_windows = sorted_unique_windows(valuation_change_windows);
    let industry_windows = sorted_unique_windows(industry_windows);
    let relative_industry_windows = sorted_unique_windows(relative_industry_windows);
    if flow_windows.is_empty() {
        return Err(
            "tushare_factor_features requires at least one free turnover window".to_owned(),
        );
    }
    if industry_windows.len() < 2 {
        return Err("tushare_factor_features requires at least two industry windows".to_owned());
    }
    let short_flow_window = flow_windows[0];
    let long_flow_window = *flow_windows.last().expect("flow windows checked");
    let short_industry_window = industry_windows[0];
    let mid_industry_window = industry_windows[industry_windows.len() - 2];
    let long_industry_window = *industry_windows.last().expect("industry windows checked");
    let zscore_window = zscore_window.max(1);

    let column = |name: &str| get_tushare_column(columns, len, name);
    let close = replace_zero_with_nan(close_source);
    let amount = column("amount");
    let up_limit = column("up_limit");
    let down_limit = column("down_limit");
    let limit_pre_close = replace_zero_with_nan(&get_tushare_column_with_fallback(
        columns,
        len,
        "limit_pre_close",
        "pre_close",
    ));
    let total_share = column("total_share");
    let circ_share = column("circ_share");
    let free_share = column("free_share");
    let turnover = column("turnover");
    let turnover_free = column("turnover_free");
    let volume_ratio = column("volume_ratio");
    let total_mv = column("total_mv");
    let circ_mv = column("circ_mv");
    let pb = column("pb");
    let pe = column("pe");
    let pe_ttm = column("pe_ttm");
    let ps = column("ps");
    let ps_ttm = column("ps_ttm");
    let dv_ratio = column("dv_ratio");
    let dv_ttm = column("dv_ttm");
    let amplitude = column("amplitude");
    let pct_chg = column("pct_chg");

    let fi_eps = column("fi_eps");
    let fi_dt_eps = column("fi_dt_eps");
    let fi_bps = column("fi_bps");
    let fi_ocfps = column("fi_ocfps");
    let fi_roe = column("fi_roe");
    let fi_roe_dt = column("fi_roe_dt");
    let fi_roa = column("fi_roa");
    let fi_gpm = column("fi_grossprofit_margin");
    let fi_npm = column("fi_netprofit_margin");
    let fi_debt_to_assets = column("fi_debt_to_assets");
    let fi_q_eps = column("fi_q_eps");
    let fi_q_dtprofit = column("fi_q_dtprofit");
    let fi_q_roe = column("fi_q_roe");
    let fi_q_dt_roe = column("fi_q_dt_roe");
    let fi_tr_yoy = column("fi_tr_yoy");
    let fi_or_yoy = column("fi_or_yoy");
    let fi_op_yoy = column("fi_op_yoy");
    let fi_netprofit_yoy = column("fi_netprofit_yoy");
    let fi_ocf_yoy = column("fi_ocf_yoy");
    let div_cash_div = column("div_cash_div");
    let div_cash_div_tax = column("div_cash_div_tax");
    let div_stk_div = column("div_stk_div");
    let div_stk_bo_rate = column("div_stk_bo_rate");
    let div_stk_co_rate = column("div_stk_co_rate");
    let div_base_share = column("div_base_share");
    let fc_p_change_min = column("fc_p_change_min");
    let fc_p_change_max = column("fc_p_change_max");
    let fc_net_profit_min = column("fc_net_profit_min");
    let fc_net_profit_max = column("fc_net_profit_max");
    let fc_last_parent_net = column("fc_last_parent_net");
    let fc_days_since_ann = column("fc_days_since_ann");
    let exp_revenue = column("exp_revenue");
    let exp_operate_profit = column("exp_operate_profit");
    let exp_total_profit = column("exp_total_profit");
    let exp_n_income = column("exp_n_income");
    let exp_total_assets = column("exp_total_assets");
    let exp_diluted_eps = column("exp_diluted_eps");
    let exp_diluted_roe = column("exp_diluted_roe");
    let exp_yoy_sales = column("exp_yoy_sales");
    let exp_yoy_op = column("exp_yoy_op");
    let exp_yoy_tp = column("exp_yoy_tp");
    let exp_yoy_dedu_np = column("exp_yoy_dedu_np");
    let exp_yoy_eps = column("exp_yoy_eps");
    let exp_yoy_roe = column("exp_yoy_roe");
    let exp_growth_assets = column("exp_growth_assets");
    let exp_yoy_assets = column("exp_yoy_assets");
    let exp_days_since_ann = column("exp_days_since_ann");
    let ind_member_count = column("ind_member_count");
    let ind_daily_ret = column("ind_daily_ret");
    let ind_excess_daily_ret = column("ind_excess_daily_ret");

    let mut out = Vec::new();
    macro_rules! push_feature {
        ($name:expr, $values:expr) => {{
            out.push((($name).to_string(), $values));
        }};
    }

    let gap_up_limit = sub_scalar(&safe_divide(&up_limit, &close), 1.0);
    let gap_down_limit = sub_scalar(&safe_divide(&close, &down_limit), 1.0);
    let limit_band_pct = safe_divide(
        &subtract(&up_limit, &down_limit),
        &add_scalar(&limit_pre_close, EPS),
    );
    let limit_band_pos = safe_divide(
        &subtract(&close, &down_limit),
        &add_scalar(&subtract(&up_limit, &down_limit), EPS),
    );
    let hit_up_limit = binary_map(&close, &up_limit, |close_value, limit_value| {
        if close_value.is_finite() && limit_value.is_finite() {
            if close_value >= limit_value * (1.0 - 1e-6) {
                1.0
            } else {
                0.0
            }
        } else {
            f64::NAN
        }
    });
    let hit_down_limit = binary_map(&close, &down_limit, |close_value, limit_value| {
        if close_value.is_finite() && limit_value.is_finite() {
            if close_value <= limit_value * (1.0 + 1e-6) {
                1.0
            } else {
                0.0
            }
        } else {
            f64::NAN
        }
    });
    push_feature!("gap_up_limit", gap_up_limit.clone());
    push_feature!("gap_down_limit", gap_down_limit.clone());
    push_feature!("limit_band_pct", limit_band_pct.clone());
    push_feature!("limit_band_pos", limit_band_pos.clone());
    push_feature!("hit_up_limit", hit_up_limit.clone());
    push_feature!("hit_down_limit", hit_down_limit.clone());

    let total_share_safe = replace_zero_with_nan(&total_share);
    let circ_share_safe = replace_zero_with_nan(&circ_share);
    let free_float_ratio = safe_divide(&free_share, &total_share_safe);
    let circ_float_ratio = safe_divide(&circ_share, &total_share_safe);
    let free_to_circ_ratio = safe_divide(&free_share, &circ_share_safe);
    let turnover_safe = replace_zero_with_nan(&turnover);
    let mut free_turnover_ratio = safe_divide(&turnover_free, &turnover_safe);
    let mut free_turnover_spread = subtract(&turnover_free, &turnover);
    if pandas_float32_mode {
        free_turnover_ratio = round_to_f32(&free_turnover_ratio);
        free_turnover_spread = round_to_f32(&free_turnover_spread);
    }
    push_feature!("free_float_ratio", free_float_ratio.clone());
    push_feature!("circ_float_ratio", circ_float_ratio);
    push_feature!("free_to_circ_ratio", free_to_circ_ratio.clone());
    push_feature!("free_turnover_ratio", free_turnover_ratio.clone());
    push_feature!("free_turnover_spread", free_turnover_spread.clone());
    for &window in free_turnover_windows {
        push_feature!(
            format!("free_turnover_mean_{window}"),
            rolling_mean(&turnover_free, window)
        );
    }
    let turnover_short_mean = rolling_mean(&turnover, short_flow_window);
    let turnover_long_mean = rolling_mean(&turnover, long_flow_window);
    let free_turnover_short_mean = rolling_mean(&turnover_free, short_flow_window);
    let free_turnover_long_mean = rolling_mean(&turnover_free, long_flow_window);
    let turnover_accel = sub_scalar(
        &safe_divide(&turnover_short_mean, &add_scalar(&turnover_long_mean, EPS)),
        1.0,
    );
    let free_turnover_accel = sub_scalar(
        &safe_divide(
            &free_turnover_short_mean,
            &add_scalar(&free_turnover_long_mean, EPS),
        ),
        1.0,
    );
    push_feature!(
        format!("turnover_accel_{short_flow_window}_{long_flow_window}"),
        turnover_accel
    );
    push_feature!(
        format!("free_turnover_accel_{short_flow_window}_{long_flow_window}"),
        free_turnover_accel
    );
    push_feature!("volume_ratio_raw", volume_ratio.clone());
    let total_mv_safe = replace_zero_with_nan(&total_mv);
    let float_mv_ratio = safe_divide(&circ_mv, &total_mv_safe);
    push_feature!("float_mv_ratio", float_mv_ratio.clone());

    let ep_clean = positive_inverse(&pe);
    let sp_clean = positive_inverse(&ps);
    let sp_ttm_clean = positive_inverse(&ps_ttm);
    let ep_ttm_clean = positive_inverse(&pe_ttm);
    let bp_clean = positive_inverse(&pb);
    let ep = fill_nan(&ep_clean, -1.0);
    let bp = fill_nan(&bp_clean, -1.0);
    let sp = fill_nan(&sp_clean, -1.0);
    let sp_ttm = fill_nan(&sp_ttm_clean, -1.0);
    let ep_ttm = fill_nan(&ep_ttm_clean, -1.0);
    push_feature!("ep", ep.clone());
    push_feature!("sp", sp.clone());
    push_feature!("sp_ttm", sp_ttm.clone());
    push_feature!("ep_ttm_gap", subtract(&ep, &ep_ttm));
    push_feature!("ep_clean", ep_clean.clone());
    push_feature!("ep_invalid", nonpositive_invalid_flag(&pe));
    push_feature!("sp_clean", sp_clean.clone());
    push_feature!("sp_invalid", nonpositive_invalid_flag(&ps));
    push_feature!("sp_ttm_clean", sp_ttm_clean.clone());
    push_feature!("sp_ttm_invalid", nonpositive_invalid_flag(&ps_ttm));
    push_feature!("ep_ttm_clean", ep_ttm_clean.clone());
    push_feature!("ep_ttm_invalid", nonpositive_invalid_flag(&pe_ttm));
    push_feature!("ep_ttm_gap_clean", subtract(&ep_clean, &ep_ttm_clean));
    push_feature!("bp_clean", bp_clean.clone());
    push_feature!("bp_invalid", nonpositive_invalid_flag(&pb));
    push_feature!("dividend_yield", dv_ratio.clone());
    push_feature!("dividend_yield_ttm", dv_ttm.clone());
    let has_dividend = dv_ttm
        .iter()
        .map(|value| {
            if value.is_finite() {
                if *value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    push_feature!("has_dividend", has_dividend);
    let close_positive = close
        .iter()
        .map(|value| if *value > EPS { *value } else { f64::NAN })
        .collect::<Vec<_>>();
    let dividend_cash_yield_proxy = safe_divide(&div_cash_div, &close_positive);
    let dividend_cash_to_eps = positive_safe_ratio(&div_cash_div, &fi_eps);
    let dividend_cash_to_ocfps = positive_safe_ratio(&div_cash_div, &fi_ocfps);
    push_feature!("dividend_cash_to_eps", dividend_cash_to_eps.clone());
    push_feature!("dividend_cash_to_ocfps", dividend_cash_to_ocfps.clone());
    push_feature!(
        "dividend_cash_yield_proxy",
        dividend_cash_yield_proxy.clone()
    );
    push_feature!("industry_member_count", ind_member_count);
    push_feature!("industry_daily_ret", ind_daily_ret);
    push_feature!("industry_excess_daily_ret", ind_excess_daily_ret);

    let mut hit_up_limit_count_by_window: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut hit_down_limit_count_by_window: HashMap<usize, Vec<f64>> = HashMap::new();
    for &window in limit_stat_windows {
        push_feature!(
            format!("limit_band_pct_mean_{window}"),
            rolling_mean(&limit_band_pct, window)
        );
        push_feature!(
            format!("limit_band_pos_mean_{window}"),
            rolling_mean(&limit_band_pos, window)
        );
        push_feature!(
            format!("gap_up_limit_mean_{window}"),
            rolling_mean(&gap_up_limit, window)
        );
        push_feature!(
            format!("gap_down_limit_mean_{window}"),
            rolling_mean(&gap_down_limit, window)
        );
        let hit_up_count = rolling_sum_min_count(&hit_up_limit, window, 1);
        let hit_down_count = rolling_sum_min_count(&hit_down_limit, window, 1);
        hit_up_limit_count_by_window.insert(window, hit_up_count.clone());
        hit_down_limit_count_by_window.insert(window, hit_down_count.clone());
        push_feature!(format!("hit_up_limit_count_{window}"), hit_up_count);
        push_feature!(format!("hit_down_limit_count_{window}"), hit_down_count);
    }
    for &window in amplitude_windows {
        push_feature!(
            format!("amplitude_mean_{window}"),
            rolling_mean(&amplitude, window)
        );
    }
    for &window in pct_chg_windows {
        push_feature!(
            format!("pct_chg_mean_{window}"),
            rolling_mean(&pct_chg, window)
        );
    }
    for &window in ratio_change_windows {
        push_feature!(
            format!("free_float_ratio_change_{window}"),
            pct_change(&free_float_ratio, window)
        );
        push_feature!(
            format!("free_to_circ_ratio_change_{window}"),
            pct_change(&free_to_circ_ratio, window)
        );
        push_feature!(
            format!("float_mv_ratio_change_{window}"),
            pct_change(&float_mv_ratio, window)
        );
    }
    for &window in valuation_change_windows {
        push_feature!(
            format!("sp_ttm_change_{window}"),
            pct_change(&sp_ttm, window)
        );
        push_feature!(
            format!("sp_ttm_clean_change_{window}"),
            pct_change(&sp_ttm_clean, window)
        );
        push_feature!(
            format!("dividend_yield_ttm_change_{window}"),
            pct_change(&dv_ttm, window)
        );
    }
    for &window in &valuation_windows {
        push_feature!(
            format!("ep_ttm_change_{window}"),
            diff_window(&ep_ttm, window)
        );
        push_feature!(
            format!("ep_ttm_clean_change_{window}"),
            diff_window(&ep_ttm_clean, window)
        );
        push_feature!(format!("bp_change_{window}"), diff_window(&bp, window));
        push_feature!(
            format!("bp_clean_change_{window}"),
            diff_window(&bp_clean, window)
        );
    }
    let own_daily_ret = pct_change(&close, 1);
    let mut stock_vs_industry_std_ratio_by_window: HashMap<usize, Vec<f64>> = HashMap::new();
    for &window in &industry_windows {
        let industry_ret = column(&format!("ind_ret_{window}"));
        let industry_std = column(&format!("ind_std_{window}"));
        let industry_excess_ret = column(&format!("ind_excess_ret_{window}"));
        let industry_pos_rate = column(&format!("ind_pos_rate_{window}"));
        let industry_dispersion = column(&format!("ind_dispersion_{window}"));
        let own_ret = pct_change(&close, window);
        let own_std = rolling_std(&own_daily_ret, window);
        let std_ratio = safe_divide(&own_std, &add_scalar(&industry_std, EPS));
        stock_vs_industry_std_ratio_by_window.insert(window, std_ratio.clone());
        push_feature!(format!("industry_ret_{window}"), industry_ret.clone());
        push_feature!(format!("industry_std_{window}"), industry_std);
        push_feature!(format!("industry_excess_ret_{window}"), industry_excess_ret);
        push_feature!(
            format!("industry_rel_ret_{window}"),
            subtract(&own_ret, &industry_ret)
        );
        push_feature!(format!("industry_pos_rate_{window}"), industry_pos_rate);
        push_feature!(format!("industry_dispersion_{window}"), industry_dispersion);
        push_feature!(format!("stock_vs_industry_std_ratio_{window}"), std_ratio);
    }

    let amplitude_mean = rolling_mean(&amplitude, zscore_window);
    let amplitude_std = rolling_std(&amplitude, zscore_window);
    let pct_chg_mean = rolling_mean(&pct_chg, zscore_window);
    let pct_chg_std = rolling_std(&pct_chg, zscore_window);
    let free_turnover_mean = rolling_mean(&free_turnover_ratio, zscore_window);
    let free_turnover_std = rolling_std(&free_turnover_ratio, zscore_window);
    let free_turnover_spread_mean = rolling_mean(&free_turnover_spread, zscore_window);
    let free_turnover_spread_std = rolling_std(&free_turnover_spread, zscore_window);
    let volume_ratio_mean = rolling_mean(&volume_ratio, zscore_window);
    let volume_ratio_std = rolling_std(&volume_ratio, zscore_window);
    let dividend_yield_ttm_baseline = shift(&rolling_mean(&dv_ttm, zscore_window), 1);
    let dividend_cash_yield_baseline =
        shift(&rolling_mean(&dividend_cash_yield_proxy, zscore_window), 1);
    let daily_ret_abs = abs_values(&own_daily_ret);
    let amount_abs = abs_values(&amount);
    let amihud_values = safe_divide(&daily_ret_abs, &add_scalar(&amount_abs, EPS));
    let amihud_short = rolling_mean(&amihud_values, short_flow_window);
    let amihud_long = rolling_mean(&amihud_values, long_flow_window);
    let downside_daily_ret_abs = where_mask(&daily_ret_abs, &own_daily_ret, |value| value < 0.0);
    let downside_amihud_values =
        safe_divide(&downside_daily_ret_abs, &add_scalar(&amount_abs, EPS));
    let downside_amihud = rolling_mean(&downside_amihud_values, long_flow_window);
    push_feature!(
        format!("amplitude_zscore_{zscore_window}"),
        safe_divide(
            &subtract(&amplitude, &amplitude_mean),
            &add_scalar(&amplitude_std, EPS)
        )
    );
    push_feature!(
        format!("pct_chg_zscore_{zscore_window}"),
        safe_divide(
            &subtract(&pct_chg, &pct_chg_mean),
            &add_scalar(&pct_chg_std, EPS)
        )
    );
    push_feature!(
        format!("free_turnover_ratio_zscore_{zscore_window}"),
        safe_divide(
            &subtract(&free_turnover_ratio, &free_turnover_mean),
            &add_scalar(&free_turnover_std, EPS)
        )
    );
    push_feature!(
        format!("free_turnover_spread_zscore_{zscore_window}"),
        safe_divide(
            &subtract(&free_turnover_spread, &free_turnover_spread_mean),
            &add_scalar(&free_turnover_spread_std, EPS)
        )
    );
    push_feature!(
        format!("volume_ratio_raw_zscore_{zscore_window}"),
        safe_divide(
            &subtract(&volume_ratio, &volume_ratio_mean),
            &add_scalar(&volume_ratio_std, EPS)
        )
    );
    push_feature!(
        format!("amihud_term_{short_flow_window}_{long_flow_window}"),
        sub_scalar(
            &safe_divide(&amihud_short, &add_scalar(&amihud_long, EPS)),
            1.0
        )
    );
    push_feature!(
        format!("downside_amihud_{long_flow_window}"),
        downside_amihud.clone()
    );
    push_feature!(
        format!("dividend_yield_ttm_surprise_{zscore_window}"),
        subtract(&dv_ttm, &dividend_yield_ttm_baseline)
    );
    push_feature!(
        format!("dividend_cash_yield_proxy_surprise_{zscore_window}"),
        subtract(&dividend_cash_yield_proxy, &dividend_cash_yield_baseline)
    );

    let fi_ocf_to_eps = safe_ratio_abs_denominator(&fi_ocfps, &fi_eps);
    let fi_ocfps_minus_eps = subtract(&fi_ocfps, &fi_eps);
    let fi_ocf_yoy_minus_np_yoy = subtract(&fi_ocf_yoy, &fi_netprofit_yoy);
    let fi_roe_quality_gap = subtract(&fi_roe_dt, &fi_roe);
    let fi_q_roe_quality_gap = subtract(&fi_q_dt_roe, &fi_q_roe);
    let fi_margin_quality = subtract(&fi_gpm, &fi_npm);
    let fi_profitability_combo = add_many(&[&fi_roe_dt, &fi_roa, &fi_npm]);
    let fi_growth_quality_combo =
        add_many(&[&fi_or_yoy, &fi_op_yoy, &fi_netprofit_yoy, &fi_ocf_yoy]);
    push_feature!("fi_ocf_to_eps", fi_ocf_to_eps.clone());
    push_feature!("fi_ocfps_minus_eps", fi_ocfps_minus_eps.clone());
    push_feature!("fi_ocf_yoy_minus_np_yoy", fi_ocf_yoy_minus_np_yoy.clone());
    push_feature!("fi_roe_quality_gap", fi_roe_quality_gap.clone());
    push_feature!("fi_q_roe_quality_gap", fi_q_roe_quality_gap);
    push_feature!("fi_margin_quality", fi_margin_quality.clone());
    push_feature!("fi_profitability_combo", fi_profitability_combo);
    push_feature!("fi_growth_quality_combo", fi_growth_quality_combo);

    let fc_p_change_mid = scalar_mul(&add(&fc_p_change_min, &fc_p_change_max), 0.5);
    let fc_p_change_width = subtract(&fc_p_change_max, &fc_p_change_min);
    let fc_net_profit_mid = scalar_mul(&add(&fc_net_profit_min, &fc_net_profit_max), 0.5);
    let fc_net_profit_width = subtract(&fc_net_profit_max, &fc_net_profit_min);
    let fc_freshness_weight = exp_neg_clipped_days(&fc_days_since_ann, 20.0);
    let fc_positive_confidence = fc_positive_confidence_values(
        &fc_p_change_min,
        &fc_p_change_max,
        &fc_p_change_mid,
        &fc_p_change_width,
    );
    push_feature!("fc_p_change_mid", fc_p_change_mid.clone());
    push_feature!("fc_p_change_width", fc_p_change_width.clone());
    push_feature!("fc_net_profit_mid", fc_net_profit_mid.clone());
    push_feature!("fc_net_profit_width", fc_net_profit_width);
    push_feature!(
        "fc_net_profit_mid_ratio",
        safe_ratio_abs_denominator(&fc_net_profit_mid, &fc_last_parent_net)
    );
    push_feature!("fc_positive_confidence", fc_positive_confidence.clone());
    push_feature!("fc_days_since_ann", fc_days_since_ann.clone());
    push_feature!("fc_freshness_weight", fc_freshness_weight.clone());
    push_feature!(
        "fc_surprise_fresh",
        multiply(&fc_p_change_mid, &fc_freshness_weight)
    );

    let exp_growth_combo = add_many(&[&exp_yoy_sales, &exp_yoy_op, &exp_yoy_dedu_np, &exp_yoy_eps]);
    let exp_profit_quality_gap = subtract(&exp_yoy_dedu_np, &exp_yoy_tp);
    let exp_asset_efficiency = subtract(&exp_yoy_sales, &exp_growth_assets);
    let exp_freshness_weight = exp_neg_clipped_days(&exp_days_since_ann, 20.0);
    push_feature!("exp_growth_combo", exp_growth_combo.clone());
    push_feature!("exp_profit_quality_gap", exp_profit_quality_gap.clone());
    push_feature!("exp_asset_efficiency", exp_asset_efficiency);
    push_feature!(
        "exp_profit_margin_proxy",
        safe_divide(&exp_n_income, &add_scalar(&abs_values(&exp_revenue), EPS))
    );
    push_feature!(
        "exp_op_margin_proxy",
        safe_divide(
            &exp_operate_profit,
            &add_scalar(&abs_values(&exp_revenue), EPS)
        )
    );
    push_feature!("exp_days_since_ann", exp_days_since_ann.clone());
    push_feature!("exp_freshness_weight", exp_freshness_weight.clone());
    push_feature!(
        "exp_growth_fresh",
        multiply(&exp_growth_combo, &exp_freshness_weight)
    );

    let up_turnover = where_mask(&turnover, &own_daily_ret, |value| value > 0.0);
    let down_turnover = where_mask(&turnover, &own_daily_ret, |value| value < 0.0);
    let up_turnover_sum = rolling_sum_min_count(&up_turnover, long_flow_window, 1);
    let down_turnover_sum = rolling_sum_min_count(&down_turnover, long_flow_window, 1);
    let total_turnover_sum = rolling_sum_min_count(&turnover, long_flow_window, 1);
    let up_down_turnover_ratio =
        safe_divide(&up_turnover_sum, &add_scalar(&down_turnover_sum, EPS));
    let return_turnover_corr = rolling_corr(&own_daily_ret, &turnover, long_flow_window);
    let downside_turnover_pressure =
        safe_divide(&down_turnover_sum, &add_scalar(&total_turnover_sum, EPS));
    let days_since_last_up_limit = event_age(&hit_up_limit);
    let days_since_last_down_limit = event_age(&hit_down_limit);
    let up_limit_streak = event_streak(&hit_up_limit);
    let down_limit_streak = event_streak(&hit_down_limit);
    let near_up_limit = binary_map(&limit_band_pos, &hit_up_limit, |pos, hit| {
        if pos >= 0.9 && hit <= 0.0 {
            1.0
        } else {
            0.0
        }
    });
    let near_down_limit = binary_map(&limit_band_pos, &hit_down_limit, |pos, hit| {
        if pos <= 0.1 && hit <= 0.0 {
            1.0
        } else {
            0.0
        }
    });
    let near_up_limit_count_long = rolling_sum_min_count(&near_up_limit, long_flow_window, 1);
    let near_down_limit_count_long = rolling_sum_min_count(&near_down_limit, long_flow_window, 1);
    push_feature!(
        format!("up_down_turnover_ratio_{long_flow_window}"),
        up_down_turnover_ratio.clone()
    );
    push_feature!(
        format!("return_turnover_corr_{long_flow_window}"),
        return_turnover_corr.clone()
    );
    push_feature!(
        format!("downside_turnover_pressure_{long_flow_window}"),
        downside_turnover_pressure.clone()
    );
    push_feature!("days_since_last_up_limit", days_since_last_up_limit);
    push_feature!("days_since_last_down_limit", days_since_last_down_limit);
    push_feature!("up_limit_streak", up_limit_streak);
    push_feature!("down_limit_streak", down_limit_streak);
    push_feature!(
        format!("near_up_limit_count_{long_flow_window}"),
        near_up_limit_count_long
    );
    push_feature!(
        format!("near_down_limit_count_{long_flow_window}"),
        near_down_limit_count_long.clone()
    );

    let industry_pos_mid = column(&format!("ind_pos_rate_{mid_industry_window}"));
    let industry_pos_long = column(&format!("ind_pos_rate_{long_industry_window}"));
    let industry_ret_short = column(&format!("ind_ret_{short_industry_window}"));
    let industry_ret_mid = column(&format!("ind_ret_{mid_industry_window}"));
    let industry_ret_long = column(&format!("ind_ret_{long_industry_window}"));
    let own_ret_short = pct_change(&close, short_industry_window);
    let own_ret_mid = pct_change(&close, mid_industry_window);
    let own_ret_long = pct_change(&close, long_industry_window);
    let rel_ret_short = subtract(&own_ret_short, &industry_ret_short);
    let rel_ret_mid = subtract(&own_ret_mid, &industry_ret_mid);
    let rel_ret_long = subtract(&own_ret_long, &industry_ret_long);
    push_feature!(
        format!("industry_breadth_accel_{mid_industry_window}_{long_industry_window}"),
        subtract(&industry_pos_mid, &industry_pos_long)
    );
    push_feature!(
        format!("industry_momentum_accel_{mid_industry_window}_{long_industry_window}"),
        subtract(&industry_ret_mid, &industry_ret_long)
    );
    push_feature!(
        format!("stock_industry_ret_gap_{short_industry_window}_{mid_industry_window}"),
        subtract(&rel_ret_short, &rel_ret_mid)
    );
    push_feature!(
        format!("stock_industry_ret_gap_{mid_industry_window}_{long_industry_window}"),
        subtract(&rel_ret_mid, &rel_ret_long)
    );
    push_feature!(
        format!("stock_relative_strength_quality_{mid_industry_window}"),
        safe_divide(
            &rel_ret_mid,
            &add_scalar(&column(&format!("ind_std_{mid_industry_window}")), EPS)
        )
    );
    push_feature!(
        format!("stock_relative_strength_quality_{long_industry_window}"),
        safe_divide(
            &rel_ret_long,
            &add_scalar(&column(&format!("ind_std_{long_industry_window}")), EPS)
        )
    );

    let mut stock_vs_industry_downside_amihud_ratio_by_window: HashMap<usize, Vec<f64>> =
        HashMap::new();
    for &window in &relative_industry_windows {
        let own_turnover_mean = rolling_mean(&turnover, window);
        let own_free_turnover_mean = rolling_mean(&turnover_free, window);
        let own_volume_ratio_mean = rolling_mean(&volume_ratio, window);
        let own_amihud_mean = rolling_mean(&amihud_values, window);
        let own_downside_amihud_mean = rolling_mean(&downside_amihud_values, window);
        let own_amplitude_mean = rolling_mean(&amplitude, window);
        let own_hit_up_limit_rate = rolling_mean(&hit_up_limit, window);
        let own_hit_down_limit_rate = rolling_mean(&hit_down_limit, window);
        let turnover_ratio = safe_divide(
            &own_turnover_mean,
            &add_scalar(&column(&format!("ind_turnover_mean_{window}")), EPS),
        );
        let free_turnover_ratio_relative = safe_divide(
            &own_free_turnover_mean,
            &add_scalar(&column(&format!("ind_free_turnover_mean_{window}")), EPS),
        );
        let volume_ratio_gap = subtract(
            &own_volume_ratio_mean,
            &column(&format!("ind_volume_ratio_mean_{window}")),
        );
        let amihud_ratio = safe_divide(
            &own_amihud_mean,
            &add_scalar(&column(&format!("ind_amihud_mean_{window}")), EPS),
        );
        let downside_amihud_ratio = safe_divide(
            &own_downside_amihud_mean,
            &add_scalar(&column(&format!("ind_downside_amihud_mean_{window}")), EPS),
        );
        let amplitude_ratio = safe_divide(
            &own_amplitude_mean,
            &add_scalar(&column(&format!("ind_amplitude_mean_{window}")), EPS),
        );
        let hit_up_limit_gap = subtract(
            &own_hit_up_limit_rate,
            &column(&format!("ind_hit_up_limit_rate_{window}")),
        );
        let hit_down_limit_gap = subtract(
            &own_hit_down_limit_rate,
            &column(&format!("ind_hit_down_limit_rate_{window}")),
        );
        let std_ratio = stock_vs_industry_std_ratio_by_window
            .get(&window)
            .cloned()
            .unwrap_or_else(|| vec![f64::NAN; len]);
        let stock_vs_industry_crowding = weighted_observed_sum(
            &[
                (0.25, bounded_signal(&sub_scalar(&turnover_ratio, 1.0), 1.0)),
                (
                    0.25,
                    bounded_signal(&sub_scalar(&free_turnover_ratio_relative, 1.0), 1.0),
                ),
                (0.20, bounded_signal(&volume_ratio_gap, 2.0)),
                (
                    0.20,
                    bounded_signal(&sub_scalar(&amplitude_ratio, 1.0), 1.0),
                ),
                (
                    0.10,
                    bounded_signal(&subtract(&hit_up_limit_gap, &hit_down_limit_gap), 0.50),
                ),
            ],
            len,
            0.5,
        );
        let stock_vs_industry_liquidity_stress = weighted_observed_sum(
            &[
                (
                    0.45,
                    bounded_signal(&sub_scalar(&downside_amihud_ratio, 1.0), 1.0),
                ),
                (0.25, bounded_signal(&sub_scalar(&amihud_ratio, 1.0), 1.0)),
                (
                    0.20,
                    bounded_signal(&sub_scalar(&amplitude_ratio, 1.0), 1.0),
                ),
                (0.10, bounded_signal(&hit_down_limit_gap, 0.50)),
            ],
            len,
            0.5,
        );
        let stock_vs_industry_low_vol_liquidity = weighted_observed_sum(
            &[
                (-0.45, bounded_signal(&sub_scalar(&std_ratio, 1.0), 0.50)),
                (-0.25, bounded_signal(&sub_scalar(&amihud_ratio, 1.0), 1.0)),
                (
                    -0.15,
                    bounded_signal(&sub_scalar(&downside_amihud_ratio, 1.0), 1.0),
                ),
                (
                    0.15,
                    bounded_signal(&sub_scalar(&free_turnover_ratio_relative, 1.0), 1.0),
                ),
            ],
            len,
            0.5,
        );
        stock_vs_industry_downside_amihud_ratio_by_window
            .insert(window, downside_amihud_ratio.clone());
        push_feature!(
            format!("stock_vs_industry_turnover_ratio_{window}"),
            turnover_ratio
        );
        push_feature!(
            format!("stock_vs_industry_free_turnover_ratio_{window}"),
            free_turnover_ratio_relative
        );
        push_feature!(
            format!("stock_vs_industry_volume_ratio_gap_{window}"),
            volume_ratio_gap
        );
        push_feature!(
            format!("stock_vs_industry_amihud_ratio_{window}"),
            amihud_ratio
        );
        push_feature!(
            format!("stock_vs_industry_downside_amihud_ratio_{window}"),
            downside_amihud_ratio
        );
        push_feature!(
            format!("stock_vs_industry_amplitude_ratio_{window}"),
            amplitude_ratio
        );
        push_feature!(
            format!("stock_vs_industry_hit_up_limit_gap_{window}"),
            hit_up_limit_gap
        );
        push_feature!(
            format!("stock_vs_industry_hit_down_limit_gap_{window}"),
            hit_down_limit_gap
        );
        push_feature!(
            format!("stock_vs_industry_crowding_{window}"),
            stock_vs_industry_crowding
        );
        push_feature!(
            format!("stock_vs_industry_liquidity_stress_{window}"),
            stock_vs_industry_liquidity_stress
        );
        push_feature!(
            format!("stock_vs_industry_low_vol_liquidity_{window}"),
            stock_vs_industry_low_vol_liquidity
        );
    }

    let dividend_yield_ttm_minus_industry =
        subtract(&dv_ttm, &column("ind_dividend_yield_ttm_mean"));
    let dividend_cash_yield_rel = subtract(
        &dividend_cash_yield_proxy,
        &column("ind_dividend_cash_yield_proxy_mean"),
    );
    push_feature!(
        "ep_minus_industry_ep",
        subtract(&ep, &column("ind_ep_mean"))
    );
    push_feature!(
        "sp_minus_industry_sp",
        subtract(&sp, &column("ind_sp_mean"))
    );
    push_feature!(
        "sp_ttm_minus_industry_sp_ttm",
        subtract(&sp_ttm, &column("ind_sp_ttm_mean"))
    );
    push_feature!(
        "bp_minus_industry_bp",
        subtract(&bp, &column("ind_bp_mean"))
    );
    push_feature!(
        "ep_clean_minus_industry_ep_clean",
        subtract(&ep_clean, &column("ind_ep_clean_mean"))
    );
    push_feature!(
        "sp_clean_minus_industry_sp_clean",
        subtract(&sp_clean, &column("ind_sp_clean_mean"))
    );
    push_feature!(
        "sp_ttm_clean_minus_industry_sp_ttm_clean",
        subtract(&sp_ttm_clean, &column("ind_sp_ttm_clean_mean"))
    );
    push_feature!(
        "bp_clean_minus_industry_bp_clean",
        subtract(&bp_clean, &column("ind_bp_clean_mean"))
    );
    push_feature!(
        "dividend_yield_minus_industry",
        subtract(&dv_ratio, &column("ind_dividend_yield_mean"))
    );
    push_feature!(
        "dividend_yield_ttm_minus_industry",
        dividend_yield_ttm_minus_industry.clone()
    );
    push_feature!(
        "dividend_cash_to_eps_minus_industry",
        subtract(
            &dividend_cash_to_eps,
            &column("ind_dividend_cash_to_eps_mean")
        )
    );
    push_feature!(
        "dividend_cash_to_ocfps_minus_industry",
        subtract(
            &dividend_cash_to_ocfps,
            &column("ind_dividend_cash_to_ocfps_mean")
        )
    );
    push_feature!(
        "dividend_cash_yield_proxy_minus_industry",
        dividend_cash_yield_rel.clone()
    );
    let dividend_yield_spread_mean =
        rolling_mean(&dividend_yield_ttm_minus_industry, zscore_window);
    let dividend_yield_spread_std = rolling_std(&dividend_yield_ttm_minus_industry, zscore_window);
    let dividend_cash_yield_rel_mean = rolling_mean(&dividend_cash_yield_rel, zscore_window);
    let dividend_cash_yield_rel_std = rolling_std(&dividend_cash_yield_rel, zscore_window);
    push_feature!(
        format!("dividend_yield_ttm_industry_spread_zscore_{zscore_window}"),
        safe_divide(
            &subtract(
                &dividend_yield_ttm_minus_industry,
                &dividend_yield_spread_mean
            ),
            &add_scalar(&dividend_yield_spread_std, EPS)
        )
    );
    push_feature!(
        format!("dividend_cash_yield_industry_spread_zscore_{zscore_window}"),
        safe_divide(
            &subtract(&dividend_cash_yield_rel, &dividend_cash_yield_rel_mean),
            &add_scalar(&dividend_cash_yield_rel_std, EPS)
        )
    );
    let ocf_rel = subtract(&fi_ocf_to_eps, &column("ind_fi_ocf_to_eps_mean"));
    let margin_rel = subtract(&fi_margin_quality, &column("ind_fi_margin_quality_mean"));
    push_feature!("fi_ocf_to_eps_minus_industry", ocf_rel.clone());
    push_feature!(
        "fi_ocfps_minus_eps_minus_industry",
        subtract(&fi_ocfps_minus_eps, &column("ind_fi_ocfps_minus_eps_mean"))
    );
    push_feature!(
        "fi_roe_quality_gap_minus_industry",
        subtract(&fi_roe_quality_gap, &column("ind_fi_roe_quality_gap_mean"))
    );
    push_feature!("fi_margin_quality_minus_industry", margin_rel.clone());

    let bp_rel = subtract(&bp_clean, &column("ind_bp_clean_mean"));
    let sp_ttm_rel = subtract(&sp_ttm_clean, &column("ind_sp_ttm_clean_mean"));
    let dividend_rel = dividend_yield_ttm_minus_industry.clone();
    let quality_gate = weighted_observed_sum(
        &[
            (0.35, bounded_signal(&fi_roe_dt, 20.0)),
            (0.25, bounded_signal(&ocf_rel, 2.0)),
            (0.20, bounded_signal(&fi_ocf_yoy_minus_np_yoy, 50.0)),
            (0.10, bounded_signal(&margin_rel, 20.0)),
            (-0.20, bounded_signal(&fi_debt_to_assets, 60.0)),
        ],
        len,
        0.5,
    );
    let value_rel = weighted_observed_sum(
        &[
            (1.00, bp_rel),
            (1.00, sp_ttm_rel),
            (0.25, bounded_signal(&dividend_rel, 5.0)),
        ],
        len,
        0.5,
    );
    let growth_cash_quality = weighted_observed_sum(
        &[
            (0.30, bounded_signal(&fi_or_yoy, 50.0)),
            (0.25, bounded_signal(&fi_op_yoy, 50.0)),
            (0.30, bounded_signal(&fi_ocf_yoy, 50.0)),
            (0.15, bounded_signal(&fi_netprofit_yoy, 50.0)),
            (
                -0.25,
                bounded_signal(&abs_values(&subtract(&fi_netprofit_yoy, &fi_ocf_yoy)), 50.0),
            ),
        ],
        len,
        0.5,
    );
    let forecast_confidence = bounded_signal(&fc_positive_confidence, 1.0);
    let forecast_fresh_strength = multiply(
        &multiply(
            &bounded_signal(&fc_p_change_mid, 50.0),
            &forecast_confidence,
        ),
        &fc_freshness_weight,
    );
    let express_quality = weighted_observed_sum(
        &[
            (0.35, bounded_signal(&exp_yoy_sales, 50.0)),
            (0.35, bounded_signal(&exp_yoy_dedu_np, 50.0)),
            (0.20, bounded_signal(&exp_yoy_eps, 50.0)),
            (0.10, bounded_signal(&exp_profit_quality_gap, 30.0)),
            (-0.15, bounded_signal(&exp_growth_assets, 50.0)),
        ],
        len,
        0.5,
    );
    let trend_quality_mid = weighted_observed_sum(
        &[
            (0.35, bounded_signal(&industry_ret_mid, 0.20)),
            (
                0.25,
                bounded_signal(
                    &column(&format!("ind_excess_ret_{mid_industry_window}")),
                    0.10,
                ),
            ),
            (
                0.25,
                bounded_signal(&sub_scalar(&industry_pos_mid, 0.50), 0.20),
            ),
            (
                -0.15,
                bounded_signal(&column(&format!("ind_std_{mid_industry_window}")), 0.05),
            ),
        ],
        len,
        0.5,
    );
    let trend_quality_long = weighted_observed_sum(
        &[
            (0.35, bounded_signal(&industry_ret_long, 0.30)),
            (
                0.25,
                bounded_signal(
                    &column(&format!("ind_excess_ret_{long_industry_window}")),
                    0.15,
                ),
            ),
            (
                0.25,
                bounded_signal(&sub_scalar(&industry_pos_long, 0.50), 0.20),
            ),
            (
                -0.15,
                bounded_signal(&column(&format!("ind_std_{long_industry_window}")), 0.08),
            ),
        ],
        len,
        0.5,
    );
    let stock_vol_rel_mid = stock_vs_industry_std_ratio_by_window
        .get(&mid_industry_window)
        .cloned()
        .unwrap_or_else(|| vec![f64::NAN; len]);
    let stock_vol_rel_long = stock_vs_industry_std_ratio_by_window
        .get(&long_industry_window)
        .cloned()
        .unwrap_or_else(|| vec![f64::NAN; len]);
    let liquidity_absorption = weighted_observed_sum(
        &[
            (
                1.00,
                bounded_signal(&sub_scalar(&up_down_turnover_ratio, 1.0), 2.0),
            ),
            (1.00, bounded_signal(&return_turnover_corr, 1.0)),
            (
                -1.00,
                bounded_signal(&sub_scalar(&downside_turnover_pressure, 0.50), 0.30),
            ),
        ],
        len,
        0.5,
    );
    let stock_vs_industry_downside_amihud_mid = stock_vs_industry_downside_amihud_ratio_by_window
        .get(&mid_industry_window)
        .cloned()
        .unwrap_or_else(|| vec![f64::NAN; len]);
    let downside_liquidity_relief = weighted_observed_sum(
        &[
            (
                -1.00,
                bounded_signal(
                    &sub_scalar(&stock_vs_industry_downside_amihud_mid, 1.0),
                    1.0,
                ),
            ),
            (
                -1.00,
                bounded_signal(&sub_scalar(&downside_turnover_pressure, 0.50), 0.30),
            ),
            (-1.00, bounded_signal(&near_down_limit_count_long, 3.0)),
        ],
        len,
        0.5,
    );
    let sem_value_quality = add(&value_rel, &quality_gate);
    push_feature!("sem_value_quality", sem_value_quality.clone());
    push_feature!(
        "sem_value_quality_low_vol",
        subtract(
            &sem_value_quality,
            &bounded_signal(&sub_scalar(&stock_vol_rel_mid, 1.0), 0.50)
        )
    );
    push_feature!(
        "sem_dividend_quality",
        multiply(
            &dividend_rel,
            &add_scalar(&clip_values(&quality_gate, -0.50, 0.75), 1.0)
        )
    );
    let payout_pressure = clip_lower_vec(
        &weighted_observed_sum(
            &[
                (
                    0.50,
                    bounded_signal(&sub_scalar(&dividend_cash_to_eps, 0.80), 0.50),
                ),
                (
                    0.50,
                    bounded_signal(&sub_scalar(&dividend_cash_to_ocfps, 0.60), 0.50),
                ),
            ],
            len,
            0.25,
        ),
        0.0,
    );
    let sem_dividend_cash_quality = weighted_observed_sum(
        &[
            (0.35, bounded_signal(&dividend_rel, 5.0)),
            (0.30, bounded_signal(&dividend_cash_yield_rel, 0.05)),
            (0.25, quality_gate.clone()),
            (-0.10, payout_pressure),
        ],
        len,
        0.5,
    );
    push_feature!(
        "sem_profitability_resilience",
        subtract(
            &quality_gate,
            &bounded_signal(&sub_scalar(&stock_vol_rel_mid, 1.0), 0.50)
        )
    );
    push_feature!("sem_growth_cash_quality", growth_cash_quality.clone());
    push_feature!(
        format!("sem_growth_cash_quality_accel_{zscore_window}"),
        subtract(
            &growth_cash_quality,
            &shift(&rolling_mean(&growth_cash_quality, zscore_window), 1)
        )
    );
    push_feature!(
        "sem_forecast_confidence_fresh",
        forecast_fresh_strength.clone()
    );
    push_feature!(
        format!("sem_forecast_unpriced_{short_industry_window}"),
        subtract(
            &forecast_fresh_strength,
            &bounded_signal(&own_ret_short, 0.10)
        )
    );
    push_feature!(
        "sem_express_growth_quality_fresh",
        multiply(&express_quality, &exp_freshness_weight)
    );
    push_feature!(
        format!("sem_industry_strength_low_vol_{mid_industry_window}"),
        subtract(
            &trend_quality_mid,
            &bounded_signal(&sub_scalar(&stock_vol_rel_mid, 1.0), 0.50)
        )
    );
    push_feature!(
        format!("sem_industry_strength_low_vol_{long_industry_window}"),
        subtract(
            &trend_quality_long,
            &bounded_signal(&sub_scalar(&stock_vol_rel_long, 1.0), 0.50)
        )
    );
    push_feature!(
        format!("sem_industry_relative_winner_{mid_industry_window}"),
        subtract(
            &add(&trend_quality_mid, &bounded_signal(&rel_ret_mid, 0.20)),
            &bounded_signal(&sub_scalar(&stock_vol_rel_mid, 1.0), 0.50)
        )
    );
    push_feature!(
        format!("sem_industry_pullback_recovery_{short_industry_window}_{mid_industry_window}"),
        subtract(
            &add(&trend_quality_mid, &bounded_signal(&rel_ret_mid, 0.20)),
            &bounded_signal(&rel_ret_short, 0.10)
        )
    );
    push_feature!(
        format!("sem_liquidity_absorption_{long_flow_window}"),
        liquidity_absorption
    );
    push_feature!(
        format!("sem_downside_liquidity_relief_{long_flow_window}"),
        downside_liquidity_relief
    );
    push_feature!(
        format!("sem_limit_breakout_quality_{long_flow_window}"),
        weighted_observed_sum(
            &[
                (
                    1.00,
                    bounded_signal(
                        &hit_up_limit_count_by_window
                            .get(&long_flow_window)
                            .cloned()
                            .unwrap_or_else(|| vec![f64::NAN; len]),
                        3.0
                    )
                ),
                (
                    -1.00,
                    bounded_signal(
                        &hit_down_limit_count_by_window
                            .get(&long_flow_window)
                            .cloned()
                            .unwrap_or_else(|| vec![f64::NAN; len]),
                        3.0
                    )
                ),
                (1.00, trend_quality_mid.clone()),
                (-1.00, bounded_signal(&near_down_limit_count_long, 3.0)),
            ],
            len,
            0.5,
        )
    );
    push_feature!(
        format!("sem_low_vol_value_reversal_{short_industry_window}"),
        subtract(
            &subtract(
                &sem_value_quality,
                &bounded_signal(&sub_scalar(&stock_vol_rel_mid, 1.0), 0.50)
            ),
            &bounded_signal(&own_ret_short, 0.10)
        )
    );
    push_feature!("sem_dividend_cash_quality", sem_dividend_cash_quality);

    push_feature!("latest_eps", fi_eps);
    push_feature!("latest_dt_eps", fi_dt_eps);
    push_feature!("latest_bps", fi_bps);
    push_feature!("latest_ocfps", fi_ocfps);
    push_feature!("latest_roe", fi_roe);
    push_feature!("latest_roe_dt", fi_roe_dt);
    push_feature!("latest_roa", fi_roa);
    push_feature!("latest_grossprofit_margin", fi_gpm);
    push_feature!("latest_netprofit_margin", fi_npm);
    push_feature!("latest_debt_to_assets", fi_debt_to_assets);
    push_feature!("latest_q_eps", fi_q_eps);
    push_feature!("latest_q_dtprofit", fi_q_dtprofit);
    push_feature!("latest_q_roe", fi_q_roe);
    push_feature!("latest_q_dt_roe", fi_q_dt_roe);
    push_feature!("latest_tr_yoy", fi_tr_yoy);
    push_feature!("latest_or_yoy", fi_or_yoy);
    push_feature!("latest_op_yoy", fi_op_yoy);
    push_feature!("latest_netprofit_yoy", fi_netprofit_yoy);
    push_feature!("latest_ocf_yoy", fi_ocf_yoy);
    push_feature!("latest_div_cash", div_cash_div.clone());
    push_feature!("latest_div_cash_tax", div_cash_div_tax);
    push_feature!("latest_div_stock", div_stk_div.clone());
    push_feature!("latest_div_bo_rate", div_stk_bo_rate.clone());
    push_feature!("latest_div_co_rate", div_stk_co_rate.clone());
    push_feature!("latest_div_base_share", div_base_share);
    push_feature!(
        "latest_div_cash_yield_proxy",
        safe_divide(&div_cash_div, &add_scalar(&close, EPS))
    );
    let latest_div_stock_ratio = add_many(&[&div_stk_div, &div_stk_bo_rate, &div_stk_co_rate]);
    push_feature!("latest_div_stock_ratio", latest_div_stock_ratio.clone());
    let has_stock_dividend = latest_div_stock_ratio
        .iter()
        .map(|value| {
            if value.is_finite() {
                if *value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    push_feature!("has_stock_dividend", has_stock_dividend);
    push_feature!("latest_fc_p_change_min", fc_p_change_min);
    push_feature!("latest_fc_p_change_max", fc_p_change_max);
    push_feature!("latest_fc_net_profit_min", fc_net_profit_min);
    push_feature!("latest_fc_net_profit_max", fc_net_profit_max);
    push_feature!("latest_fc_last_parent_net", fc_last_parent_net);
    push_feature!("latest_exp_revenue", exp_revenue);
    push_feature!("latest_exp_operate_profit", exp_operate_profit);
    push_feature!("latest_exp_total_profit", exp_total_profit);
    push_feature!("latest_exp_n_income", exp_n_income);
    push_feature!("latest_exp_total_assets", exp_total_assets);
    push_feature!("latest_exp_diluted_eps", exp_diluted_eps);
    push_feature!("latest_exp_diluted_roe", exp_diluted_roe);
    push_feature!("latest_exp_yoy_sales", exp_yoy_sales);
    push_feature!("latest_exp_yoy_op", exp_yoy_op);
    push_feature!("latest_exp_yoy_tp", exp_yoy_tp);
    push_feature!("latest_exp_yoy_dedu_np", exp_yoy_dedu_np);
    push_feature!("latest_exp_yoy_eps", exp_yoy_eps);
    push_feature!("latest_exp_yoy_roe", exp_yoy_roe);
    push_feature!("latest_exp_growth_assets", exp_growth_assets);
    push_feature!("latest_exp_yoy_assets", exp_yoy_assets);

    Ok(out)
}

fn shifted_ratio(numerator: &[f64], denominator: &[f64], window: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(numerator.len());
    for idx in 0..numerator.len() {
        if idx < window {
            out.push(f64::NAN);
        } else {
            out.push(numerator[idx - window] / denominator[idx]);
        }
    }
    out
}

fn shifted_ratio_eps(numerator: &[f64], denominator: &[f64], window: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(numerator.len());
    for idx in 0..numerator.len() {
        if idx < window {
            out.push(f64::NAN);
        } else {
            out.push(numerator[idx - window] / (denominator[idx] + EPS));
        }
    }
    out
}

fn shift(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        if idx < window {
            out.push(f64::NAN);
        } else {
            out.push(values[idx - window]);
        }
    }
    out
}

fn replace_zero_with_nan(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| if *value == 0.0 { f64::NAN } else { *value })
        .collect()
}

fn replace_zero_with_nan_owned(values: Vec<f64>) -> Vec<f64> {
    values
        .into_iter()
        .map(|value| if value == 0.0 { f64::NAN } else { value })
        .collect()
}

fn pct_change(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        if idx < window {
            out.push(f64::NAN);
        } else {
            out.push(values[idx] / values[idx - window] - 1.0);
        }
    }
    out
}

fn abs_values(values: &[f64]) -> Vec<f64> {
    values.iter().map(|value| value.abs()).collect()
}

fn add_scalar(values: &[f64], scalar: f64) -> Vec<f64> {
    values.iter().map(|value| *value + scalar).collect()
}

fn log_plus_one(values: &[f64]) -> Vec<f64> {
    values.iter().map(|value| (*value + 1.0).ln()).collect()
}

fn diff(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        if idx == 0 {
            out.push(f64::NAN);
        } else {
            out.push(values[idx] - values[idx - 1]);
        }
    }
    out
}

fn clip_lower(values: &[f64], lower: f64) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            if value.is_nan() {
                f64::NAN
            } else {
                value.max(lower)
            }
        })
        .collect()
}

fn rolling_sum_min_count(values: &[f64], window: usize, min_count: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in &values[start..=end] {
            if value.is_nan() {
                continue;
            }
            sum += *value;
            count += 1;
        }
        if count < min_count {
            out.push(f64::NAN);
        } else {
            out.push(sum);
        }
    }
    out
}

fn rolling_mean_min_count(values: &[f64], window: usize, min_count: usize) -> Vec<f64> {
    let sums = rolling_sum_min_count(values, window, min_count);
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        if sums[end].is_nan() {
            out.push(f64::NAN);
            continue;
        }
        let start = (end + 1).saturating_sub(window);
        let count = values[start..=end]
            .iter()
            .filter(|value| !value.is_nan())
            .count();
        out.push(sums[end] / count as f64);
    }
    out
}

fn rolling_std_min_count(values: &[f64], window: usize, min_count: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in &values[start..=end] {
            if value.is_nan() {
                continue;
            }
            sum += *value;
            count += 1;
        }
        if count < min_count || count < 2 {
            out.push(f64::NAN);
        } else {
            let count_f = count as f64;
            let mean = sum / count_f;
            let mut sum_sq_diff = 0.0;
            for value in &values[start..=end] {
                if value.is_nan() {
                    continue;
                }
                let diff = *value - mean;
                sum_sq_diff += diff * diff;
            }
            let variance = sum_sq_diff / (count_f - 1.0);
            out.push(variance.max(0.0).sqrt());
        }
    }
    out
}

fn rolling_extreme_min_count(
    values: &[f64],
    window: usize,
    min_count: usize,
    find_max: bool,
) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut chosen = f64::NAN;
        let mut count = 0usize;
        for value in &values[start..=end] {
            if value.is_nan() {
                continue;
            }
            count += 1;
            if chosen.is_nan() || (find_max && *value > chosen) || (!find_max && *value < chosen) {
                chosen = *value;
            }
        }
        if count < min_count {
            out.push(f64::NAN);
        } else {
            out.push(chosen);
        }
    }
    out
}

fn ewm_adjust_false(values: &[f64], alpha: f64, min_periods: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    let mut state = f64::NAN;
    let mut observed = 0usize;
    for value in values {
        if value.is_nan() {
            out.push(f64::NAN);
            continue;
        }
        observed += 1;
        if state.is_nan() {
            state = *value;
        } else {
            state = (1.0 - alpha) * state + alpha * *value;
        }
        if observed >= min_periods {
            out.push(state);
        } else {
            out.push(f64::NAN);
        }
    }
    out
}

fn calculate_macd(
    close: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema_fast = ewm_adjust_false(close, 2.0 / (fast as f64 + 1.0), fast);
    let ema_slow = ewm_adjust_false(close, 2.0 / (slow as f64 + 1.0), slow);
    let macd_line = ema_fast
        .iter()
        .zip(&ema_slow)
        .map(|(fast_value, slow_value)| *fast_value - *slow_value)
        .collect::<Vec<_>>();
    let signal_line = ewm_adjust_false(&macd_line, 2.0 / (signal as f64 + 1.0), signal);
    let hist = macd_line
        .iter()
        .zip(&signal_line)
        .map(|(macd_value, signal_value)| *macd_value - *signal_value)
        .collect();
    (macd_line, signal_line, hist)
}

fn calculate_rsi(close: &[f64], period: usize) -> Vec<f64> {
    let delta = diff(close);
    let gain = clip_lower(&delta, 0.0);
    let loss = clip_lower(&delta.iter().map(|value| -*value).collect::<Vec<_>>(), 0.0);
    let avg_gain = ewm_adjust_false(&gain, 1.0 / period as f64, period);
    let avg_loss = ewm_adjust_false(&loss, 1.0 / period as f64, period);
    avg_gain
        .iter()
        .zip(&avg_loss)
        .map(|(gain_value, loss_value)| {
            let rs = *gain_value / (*loss_value + EPS);
            let mut rsi = 100.0 - (100.0 / (1.0 + rs));
            if *loss_value <= EPS || loss_value.is_nan() {
                rsi = 100.0;
            }
            if (*gain_value <= EPS || gain_value.is_nan())
                && (*loss_value <= EPS || loss_value.is_nan())
            {
                rsi = 50.0;
            }
            rsi
        })
        .collect()
}

fn calculate_bollinger(
    close: &[f64],
    window: usize,
    num_std: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ma = rolling_mean_min_count(close, window, window);
    let std = rolling_std_min_count(close, window, window);
    let pos = close
        .iter()
        .zip(&ma)
        .zip(&std)
        .map(|((close_value, ma_value), std_value)| {
            (*close_value - *ma_value) / (num_std * *std_value + EPS)
        })
        .collect();
    let width = ma
        .iter()
        .zip(&std)
        .map(|(ma_value, std_value)| (2.0 * num_std * *std_value) / (ma_value.abs() + EPS))
        .collect();
    let zscore = close
        .iter()
        .zip(&ma)
        .zip(&std)
        .map(|((close_value, ma_value), std_value)| (*close_value - *ma_value) / (*std_value + EPS))
        .collect();
    (pos, width, zscore)
}

fn calculate_adx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut plus_dm = Vec::with_capacity(close.len());
    let mut minus_dm = Vec::with_capacity(close.len());
    let mut tr = Vec::with_capacity(close.len());
    for idx in 0..close.len() {
        if idx == 0 {
            plus_dm.push(0.0);
            minus_dm.push(0.0);
            tr.push((high[idx] - low[idx]).abs());
            continue;
        }
        let up_move = high[idx] - high[idx - 1];
        let down_move = -(low[idx] - low[idx - 1]);
        plus_dm.push(if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        });
        minus_dm.push(if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        });
        tr.push(skip_nan_max(&[
            (high[idx] - low[idx]).abs(),
            (high[idx] - close[idx - 1]).abs(),
            (low[idx] - close[idx - 1]).abs(),
        ]));
    }
    let atr = ewm_adjust_false(&tr, 1.0 / period as f64, period);
    let plus_di = safe_divide(
        &scale_values(
            &ewm_adjust_false(&plus_dm, 1.0 / period as f64, period),
            100.0,
        ),
        &add_scalar(&atr, EPS),
    );
    let minus_di = safe_divide(
        &scale_values(
            &ewm_adjust_false(&minus_dm, 1.0 / period as f64, period),
            100.0,
        ),
        &add_scalar(&atr, EPS),
    );
    let dx = plus_di
        .iter()
        .zip(&minus_di)
        .map(|(plus_value, minus_value)| {
            100.0 * (*plus_value - *minus_value).abs() / (*plus_value + *minus_value + EPS)
        })
        .collect::<Vec<_>>();
    let adx = ewm_adjust_false(&dx, 1.0 / period as f64, period);
    (adx, plus_di, minus_di)
}

fn scale_values(values: &[f64], scale: f64) -> Vec<f64> {
    values.iter().map(|value| *value * scale).collect()
}

fn calculate_mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Vec<f64> {
    let typical_price = typical_price_f32(high, low, close);
    let raw_money_flow = typical_price
        .iter()
        .zip(volume)
        .map(|(tp, volume_value)| {
            *tp * if volume_value.is_nan() {
                0.0
            } else {
                *volume_value
            }
        })
        .collect::<Vec<_>>();
    let tp_delta = diff(&typical_price);
    let pos_flow = raw_money_flow
        .iter()
        .zip(&tp_delta)
        .map(|(flow, delta)| if *delta > 0.0 { *flow } else { 0.0 })
        .collect::<Vec<_>>();
    let neg_flow = raw_money_flow
        .iter()
        .zip(&tp_delta)
        .map(|(flow, delta)| if *delta < 0.0 { flow.abs() } else { 0.0 })
        .collect::<Vec<_>>();
    let pos_sum = rolling_sum_min_count(&pos_flow, period, period);
    let neg_sum = rolling_sum_min_count(&neg_flow, period, period);
    pos_sum
        .iter()
        .zip(&neg_sum)
        .map(|(pos_value, neg_value)| {
            let money_ratio = *pos_value / (*neg_value + EPS);
            100.0 - (100.0 / (1.0 + money_ratio))
        })
        .collect()
}

fn calculate_cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let typical_price = typical_price_f32(high, low, close);
    let tp_ma = rolling_mean_min_count(&typical_price, period, period);
    let tp_mad = rolling_mean_abs_dev_full_window(&typical_price, period);
    typical_price
        .iter()
        .zip(&tp_ma)
        .zip(&tp_mad)
        .map(|((tp, ma), mad)| (*tp - *ma) / (0.015 * *mad + EPS))
        .collect()
}

fn typical_price_f32(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low)
        .zip(close)
        .map(|((high_value, low_value), close_value)| {
            ((*high_value as f32 + *low_value as f32 + *close_value as f32) / 3.0f32) as f64
        })
        .collect()
}

fn rolling_mean_abs_dev_full_window(values: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        if end + 1 < window {
            out.push(f64::NAN);
            continue;
        }
        let start = end + 1 - window;
        let slice = &values[start..=end];
        if slice.iter().any(|value| value.is_nan()) {
            out.push(f64::NAN);
            continue;
        }
        let mean = slice.iter().sum::<f64>() / window as f64;
        out.push(slice.iter().map(|value| (*value - mean).abs()).sum::<f64>() / window as f64);
    }
    out
}

fn calculate_willr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let highest_high = rolling_extreme_min_count(high, period, period, true);
    let lowest_low = rolling_extreme_min_count(low, period, period, false);
    highest_high
        .iter()
        .zip(&lowest_low)
        .zip(close)
        .map(|((highest, lowest), close_value)| {
            -100.0 * (*highest - *close_value) / (*highest - *lowest + EPS)
        })
        .collect()
}

fn calculate_aroon(high: &[f64], low: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let since_high = crate::feature_kernels::rolling_periods_since_extreme(high, period, true);
    let since_low = crate::feature_kernels::rolling_periods_since_extreme(low, period, false);
    let aroon_up = since_high
        .iter()
        .map(|value| 100.0 * (period as f64 - *value) / period as f64)
        .collect::<Vec<_>>();
    let aroon_down = since_low
        .iter()
        .map(|value| 100.0 * (period as f64 - *value) / period as f64)
        .collect::<Vec<_>>();
    let osc = aroon_up
        .iter()
        .zip(&aroon_down)
        .map(|(up, down)| *up - *down)
        .collect();
    (aroon_up, aroon_down, osc)
}

fn calculate_trix(
    close: &[f64],
    period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema1 = ewm_adjust_false(close, 2.0 / (period as f64 + 1.0), period);
    let ema2 = ewm_adjust_false(&ema1, 2.0 / (period as f64 + 1.0), period);
    let ema3 = ewm_adjust_false(&ema2, 2.0 / (period as f64 + 1.0), period);
    let trix = scale_values(&pct_change(&ema3, 1), 100.0);
    let signal = ewm_adjust_false(&trix, 2.0 / (signal_period as f64 + 1.0), signal_period);
    let hist = trix
        .iter()
        .zip(&signal)
        .map(|(trix_value, signal_value)| *trix_value - *signal_value)
        .collect();
    (trix, signal, hist)
}

fn calculate_obv_flow(close: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
    let close_diff = diff(close)
        .into_iter()
        .map(|value| if value.is_nan() { 0.0 } else { value })
        .collect::<Vec<_>>();
    let signed_volume = close_diff
        .iter()
        .zip(volume)
        .map(|(delta, volume_value)| {
            let sign = if *delta > 0.0 {
                1.0
            } else if *delta < 0.0 {
                -1.0
            } else {
                0.0
            };
            sign * if volume_value.is_nan() {
                0.0
            } else {
                *volume_value
            }
        })
        .collect::<Vec<_>>();
    let clean_volume = volume
        .iter()
        .map(|value| if value.is_nan() { 0.0 } else { *value })
        .collect::<Vec<_>>();
    let signed_sum = rolling_sum_min_count(&signed_volume, window, window);
    let volume_sum = rolling_sum_min_count(&clean_volume, window, window);
    safe_divide(&signed_sum, &add_scalar(&volume_sum, EPS))
}

fn safe_divide(numerator: &[f64], denominator: &[f64]) -> Vec<f64> {
    numerator
        .iter()
        .zip(denominator)
        .map(|(left, right)| *left / *right)
        .collect()
}

fn safe_divide_eps(numerator: &[f64], denominator: &[f64]) -> Vec<f64> {
    numerator
        .iter()
        .zip(denominator)
        .map(|(left, right)| *left / (*right + EPS))
        .collect()
}

fn subtract(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| *a - *b).collect()
}

fn ratio_minus_one(numerator: &[f64], denominator: &[f64]) -> Vec<f64> {
    numerator
        .iter()
        .zip(denominator)
        .map(|(left, right)| *left / *right - 1.0)
        .collect()
}

fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in &values[start..=end] {
            if value.is_nan() {
                continue;
            }
            sum += *value;
            count += 1;
        }
        if count == 0 {
            out.push(f64::NAN);
        } else {
            out.push(sum / count as f64);
        }
    }
    out
}

fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in &values[start..=end] {
            if value.is_nan() {
                continue;
            }
            sum += *value;
            count += 1;
        }
        if count < 2 {
            out.push(f64::NAN);
        } else {
            let count_f = count as f64;
            let mean = sum / count_f;
            let mut sum_sq_diff = 0.0;
            for value in &values[start..=end] {
                if value.is_nan() {
                    continue;
                }
                let diff = *value - mean;
                sum_sq_diff += diff * diff;
            }
            let variance = sum_sq_diff / (count_f - 1.0);
            out.push(variance.max(0.0).sqrt());
        }
    }
    out
}

fn rolling_max(values: &[f64], window: usize) -> Vec<f64> {
    rolling_extreme(values, window, true)
}

fn rolling_min(values: &[f64], window: usize) -> Vec<f64> {
    rolling_extreme(values, window, false)
}

fn rolling_extreme(values: &[f64], window: usize, find_max: bool) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut chosen = f64::NAN;
        for value in &values[start..=end] {
            if value.is_nan() {
                continue;
            }
            if chosen.is_nan() || (find_max && *value > chosen) || (!find_max && *value < chosen) {
                chosen = *value;
            }
        }
        out.push(chosen);
    }
    out
}

fn rolling_corr(left: &[f64], right: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(left.len());
    for end in 0..left.len() {
        let start = (end + 1).saturating_sub(window);
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        let mut sxy = 0.0;
        let mut count = 0usize;
        for idx in start..=end {
            let x = left[idx];
            let y = right[idx];
            if x.is_nan() || y.is_nan() {
                continue;
            }
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            count += 1;
        }
        if count < 2 {
            out.push(f64::NAN);
            continue;
        }
        let count_f = count as f64;
        let numerator = count_f * sxy - sx * sy;
        let denom_x = count_f * sxx - sx * sx;
        let denom_y = count_f * syy - sy * sy;
        if denom_x <= 0.0 || denom_y <= 0.0 {
            out.push(f64::NAN);
        } else {
            out.push(numerator / (denom_x * denom_y).sqrt());
        }
    }
    out
}

fn rolling_corr_with_std_guard(left: &[f64], right: &[f64], window: usize) -> Vec<f64> {
    let mut corr = rolling_corr(left, right, window);
    let left_std = rolling_std(left, window);
    let right_std = rolling_std(right, window);
    for idx in 0..corr.len() {
        if (left_std[idx] - 0.0).abs() <= 2e-5 || (right_std[idx] - 0.0).abs() <= 2e-5 {
            corr[idx] = f64::NAN;
        }
    }
    corr
}

fn rolling_quantile(values: &[f64], window: usize, quantile: f64) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut observed = values[start..=end]
            .iter()
            .filter(|value| !value.is_nan())
            .copied()
            .collect::<Vec<_>>();
        if observed.is_empty() {
            out.push(f64::NAN);
            continue;
        }
        observed.sort_by(|left, right| left.total_cmp(right));
        let pos = (observed.len() - 1) as f64 * quantile;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        if lower == upper {
            out.push(observed[lower]);
        } else {
            let weight = pos - lower as f64;
            out.push(observed[lower] * (1.0 - weight) + observed[upper] * weight);
        }
    }
    out
}

fn rolling_regression_stats(values: &[f64], window: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let window = window.max(1);
    let mut slopes = Vec::with_capacity(values.len());
    let mut rsquares = Vec::with_capacity(values.len());
    let mut residuals = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut ys = Vec::new();
        for value in &values[start..=end] {
            if !value.is_finite() {
                continue;
            }
            ys.push(*value);
        }
        let count = ys.len();
        if count < 2 {
            slopes.push(f64::NAN);
            rsquares.push(f64::NAN);
            residuals.push(f64::NAN);
            continue;
        }
        let count_f = count as f64;
        let sx = count_f * (count_f + 1.0) / 2.0;
        let sxx = count_f * (count_f + 1.0) * (2.0 * count_f + 1.0) / 6.0;
        let sy = ys.iter().sum::<f64>();
        let syy = ys.iter().map(|value| value * value).sum::<f64>();
        let sxy = ys
            .iter()
            .enumerate()
            .map(|(idx, value)| (idx + 1) as f64 * *value)
            .sum::<f64>();
        let numer = count_f * sxy - sx * sy;
        let denom_x = count_f * sxx - sx * sx;
        if denom_x.abs() <= f64::EPSILON {
            slopes.push(f64::NAN);
            rsquares.push(f64::NAN);
            residuals.push(f64::NAN);
            continue;
        }
        let slope = numer / denom_x;
        let denom_y = count_f * syy - sy * sy;
        let rsquare = if denom_y > 0.0 {
            ((numer * numer) / (denom_x * denom_y)).clamp(0.0, 1.0)
        } else {
            f64::NAN
        };
        let alpha = (sy - slope * sx) / count_f;
        let residual = ys.last().copied().unwrap_or(f64::NAN) - (alpha + slope * count_f);
        slopes.push(slope);
        rsquares.push(rsquare);
        residuals.push(residual);
    }
    (slopes, rsquares, residuals)
}

fn greater_than(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(a, b)| if *a > *b { 1.0 } else { 0.0 })
        .collect()
}

fn less_than(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(a, b)| if *a < *b { 1.0 } else { 0.0 })
        .collect()
}

fn calculate_atr(
    high: &[f64],
    low: &[f64],
    raw_close: &[f64],
    close_denominator: &[f64],
    period: usize,
) -> Vec<f64> {
    let mut true_range = Vec::with_capacity(raw_close.len());
    for idx in 0..raw_close.len() {
        let high_low = high[idx] - low[idx];
        let high_close = if idx == 0 {
            f64::NAN
        } else {
            (high[idx] - raw_close[idx - 1]).abs()
        };
        let low_close = if idx == 0 {
            f64::NAN
        } else {
            (low[idx] - raw_close[idx - 1]).abs()
        };
        true_range.push(skip_nan_max(&[high_low, high_close, low_close]));
    }
    safe_divide(&rolling_mean(&true_range, period), close_denominator)
}

fn skip_nan_max(values: &[f64]) -> f64 {
    let mut chosen = f64::NAN;
    for value in values {
        if value.is_nan() {
            continue;
        }
        if chosen.is_nan() || *value > chosen {
            chosen = *value;
        }
    }
    chosen
}

fn log1p_clipped_nonnegative(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            if value.is_nan() {
                f64::NAN
            } else {
                value.max(0.0).ln_1p()
            }
        })
        .collect()
}

fn positive_inverse(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| if *value > 0.0 { 1.0 / *value } else { f64::NAN })
        .collect()
}

fn fill_nan(values: &[f64], fill_value: f64) -> Vec<f64> {
    values
        .iter()
        .map(|value| if value.is_nan() { fill_value } else { *value })
        .collect()
}

fn nonpositive_or_zero_flag(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| if *value <= 0.0 { 1.0 } else { 0.0 })
        .collect()
}

fn nonpositive_invalid_flag(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            if value.is_nan() {
                f64::NAN
            } else if *value <= 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn sorted_unique_windows(windows: &[usize]) -> Vec<usize> {
    let mut out = windows.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn get_tushare_column(columns: &HashMap<String, Vec<f64>>, len: usize, name: &str) -> Vec<f64> {
    columns
        .get(name)
        .cloned()
        .unwrap_or_else(|| vec![f64::NAN; len])
}

fn get_tushare_column_with_fallback(
    columns: &HashMap<String, Vec<f64>>,
    len: usize,
    primary: &str,
    fallback: &str,
) -> Vec<f64> {
    columns
        .get(primary)
        .or_else(|| columns.get(fallback))
        .cloned()
        .unwrap_or_else(|| vec![f64::NAN; len])
}

fn add(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| *a + *b).collect()
}

fn add_many(values: &[&Vec<f64>]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let len = values[0].len();
    let mut out = vec![0.0; len];
    for series in values {
        for idx in 0..len {
            out[idx] += series[idx];
        }
    }
    out
}

fn multiply(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| *a * *b).collect()
}

fn scalar_mul(values: &[f64], scalar: f64) -> Vec<f64> {
    values.iter().map(|value| *value * scalar).collect()
}

fn sub_scalar(values: &[f64], scalar: f64) -> Vec<f64> {
    values.iter().map(|value| *value - scalar).collect()
}

fn binary_map(left: &[f64], right: &[f64], f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| f(*left_value, *right_value))
        .collect()
}

fn where_mask(values: &[f64], mask_values: &[f64], predicate: impl Fn(f64) -> bool) -> Vec<f64> {
    values
        .iter()
        .zip(mask_values)
        .map(|(value, mask_value)| {
            if predicate(*mask_value) {
                *value
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn diff_window(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        if idx < window {
            out.push(f64::NAN);
        } else {
            out.push(values[idx] - values[idx - window]);
        }
    }
    out
}

fn safe_ratio_abs_denominator(numerator: &[f64], denominator: &[f64]) -> Vec<f64> {
    numerator
        .iter()
        .zip(denominator)
        .map(|(left, right)| *left / (right.abs() + EPS))
        .collect()
}

fn positive_safe_ratio(numerator: &[f64], denominator: &[f64]) -> Vec<f64> {
    numerator
        .iter()
        .zip(denominator)
        .map(|(left, right)| {
            if *right > EPS {
                *left / *right
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn exp_neg_clipped_days(values: &[f64], decay: f64) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            if value.is_nan() {
                f64::NAN
            } else {
                (-(value.max(0.0)) / decay).exp()
            }
        })
        .collect()
}

fn fc_positive_confidence_values(
    min_values: &[f64],
    max_values: &[f64],
    mid_values: &[f64],
    width_values: &[f64],
) -> Vec<f64> {
    min_values
        .iter()
        .zip(max_values)
        .zip(mid_values)
        .zip(width_values)
        .map(|(((min_value, max_value), mid_value), width_value)| {
            if min_value.is_nan()
                || max_value.is_nan()
                || mid_value.is_nan()
                || width_value.is_nan()
            {
                f64::NAN
            } else if *min_value > 0.0 {
                *mid_value / (width_value.abs() + EPS)
            } else if *max_value < 0.0 {
                -mid_value.abs() / (width_value.abs() + EPS)
            } else {
                0.0
            }
        })
        .collect()
}

fn event_age(event: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(event.len());
    let mut last_idx = None;
    for (idx, value) in event.iter().enumerate() {
        if value.is_finite() && *value > 0.0 {
            last_idx = Some(idx);
            out.push(0.0);
        } else if let Some(last) = last_idx {
            out.push((idx - last) as f64);
        } else {
            out.push(f64::NAN);
        }
    }
    out
}

fn event_streak(event: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(event.len());
    let mut streak = 0.0;
    for value in event {
        if value.is_finite() && *value > 0.0 {
            streak += 1.0;
        } else {
            streak = 0.0;
        }
        out.push(streak);
    }
    out
}

fn bounded_signal(values: &[f64], scale: f64) -> Vec<f64> {
    let scale = scale.max(EPS);
    values.iter().map(|value| (*value / scale).tanh()).collect()
}

fn weighted_observed_sum(
    components: &[(f64, Vec<f64>)],
    len: usize,
    min_valid_weight_ratio: f64,
) -> Vec<f64> {
    let total_weight = components
        .iter()
        .map(|(weight, _)| weight.abs())
        .sum::<f64>();
    if total_weight <= EPS {
        return vec![f64::NAN; len];
    }
    let required_weight = total_weight * min_valid_weight_ratio;
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let mut weighted_sum = 0.0;
        let mut observed_weight = 0.0;
        for (weight, values) in components {
            let value = values[idx];
            if value.is_nan() {
                continue;
            }
            weighted_sum += value * *weight;
            observed_weight += weight.abs();
        }
        if observed_weight >= required_weight {
            out.push(weighted_sum);
        } else {
            out.push(f64::NAN);
        }
    }
    out
}

fn clip_values(values: &[f64], lower: f64, upper: f64) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            if value.is_nan() {
                f64::NAN
            } else {
                value.max(lower).min(upper)
            }
        })
        .collect()
}

fn clip_lower_vec(values: &[f64], lower: f64) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            if value.is_nan() {
                f64::NAN
            } else {
                value.max(lower)
            }
        })
        .collect()
}

fn round_to_f32(values: &[f64]) -> Vec<f64> {
    values.iter().map(|value| (*value as f32) as f64).collect()
}

fn push_nan_columns(out: &mut Vec<(String, Vec<f64>)>, len: usize, names: &[&str]) {
    for name in names {
        out.push(((*name).to_owned(), vec![f64::NAN; len]));
    }
}

fn has_group(groups: &[String], target: &str) -> bool {
    groups.iter().any(|group| group == target)
}

fn numpy_max(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left >= right {
        left
    } else {
        right
    }
}

fn numpy_min(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left <= right {
        left
    } else {
        right
    }
}

#[cfg(test)]
mod tests {
    use super::alpha158_kbar_price;

    #[test]
    fn alpha158_kbar_price_matches_expected_basic_formulas() {
        let open = vec![10.0, 11.0, 12.0];
        let high = vec![12.0, 12.0, 13.0];
        let low = vec![9.0, 10.0, 11.0];
        let close = vec![11.0, 10.5, 12.5];
        let vwap = vec![10.5, 11.0, 12.0];
        let result = alpha158_kbar_price(
            &open,
            &high,
            &low,
            &close,
            &vwap,
            true,
            &["OPEN".to_owned(), "VWAP".to_owned()],
            &[0, 1],
        )
        .unwrap();
        let names = result
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2", "OPEN0",
                "OPEN1", "VWAP0", "VWAP1"
            ]
        );
        assert!((result[0].1[0] - 0.1).abs() < 1e-12);
        assert!(result[10].1[0].is_nan());
        assert!((result[10].1[1] - (10.0 / 10.5)).abs() < 1e-12);
    }
}
