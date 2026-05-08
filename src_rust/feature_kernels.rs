pub fn rolling_rank_pct(values: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let last = values[end];
        if !last.is_finite() {
            out.push(f64::NAN);
            continue;
        }

        let start = (end + 1).saturating_sub(window);
        let mut less = 0usize;
        let mut equal = 0usize;
        let mut valid = 0usize;
        for value in &values[start..=end] {
            if !value.is_finite() {
                continue;
            }
            valid += 1;
            if *value < last {
                less += 1;
            } else if *value == last {
                equal += 1;
            }
        }

        if valid == 0 {
            out.push(f64::NAN);
        } else {
            out.push((less as f64 + (equal as f64 + 1.0) * 0.5) / valid as f64);
        }
    }
    out
}

pub fn rolling_arg_extreme_position(values: &[f64], window: usize, find_max: bool) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut chosen: Option<usize> = None;
        let mut chosen_value = 0.0f64;
        let has_observed = values[start..=end].iter().any(|value| !value.is_nan());
        if !has_observed {
            out.push(f64::NAN);
            continue;
        }

        for (offset, value) in values[start..=end].iter().enumerate() {
            if value.is_nan() {
                chosen = Some(offset);
                break;
            }
            if chosen.is_none() {
                chosen = Some(offset);
                chosen_value = *value;
                continue;
            }
            let better = if find_max {
                *value > chosen_value
            } else {
                *value < chosen_value
            };
            if better {
                chosen = Some(offset);
                chosen_value = *value;
            }
        }

        out.push(chosen.map(|idx| (idx + 1) as f64).unwrap_or(f64::NAN));
    }
    out
}

pub fn rolling_periods_since_extreme(values: &[f64], window: usize, find_max: bool) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        if end + 1 < window {
            out.push(f64::NAN);
            continue;
        }
        let start = end + 1 - window;
        let mut chosen = 0usize;
        let mut chosen_value = 0.0f64;
        let mut valid = true;
        for (offset, value) in values[start..=end].iter().enumerate() {
            if value.is_nan() {
                valid = false;
                break;
            }
            if offset == 0 {
                chosen = 0;
                chosen_value = *value;
                continue;
            }
            let better = if find_max {
                *value > chosen_value
            } else {
                *value < chosen_value
            };
            if better {
                chosen = offset;
                chosen_value = *value;
            }
        }
        if valid {
            out.push((window - 1 - chosen) as f64);
        } else {
            out.push(f64::NAN);
        }
    }
    out
}

pub fn rolling_mean_abs_dev(values: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        if end + 1 < window {
            out.push(f64::NAN);
            continue;
        }
        let start = end + 1 - window;
        let slice = &values[start..=end];
        if slice.iter().any(|value| !value.is_finite()) {
            out.push(f64::NAN);
            continue;
        }
        let mean = slice.iter().sum::<f64>() / window as f64;
        let mad = slice.iter().map(|value| (value - mean).abs()).sum::<f64>() / window as f64;
        out.push(mad);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::rolling_rank_pct;
    use super::{
        rolling_arg_extreme_position, rolling_mean_abs_dev, rolling_periods_since_extreme,
    };

    fn assert_eq_nan_aware(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
            if right.is_nan() {
                assert!(left.is_nan(), "index {idx}: expected NaN, got {left}");
            } else {
                assert!(
                    (*left - *right).abs() <= 1e-12,
                    "index {idx}: expected {right}, got {left}"
                );
            }
        }
    }

    #[test]
    fn rolling_rank_pct_matches_current_reference_semantics() {
        let values = [1.0, 2.0, 2.0, f64::NAN, 3.0, 1.5, 1.5, 4.0];
        let actual = rolling_rank_pct(&values, 4);
        let expected = [
            1.0,
            1.0,
            0.8333333333333334,
            f64::NAN,
            1.0,
            0.3333333333333333,
            0.5,
            1.0,
        ];
        assert_eq_nan_aware(&actual, &expected);
    }

    #[test]
    fn rolling_rank_pct_treats_infinities_as_missing() {
        let values = [1.0, f64::INFINITY, 2.0, f64::NAN, f64::NEG_INFINITY, 2.0];
        let actual = rolling_rank_pct(&values, 3);
        let expected = [1.0, f64::NAN, 1.0, f64::NAN, f64::NAN, 1.0];
        assert_eq_nan_aware(&actual, &expected);
    }

    #[test]
    fn rolling_arg_extreme_position_matches_numpy_arg_semantics() {
        let values = [1.0, f64::NAN, 2.0, 2.0, 0.0];
        let actual_max = rolling_arg_extreme_position(&values, 3, true);
        let actual_min = rolling_arg_extreme_position(&values, 3, false);
        assert_eq_nan_aware(&actual_max, &[1.0, 2.0, 2.0, 1.0, 1.0]);
        assert_eq_nan_aware(&actual_min, &[1.0, 2.0, 2.0, 1.0, 3.0]);
    }

    #[test]
    fn rolling_periods_since_extreme_requires_full_valid_window() {
        let values = [1.0, 3.0, 2.0, f64::NAN, 5.0, 4.0, 6.0];
        let actual = rolling_periods_since_extreme(&values, 3, true);
        assert_eq_nan_aware(
            &actual,
            &[f64::NAN, f64::NAN, 1.0, f64::NAN, f64::NAN, f64::NAN, 0.0],
        );
    }

    #[test]
    fn rolling_mean_abs_dev_requires_full_finite_window() {
        let values = [1.0, 3.0, 5.0, f64::NAN, 9.0];
        let actual = rolling_mean_abs_dev(&values, 3);
        assert_eq_nan_aware(
            &actual,
            &[f64::NAN, f64::NAN, 4.0 / 3.0, f64::NAN, f64::NAN],
        );
    }
}
