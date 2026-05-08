pub(crate) fn lag(values: &[f64], window: usize) -> Vec<f64> {
    (0..values.len())
        .map(|idx| {
            if idx < window {
                f64::NAN
            } else {
                values[idx - window]
            }
        })
        .collect()
}

pub(crate) fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    rolling_sum_count(values, window)
        .into_iter()
        .map(|(sum, count)| {
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect()
}

pub(crate) fn rolling_sum(values: &[f64], window: usize) -> Vec<f64> {
    rolling_sum_count(values, window)
        .into_iter()
        .map(|(sum, count)| if count == 0 { f64::NAN } else { sum })
        .collect()
}

fn rolling_sum_count(values: &[f64], window: usize) -> Vec<(f64, usize)> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    let mut count = 0usize;
    for idx in 0..values.len() {
        let value = values[idx];
        if value.is_finite() {
            sum += value;
            count += 1;
        }
        if idx >= window {
            let leaving = values[idx - window];
            if leaving.is_finite() {
                sum -= leaving;
                count -= 1;
            }
        }
        out.push((sum, count));
    }
    out
}

pub(crate) fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let observed = values[start..=end]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if observed.len() < 2 {
            out.push(f64::NAN);
            continue;
        }
        let mean = observed.iter().sum::<f64>() / observed.len() as f64;
        let var = observed
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f64>()
            / (observed.len() - 1) as f64;
        out.push(var.sqrt());
    }
    out
}

pub(crate) fn rolling_extreme(values: &[f64], window: usize, find_max: bool) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let start = (end + 1).saturating_sub(window);
        let mut best = f64::NAN;
        for value in &values[start..=end] {
            if !value.is_finite() {
                continue;
            }
            if !best.is_finite() || (find_max && *value > best) || (!find_max && *value < best) {
                best = *value;
            }
        }
        out.push(best);
    }
    out
}

pub(crate) fn rolling_rank_pct(values: &[f64], window: usize) -> Vec<f64> {
    let window = window.max(1);
    let mut out = Vec::with_capacity(values.len());
    for end in 0..values.len() {
        let current = values[end];
        if !current.is_finite() {
            out.push(f64::NAN);
            continue;
        }
        let start = (end + 1).saturating_sub(window);
        let mut less = 0usize;
        let mut equal = 0usize;
        let mut count = 0usize;
        for value in &values[start..=end] {
            if !value.is_finite() {
                continue;
            }
            count += 1;
            if *value < current {
                less += 1;
            } else if *value == current {
                equal += 1;
            }
        }
        if count == 0 {
            out.push(f64::NAN);
        } else {
            out.push((less as f64 + 0.5 * (equal as f64 + 1.0)) / count as f64);
        }
    }
    out
}
