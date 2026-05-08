use super::analyze::require_window;
use super::ast::{BinaryOp, Expr, UnaryOp};
use super::operator::Operator;
use super::series::{
    lag, rolling_extreme, rolling_mean, rolling_rank_pct, rolling_std, rolling_sum,
};
use std::collections::{BTreeMap, HashMap};

pub(crate) fn eval_expr(
    expr: &Expr,
    columns: &HashMap<String, Vec<f64>>,
    len: usize,
    cache: &mut BTreeMap<String, Vec<f64>>,
) -> Result<Vec<f64>, String> {
    let cache_key = format!("{expr:?}");
    if let Some(values) = cache.get(&cache_key) {
        return Ok(values.clone());
    }
    let values = match expr {
        Expr::Number(value) => vec![*value; len],
        Expr::Column(name) => columns
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown source column: {name}"))?,
        Expr::Unary {
            op: UnaryOp::Neg,
            expr,
        } => eval_expr(expr, columns, len, cache)?
            .into_iter()
            .map(|value| -value)
            .collect(),
        Expr::Binary { op, left, right } => {
            let left = eval_expr(left, columns, len, cache)?;
            let right = eval_expr(right, columns, len, cache)?;
            eval_binary(*op, &left, &right)?
        }
        Expr::Call { op, args } => eval_call(*op, args, columns, len, cache)?,
    };
    if values.len() != len {
        return Err(format!(
            "expression produced length {}, expected {len}",
            values.len()
        ));
    }
    cache.insert(cache_key, values.clone());
    Ok(values)
}

fn eval_binary(op: BinaryOp, left: &[f64], right: &[f64]) -> Result<Vec<f64>, String> {
    if left.len() != right.len() {
        return Err(format!(
            "binary operands length mismatch: {} vs {}",
            left.len(),
            right.len()
        ));
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| match op {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => {
                if !right.is_finite() || right.abs() <= 1e-12 {
                    f64::NAN
                } else {
                    left / right
                }
            }
        })
        .collect())
}

fn eval_call(
    op: Operator,
    args: &[Expr],
    columns: &HashMap<String, Vec<f64>>,
    len: usize,
    cache: &mut BTreeMap<String, Vec<f64>>,
) -> Result<Vec<f64>, String> {
    match op {
        Operator::Abs | Operator::Log | Operator::Sqrt => {
            let values = eval_expr(&args[0], columns, len, cache)?;
            Ok(values
                .into_iter()
                .map(|value| match op {
                    Operator::Abs => value.abs(),
                    Operator::Log => {
                        if value > 0.0 {
                            value.ln()
                        } else {
                            f64::NAN
                        }
                    }
                    Operator::Sqrt => {
                        if value >= 0.0 {
                            value.sqrt()
                        } else {
                            f64::NAN
                        }
                    }
                    _ => unreachable!(),
                })
                .collect())
        }
        Operator::Delay => {
            let values = eval_expr(&args[0], columns, len, cache)?;
            Ok(lag(&values, require_window(&args[1])?))
        }
        Operator::Delta => {
            let values = eval_expr(&args[0], columns, len, cache)?;
            let lagged = lag(&values, require_window(&args[1])?);
            eval_binary(BinaryOp::Sub, &values, &lagged)
        }
        Operator::TsMean
        | Operator::TsSum
        | Operator::TsStd
        | Operator::TsMin
        | Operator::TsMax
        | Operator::TsRank => {
            let values = eval_expr(&args[0], columns, len, cache)?;
            let window = require_window(&args[1])?;
            Ok(match op {
                Operator::TsMean => rolling_mean(&values, window),
                Operator::TsSum => rolling_sum(&values, window),
                Operator::TsStd => rolling_std(&values, window),
                Operator::TsMin => rolling_extreme(&values, window, false),
                Operator::TsMax => rolling_extreme(&values, window, true),
                Operator::TsRank => rolling_rank_pct(&values, window),
                _ => unreachable!(),
            })
        }
    }
}
