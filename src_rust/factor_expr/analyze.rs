use super::ast::Expr;
use super::FactorExpressionProfileSummary;
use crate::factor_expr::{CompiledFactorExpressionProfile, FactorExpressionSpec};
use std::collections::BTreeSet;

pub(crate) fn validate_expression(expr: &Expr) -> Result<(), String> {
    match expr {
        Expr::Number(_) | Expr::Column(_) => Ok(()),
        Expr::Unary { expr, .. } => validate_expression(expr),
        Expr::Binary { left, right, .. } => {
            validate_expression(left)?;
            validate_expression(right)
        }
        Expr::Call { op, args } => {
            require_arg_count(op.canonical_name(), args, op.arity())?;
            if let Some(index) = op.window_arg_index() {
                require_window(&args[index])?;
            }
            for arg in args {
                validate_expression(arg)?;
            }
            Ok(())
        }
    }
}

pub(crate) fn summarize_profile(
    profile: &CompiledFactorExpressionProfile,
) -> FactorExpressionProfileSummary {
    let mut required_columns = BTreeSet::new();
    let mut functions = BTreeSet::new();
    let mut max_lookback = 0usize;
    for factor in &profile.factors {
        collect_expr_summary(
            &factor.ast,
            &mut required_columns,
            &mut functions,
            &mut max_lookback,
        );
    }
    FactorExpressionProfileSummary {
        name: profile.name.clone(),
        factor_store_name: profile.factor_store_name.clone(),
        description: profile.description.clone(),
        stage: "time_series".to_owned(),
        factor_count: profile.factors.len(),
        factor_names: profile
            .factors
            .iter()
            .map(|factor| factor.name.clone())
            .collect(),
        required_columns: required_columns.into_iter().collect(),
        functions: functions.into_iter().collect(),
        max_lookback,
        expressions: profile
            .factors
            .iter()
            .map(|factor| FactorExpressionSpec {
                name: factor.name.clone(),
                expr: factor.expr.clone(),
                description: factor.description.clone(),
            })
            .collect(),
    }
}

pub(crate) fn require_window(expr: &Expr) -> Result<usize, String> {
    let Expr::Number(value) = expr else {
        return Err("window argument must be a positive integer literal".to_owned());
    };
    if !value.is_finite() || *value < 1.0 || value.fract().abs() > 1e-12 {
        return Err(format!(
            "window argument must be a positive integer, got {value}"
        ));
    }
    Ok(*value as usize)
}

fn collect_expr_summary(
    expr: &Expr,
    required_columns: &mut BTreeSet<String>,
    functions: &mut BTreeSet<String>,
    max_lookback: &mut usize,
) {
    match expr {
        Expr::Number(_) => {}
        Expr::Column(name) => {
            required_columns.insert(name.clone());
        }
        Expr::Unary { expr, .. } => {
            collect_expr_summary(expr, required_columns, functions, max_lookback);
        }
        Expr::Binary { left, right, .. } => {
            collect_expr_summary(left, required_columns, functions, max_lookback);
            collect_expr_summary(right, required_columns, functions, max_lookback);
        }
        Expr::Call { op, args } => {
            functions.insert(op.canonical_name().to_owned());
            if let Some(index) = op.window_arg_index() {
                if let Some(Expr::Number(value)) = args.get(index) {
                    *max_lookback = (*max_lookback).max(*value as usize);
                }
            }
            for arg in args {
                collect_expr_summary(arg, required_columns, functions, max_lookback);
            }
        }
    }
}

fn require_arg_count(name: &str, args: &[Expr], expected: usize) -> Result<(), String> {
    if args.len() != expected {
        return Err(format!(
            "{name} expects {expected} argument(s), got {}",
            args.len()
        ));
    }
    Ok(())
}
