mod analyze;
mod ast;
mod eval;
mod operator;
mod series;

use analyze::{summarize_profile, validate_expression};
use ast::{parse_expression, Expr};
use eval::eval_expr;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FactorExpressionProfile {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub factor_store_name: String,
    #[serde(default)]
    pub description: String,
    pub factors: Vec<FactorExpressionSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FactorExpressionSpec {
    pub name: String,
    pub expr: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct CompiledFactorExpression {
    pub name: String,
    pub expr: String,
    pub description: String,
    ast: Expr,
}

#[derive(Debug, Clone)]
pub struct CompiledFactorExpressionProfile {
    pub name: String,
    pub factor_store_name: String,
    pub description: String,
    pub factors: Vec<CompiledFactorExpression>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FactorExpressionProfileSummary {
    pub name: String,
    pub factor_store_name: String,
    pub description: String,
    pub stage: String,
    pub factor_count: usize,
    pub factor_names: Vec<String>,
    pub required_columns: Vec<String>,
    pub functions: Vec<String>,
    pub max_lookback: usize,
    pub expressions: Vec<FactorExpressionSpec>,
}

pub fn load_factor_expression_profile(
    path: &Path,
) -> Result<CompiledFactorExpressionProfile, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read factor expression profile {}: {err}",
            path.display()
        )
    })?;
    let profile: FactorExpressionProfile = serde_yaml::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse factor expression profile {}: {err}",
            path.display()
        )
    })?;
    compile_factor_expression_profile(profile)
}

pub fn compile_factor_expression_profile(
    profile: FactorExpressionProfile,
) -> Result<CompiledFactorExpressionProfile, String> {
    if profile.factors.is_empty() {
        return Err("factor expression profile must define at least one factor".to_owned());
    }
    let mut seen = HashSet::new();
    let mut factors = Vec::with_capacity(profile.factors.len());
    for factor in profile.factors {
        validate_factor_name(&factor.name)?;
        if !seen.insert(factor.name.clone()) {
            return Err(format!("duplicate factor expression name: {}", factor.name));
        }
        let ast = parse_expression(&factor.expr)
            .map_err(|err| format!("factor {} expression error: {err}", factor.name))?;
        validate_expression(&ast)?;
        factors.push(CompiledFactorExpression {
            name: factor.name,
            expr: factor.expr,
            description: factor.description,
            ast,
        });
    }
    Ok(CompiledFactorExpressionProfile {
        name: profile.name,
        factor_store_name: profile.factor_store_name,
        description: profile.description,
        factors,
    })
}

pub fn summarize_factor_expression_profile(
    profile: &CompiledFactorExpressionProfile,
) -> FactorExpressionProfileSummary {
    summarize_profile(profile)
}

pub fn evaluate_factor_expressions(
    profile: &CompiledFactorExpressionProfile,
    columns: &HashMap<String, Vec<f64>>,
    len: usize,
) -> Result<Vec<(String, Vec<f64>)>, String> {
    let mut cache = BTreeMap::new();
    profile
        .factors
        .iter()
        .map(|factor| {
            let values = eval_expr(&factor.ast, columns, len, &mut cache)
                .map_err(|err| format!("factor {} evaluation error: {err}", factor.name))?;
            Ok((factor.name.clone(), values))
        })
        .collect()
}

fn validate_factor_name(name: &str) -> Result<(), String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("factor name cannot be empty".to_owned());
    }
    if name != trimmed {
        return Err(format!(
            "factor name cannot contain leading or trailing whitespace: {name:?}"
        ));
    }
    if matches!(trimmed, "date" | "symbol" | "label") || trimmed.starts_with("label_") {
        return Err(format!(
            "factor name conflicts with reserved column: {trimmed}"
        ));
    }
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return Err("factor name cannot be empty".to_owned());
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(format!(
            "factor name must start with ASCII letter or underscore: {trimmed}"
        ));
    }
    if !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric()) {
        return Err(format!(
            "factor name may only contain ASCII letters, digits, and underscores: {trimmed}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn parses_and_evaluates_basic_time_series_expression() {
        let profile = compile_factor_expression_profile(FactorExpressionProfile {
            name: "smoke".to_owned(),
            factor_store_name: "expr_smoke".to_owned(),
            description: String::new(),
            factors: vec![FactorExpressionSpec {
                name: "expr_ret_2".to_owned(),
                expr: "close / delay(close, 2) - 1".to_owned(),
                description: String::new(),
            }],
        })
        .unwrap();
        let mut columns = HashMap::new();
        columns.insert("close".to_owned(), vec![10.0, 11.0, 12.0, 15.0]);

        let factors = evaluate_factor_expressions(&profile, &columns, 4).unwrap();

        assert!(factors[0].1[0].is_nan());
        assert!(factors[0].1[1].is_nan());
        assert_close(factors[0].1[2], 0.2);
        assert_close(factors[0].1[3], 15.0 / 11.0 - 1.0);
    }

    #[test]
    fn rejects_future_or_ambiguous_shift_ops() {
        let error = compile_factor_expression_profile(FactorExpressionProfile {
            name: "bad".to_owned(),
            factor_store_name: String::new(),
            description: String::new(),
            factors: vec![FactorExpressionSpec {
                name: "bad_factor".to_owned(),
                expr: "lead(close, 1)".to_owned(),
                description: String::new(),
            }],
        })
        .unwrap_err();

        assert!(error.contains("not allowed"));
    }

    #[test]
    fn rejects_bare_cross_sectional_rank_until_cs_stage_exists() {
        let error = compile_factor_expression_profile(FactorExpressionProfile {
            name: "bad".to_owned(),
            factor_store_name: String::new(),
            description: String::new(),
            factors: vec![FactorExpressionSpec {
                name: "bad_factor".to_owned(),
                expr: "rank(close)".to_owned(),
                description: String::new(),
            }],
        })
        .unwrap_err();

        assert!(error.contains("cross-sectional"));
    }

    #[test]
    fn summarizes_required_columns_functions_and_lookback() {
        let profile = compile_factor_expression_profile(FactorExpressionProfile {
            name: "summary".to_owned(),
            factor_store_name: "summary_store".to_owned(),
            description: "summary smoke".to_owned(),
            factors: vec![
                FactorExpressionSpec {
                    name: "ret5".to_owned(),
                    expr: "close / delay(open, 5) - 1".to_owned(),
                    description: "ret".to_owned(),
                },
                FactorExpressionSpec {
                    name: "vol20".to_owned(),
                    expr: "ts_rank(volume, 20)".to_owned(),
                    description: "vol".to_owned(),
                },
            ],
        })
        .unwrap();

        let summary = summarize_factor_expression_profile(&profile);

        assert_eq!(summary.factor_count, 2);
        assert_eq!(summary.stage, "time_series");
        assert_eq!(summary.required_columns, vec!["close", "open", "volume"]);
        assert_eq!(summary.functions, vec!["delay", "ts_rank"]);
        assert_eq!(summary.max_lookback, 20);
        assert_eq!(summary.expressions[0].description, "ret");
    }

    #[test]
    fn rejects_invalid_factor_names() {
        let error = compile_factor_expression_profile(FactorExpressionProfile {
            name: "bad".to_owned(),
            factor_store_name: String::new(),
            description: String::new(),
            factors: vec![FactorExpressionSpec {
                name: " label".to_owned(),
                expr: "close".to_owned(),
                description: String::new(),
            }],
        })
        .unwrap_err();

        assert!(error.contains("whitespace"));
    }

    #[test]
    fn tokenizer_handles_identifier_boundaries() {
        let error = compile_factor_expression_profile(FactorExpressionProfile {
            name: "bad".to_owned(),
            factor_store_name: String::new(),
            description: String::new(),
            factors: vec![FactorExpressionSpec {
                name: "bad_expr".to_owned(),
                expr: "close#volume".to_owned(),
                description: String::new(),
            }],
        })
        .unwrap_err();

        assert!(error.contains("unexpected character"));
    }

    #[test]
    fn rolling_functions_use_past_and_current_values_only() {
        let profile = compile_factor_expression_profile(FactorExpressionProfile {
            name: "rolling".to_owned(),
            factor_store_name: String::new(),
            description: String::new(),
            factors: vec![
                FactorExpressionSpec {
                    name: "mean3".to_owned(),
                    expr: "ts_mean(close, 3)".to_owned(),
                    description: String::new(),
                },
                FactorExpressionSpec {
                    name: "rank3".to_owned(),
                    expr: "ts_rank(close, 3)".to_owned(),
                    description: String::new(),
                },
            ],
        })
        .unwrap();
        let mut columns = HashMap::new();
        columns.insert("close".to_owned(), vec![1.0, 3.0, 2.0, 4.0]);

        let factors = evaluate_factor_expressions(&profile, &columns, 4).unwrap();

        assert_eq!(factors[0].0, "mean3");
        assert_close(factors[0].1[3], 3.0);
        assert_eq!(factors[1].0, "rank3");
        assert_close(factors[1].1[2], 2.0 / 3.0);
        assert_close(factors[1].1[3], 1.0);
    }
}
