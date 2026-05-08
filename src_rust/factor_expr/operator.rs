#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Operator {
    Abs,
    Log,
    Sqrt,
    Delay,
    Delta,
    TsMean,
    TsSum,
    TsStd,
    TsMin,
    TsMax,
    TsRank,
}

impl Operator {
    pub(crate) fn from_name(raw_name: &str) -> Result<Self, String> {
        let name = normalize_func_name(raw_name);
        match name.as_str() {
            "lead" | "ts_lead" | "future" | "shift" => Err(format!(
                "future-looking or ambiguous operator is not allowed in factor expressions: {name}"
            )),
            "rank" => Err(
                "bare rank is cross-sectional in common alpha DSLs; use ts_rank for per-symbol time-series rank"
                    .to_owned(),
            ),
            "abs" => Ok(Self::Abs),
            "log" => Ok(Self::Log),
            "sqrt" => Ok(Self::Sqrt),
            "delay" | "ts_delay" | "ref" => Ok(Self::Delay),
            "delta" | "ts_delta" => Ok(Self::Delta),
            "mean" | "ts_mean" => Ok(Self::TsMean),
            "sum" | "ts_sum" => Ok(Self::TsSum),
            "std" | "ts_std" => Ok(Self::TsStd),
            "min" | "ts_min" => Ok(Self::TsMin),
            "max" | "ts_max" => Ok(Self::TsMax),
            "ts_rank" => Ok(Self::TsRank),
            other => Err(format!("unsupported factor expression function: {other}")),
        }
    }

    pub(crate) fn canonical_name(self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Delay => "delay",
            Self::Delta => "delta",
            Self::TsMean => "ts_mean",
            Self::TsSum => "ts_sum",
            Self::TsStd => "ts_std",
            Self::TsMin => "ts_min",
            Self::TsMax => "ts_max",
            Self::TsRank => "ts_rank",
        }
    }

    pub(crate) fn arity(self) -> usize {
        match self {
            Self::Abs | Self::Log | Self::Sqrt => 1,
            Self::Delay
            | Self::Delta
            | Self::TsMean
            | Self::TsSum
            | Self::TsStd
            | Self::TsMin
            | Self::TsMax
            | Self::TsRank => 2,
        }
    }

    pub(crate) fn window_arg_index(self) -> Option<usize> {
        match self {
            Self::Delay
            | Self::Delta
            | Self::TsMean
            | Self::TsSum
            | Self::TsStd
            | Self::TsMin
            | Self::TsMax
            | Self::TsRank => Some(1),
            Self::Abs | Self::Log | Self::Sqrt => None,
        }
    }
}

fn normalize_func_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}
