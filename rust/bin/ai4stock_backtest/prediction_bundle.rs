use arrow_array::{
    Array, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PREDICTION_ARTIFACT_DIRNAME: &str = "prediction_artifacts";
const PREDICTION_METADATA_FILENAME: &str = "metadata.json";
const PREDICTIONS_FILENAME: &str = "final_predictions.parquet";
const SIGNAL_LABELS_FILENAME: &str = "signal_labels.parquet";
const BACKTEST_LABELS_FILENAME: &str = "backtest_labels.parquet";
const TRAINING_SUMMARY_FILENAME: &str = "training_summary.csv";

const BASELINE_FILES: &[(&str, &str)] = &[
    (
        "avg_factor_baseline",
        "avg_factor_baseline_predictions.parquet",
    ),
    (
        "sign_aligned_factor_baseline",
        "sign_aligned_factor_baseline_predictions.parquet",
    ),
    (
        "rank_avg_factor_baseline",
        "rank_avg_factor_baseline_predictions.parquet",
    ),
    (
        "rank_ic_weighted_factor_baseline",
        "rank_ic_weighted_factor_baseline_predictions.parquet",
    ),
];

#[derive(Debug, Clone)]
struct LongRow {
    date_ns: i64,
    instrument: String,
    value: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct MatrixFrame {
    pub(crate) dates_ns: Vec<i64>,
    pub(crate) instruments: Vec<String>,
    pub(crate) values: Vec<f64>,
    pub(crate) input_rows: usize,
    pub(crate) finite_values: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PredictionBundle {
    pub(crate) dir: PathBuf,
    pub(crate) metadata: Value,
    pub(crate) final_predictions: MatrixFrame,
    pub(crate) signal_labels: MatrixFrame,
    pub(crate) backtest_labels: MatrixFrame,
    pub(crate) baselines: Vec<OptionalMatrix>,
    pub(crate) training_summary_records: Vec<BTreeMap<String, String>>,
}

#[derive(Debug, Clone)]
pub(crate) struct OptionalMatrix {
    pub(crate) name: String,
    pub(crate) filename: String,
    pub(crate) matrix: Option<MatrixFrame>,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixSummary {
    rows: usize,
    dates: usize,
    instruments: usize,
    matrix_cells: usize,
    finite_values: usize,
    nonfinite_values: usize,
    missing_cells: usize,
    density: f64,
    min_datetime_ns: Option<i64>,
    max_datetime_ns: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
struct OptionalMatrixSummary {
    name: String,
    filename: String,
    present: bool,
    summary: Option<MatrixSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct BundleInspectSummary {
    bundle_dir: String,
    metadata_keys: usize,
    metadata_selected_feature_count: usize,
    metadata_core_fields: BTreeMap<String, Value>,
    core_axes_match: bool,
    final_predictions: MatrixSummary,
    signal_labels: MatrixSummary,
    backtest_labels: MatrixSummary,
    baselines: Vec<OptionalMatrixSummary>,
    training_summary_rows: usize,
}

#[derive(Debug, Clone)]
struct InspectOptions {
    bundle_dir: PathBuf,
    json: bool,
}

pub(crate) fn inspect(args: &[String]) -> Result<ExitCode, String> {
    let options = parse_inspect_options(args)?;
    let bundle = read_prediction_bundle(&options.bundle_dir)?;
    let summary = inspect_summary(&bundle);
    if options.json {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        serde_json::to_writer_pretty(&mut handle, &summary)
            .map_err(|err| format!("failed to write JSON summary: {err}"))?;
        println!();
    } else {
        print_human_summary(&summary);
    }
    Ok(ExitCode::SUCCESS)
}

fn parse_inspect_options(args: &[String]) -> Result<InspectOptions, String> {
    let mut bundle_dir: Option<PathBuf> = None;
    let mut json = false;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--bundle" | "--bundle-dir" | "--load-predictions-dir" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --bundle".to_owned())?;
                bundle_dir = Some(PathBuf::from(raw));
            }
            "--json" => json = true,
            "-h" | "--help" => return Err(inspect_usage().to_owned()),
            value if !value.starts_with('-') && bundle_dir.is_none() => {
                bundle_dir = Some(PathBuf::from(value));
            }
            other => {
                return Err(format!(
                    "unknown inspect-bundle option: {other}\n\n{}",
                    inspect_usage()
                ))
            }
        }
        idx += 1;
    }
    let bundle_dir =
        bundle_dir.ok_or_else(|| format!("--bundle is required\n\n{}", inspect_usage()))?;
    Ok(InspectOptions { bundle_dir, json })
}

fn inspect_usage() -> &'static str {
    "\
Usage:
  ai4stock-backtest inspect-bundle --bundle <PATH> [--json]

<PATH> may be either a run directory or the prediction_artifacts directory itself.
"
}

pub(crate) fn read_prediction_bundle(raw_path: &Path) -> Result<PredictionBundle, String> {
    let dir = resolve_prediction_artifact_dir(raw_path)?;
    let metadata = read_metadata(&dir.join(PREDICTION_METADATA_FILENAME))?;
    let final_predictions = read_matrix(&dir.join(PREDICTIONS_FILENAME), "prediction")?;
    let signal_labels = read_matrix(&dir.join(SIGNAL_LABELS_FILENAME), "label")?;
    let backtest_labels = read_matrix(&dir.join(BACKTEST_LABELS_FILENAME), "label")?;
    let baselines = BASELINE_FILES
        .iter()
        .map(|(name, filename)| {
            let path = dir.join(filename);
            let matrix = if path.exists() {
                Some(read_matrix(&path, "prediction")?)
            } else {
                None
            };
            Ok(OptionalMatrix {
                name: (*name).to_owned(),
                filename: (*filename).to_owned(),
                matrix,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let training_summary_records =
        read_training_summary_records(&dir.join(TRAINING_SUMMARY_FILENAME))?;
    Ok(PredictionBundle {
        dir,
        metadata,
        final_predictions,
        signal_labels,
        backtest_labels,
        baselines,
        training_summary_records,
    })
}

fn resolve_prediction_artifact_dir(raw_path: &Path) -> Result<PathBuf, String> {
    if raw_path.is_dir() && raw_path.join(PREDICTION_METADATA_FILENAME).is_file() {
        return Ok(raw_path.to_path_buf());
    }
    let nested = raw_path.join(PREDICTION_ARTIFACT_DIRNAME);
    if nested.is_dir() && nested.join(PREDICTION_METADATA_FILENAME).is_file() {
        return Ok(nested);
    }
    Err(format!(
        "prediction artifact directory not found under {}; expected {} directly or under {}/",
        raw_path.display(),
        PREDICTION_METADATA_FILENAME,
        PREDICTION_ARTIFACT_DIRNAME
    ))
}

fn read_metadata(path: &Path) -> Result<Value, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    serde_json::from_reader(file)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn read_training_summary_records(path: &Path) -> Result<Vec<BTreeMap<String, String>>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut reader = csv::Reader::from_path(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|err| format!("failed to parse {} headers: {err}", path.display()))?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
        let mut row = BTreeMap::new();
        for (header, value) in headers.iter().zip(record.iter()) {
            row.insert(header.clone(), value.to_owned());
        }
        rows.push(row);
    }
    Ok(rows)
}

fn read_matrix(path: &Path, value_column: &str) -> Result<MatrixFrame, String> {
    let rows = read_long_rows(path, value_column)?;
    MatrixFrame::from_rows(rows, path)
}

fn read_long_rows(path: &Path, value_column: &str) -> Result<Vec<LongRow>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| format!("failed to open parquet {}: {err}", path.display()))?
        .with_batch_size(65_536)
        .build()
        .map_err(|err| {
            format!(
                "failed to build parquet reader for {}: {err}",
                path.display()
            )
        })?;

    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        append_batch_rows(&mut rows, &batch, value_column, path)?;
    }
    Ok(rows)
}

fn append_batch_rows(
    rows: &mut Vec<LongRow>,
    batch: &RecordBatch,
    value_column: &str,
    path: &Path,
) -> Result<(), String> {
    let dates = column(batch, "datetime", path)?;
    let instruments = column(batch, "instrument", path)?;
    let values = column(batch, value_column, path)?;
    for row in 0..batch.num_rows() {
        rows.push(LongRow {
            date_ns: datetime_ns_at(dates, row, path)?,
            instrument: instrument_at(instruments, row, path)?,
            value: value_at(values, row, path)?,
        });
    }
    Ok(())
}

fn column<'a>(batch: &'a RecordBatch, name: &str, path: &Path) -> Result<&'a dyn Array, String> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| format!("{} is missing required column {name}", path.display()))?;
    Ok(batch.column(idx).as_ref())
}

fn datetime_ns_at(array: &dyn Array, row: usize, path: &Path) -> Result<i64, String> {
    if array.is_null(row) {
        return Err(format!("{} has null datetime at row {row}", path.display()));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Ok(values.value(row) * 1_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return Ok(values.value(row) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        return Ok(values.value(row) * 1_000_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date32Array>() {
        return Ok(values.value(row) as i64 * 86_400_000_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date64Array>() {
        return Ok(values.value(row) * 1_000_000);
    }
    Err(format!(
        "{} has unsupported datetime type {:?}",
        path.display(),
        array.data_type()
    ))
}

fn instrument_at(array: &dyn Array, row: usize, path: &Path) -> Result<String, String> {
    if array.is_null(row) {
        return Err(format!(
            "{} has null instrument at row {row}",
            path.display()
        ));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row).to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row).to_owned());
    }
    Err(format!(
        "{} has unsupported instrument type {:?}",
        path.display(),
        array.data_type()
    ))
}

fn value_at(array: &dyn Array, row: usize, path: &Path) -> Result<f64, String> {
    if array.is_null(row) {
        return Ok(f64::NAN);
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(values.value(row) as f64);
    }
    Err(format!(
        "{} has unsupported value type {:?}",
        path.display(),
        array.data_type()
    ))
}

impl MatrixFrame {
    fn from_rows(rows: Vec<LongRow>, path: &Path) -> Result<Self, String> {
        let mut date_set = BTreeSet::new();
        let mut instrument_set = BTreeSet::new();
        for row in &rows {
            date_set.insert(row.date_ns);
            instrument_set.insert(row.instrument.clone());
        }
        let dates_ns = date_set.into_iter().collect::<Vec<_>>();
        let instruments = instrument_set.into_iter().collect::<Vec<_>>();
        let date_pos = dates_ns
            .iter()
            .enumerate()
            .map(|(idx, value)| (*value, idx))
            .collect::<BTreeMap<_, _>>();
        let instrument_pos = instruments
            .iter()
            .enumerate()
            .map(|(idx, value)| (value.as_str(), idx))
            .collect::<BTreeMap<_, _>>();
        let n_cols = instruments.len();
        let mut values = vec![f64::NAN; dates_ns.len() * n_cols];
        let mut filled = vec![false; values.len()];
        let mut finite_values = 0usize;
        let input_rows = rows.len();
        for row in rows {
            let row_idx = *date_pos
                .get(&row.date_ns)
                .ok_or_else(|| "internal date index error".to_owned())?;
            let col_idx = *instrument_pos
                .get(row.instrument.as_str())
                .ok_or_else(|| "internal instrument index error".to_owned())?;
            let idx = row_idx * n_cols + col_idx;
            if filled[idx] {
                return Err(format!(
                    "{} has duplicate row for datetime_ns={} instrument={}",
                    path.display(),
                    row.date_ns,
                    row.instrument
                ));
            }
            filled[idx] = true;
            values[idx] = row.value;
            if row.value.is_finite() {
                finite_values += 1;
            }
        }
        Ok(Self {
            dates_ns,
            instruments,
            values,
            input_rows,
            finite_values,
        })
    }

    pub(crate) fn axes_match(&self, other: &Self) -> bool {
        self.dates_ns == other.dates_ns && self.instruments == other.instruments
    }

    fn summary(&self) -> MatrixSummary {
        let matrix_cells = self.values.len();
        MatrixSummary {
            rows: self.input_rows,
            dates: self.dates_ns.len(),
            instruments: self.instruments.len(),
            matrix_cells,
            finite_values: self.finite_values,
            nonfinite_values: self.input_rows.saturating_sub(self.finite_values),
            missing_cells: matrix_cells.saturating_sub(self.input_rows),
            density: if matrix_cells == 0 {
                0.0
            } else {
                self.finite_values as f64 / matrix_cells as f64
            },
            min_datetime_ns: self.dates_ns.first().copied(),
            max_datetime_ns: self.dates_ns.last().copied(),
        }
    }
}

fn inspect_summary(bundle: &PredictionBundle) -> BundleInspectSummary {
    let final_summary = bundle.final_predictions.summary();
    let signal_summary = bundle.signal_labels.summary();
    let backtest_summary = bundle.backtest_labels.summary();
    let baselines = bundle
        .baselines
        .iter()
        .map(|baseline| OptionalMatrixSummary {
            name: baseline.name.clone(),
            filename: baseline.filename.clone(),
            present: baseline.matrix.is_some(),
            summary: baseline.matrix.as_ref().map(MatrixFrame::summary),
        })
        .collect::<Vec<_>>();
    BundleInspectSummary {
        bundle_dir: bundle.dir.to_string_lossy().into_owned(),
        metadata_keys: bundle.metadata.as_object().map_or(0, serde_json::Map::len),
        metadata_selected_feature_count: bundle
            .metadata
            .get("selected_features")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        metadata_core_fields: metadata_core_fields(&bundle.metadata),
        core_axes_match: bundle.final_predictions.axes_match(&bundle.signal_labels)
            && bundle.final_predictions.axes_match(&bundle.backtest_labels),
        final_predictions: final_summary,
        signal_labels: signal_summary,
        backtest_labels: backtest_summary,
        baselines,
        training_summary_rows: bundle.training_summary_records.len(),
    }
}

fn metadata_core_fields(metadata: &Value) -> BTreeMap<String, Value> {
    [
        "signal_horizon",
        "retrain_step",
        "train_days",
        "valid_days",
        "label_embargo_days",
        "test_start",
        "test_end",
        "fusion_enabled",
        "fusion_mode",
    ]
    .into_iter()
    .filter_map(|key| {
        metadata
            .get(key)
            .cloned()
            .map(|value| (key.to_owned(), value))
    })
    .collect()
}

fn print_human_summary(summary: &BundleInspectSummary) {
    println!("[bundle] {}", summary.bundle_dir);
    println!(
        "[metadata] keys={} selected_features={} core={}",
        summary.metadata_keys,
        summary.metadata_selected_feature_count,
        serde_json::to_string(&summary.metadata_core_fields).unwrap_or_else(|_| "{}".to_owned())
    );
    println!("[core_axes_match] {}", summary.core_axes_match);
    print_matrix_summary("final_predictions", &summary.final_predictions);
    print_matrix_summary("signal_labels", &summary.signal_labels);
    print_matrix_summary("backtest_labels", &summary.backtest_labels);
    for baseline in &summary.baselines {
        if let Some(matrix) = &baseline.summary {
            print_matrix_summary(&baseline.name, matrix);
        } else {
            println!("[baseline] {} present=false", baseline.name);
        }
    }
    println!("[training_summary] rows={}", summary.training_summary_rows);
}

fn print_matrix_summary(name: &str, summary: &MatrixSummary) {
    println!(
        "[matrix] {name} rows={} dates={} instruments={} cells={} finite={} nonfinite={} missing={} density={:.6} min_ns={} max_ns={}",
        summary.rows,
        summary.dates,
        summary.instruments,
        summary.matrix_cells,
        summary.finite_values,
        summary.nonfinite_values,
        summary.missing_cells,
        summary.density,
        summary
            .min_datetime_ns
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_owned()),
        summary
            .max_datetime_ns
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_owned())
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_dense_matrix_from_unsorted_rows() {
        let rows = vec![
            LongRow {
                date_ns: 20,
                instrument: "b".to_owned(),
                value: 2.0,
            },
            LongRow {
                date_ns: 10,
                instrument: "a".to_owned(),
                value: 1.0,
            },
            LongRow {
                date_ns: 20,
                instrument: "a".to_owned(),
                value: f64::NAN,
            },
        ];
        let matrix = MatrixFrame::from_rows(rows, Path::new("test.parquet")).unwrap();
        assert_eq!(matrix.dates_ns, [10, 20]);
        assert_eq!(matrix.instruments, ["a", "b"]);
        assert_eq!(matrix.values.len(), 4);
        assert_eq!(matrix.values[0], 1.0);
        assert!(matrix.values[1].is_nan());
        assert!(matrix.values[2].is_nan());
        assert_eq!(matrix.values[3], 2.0);

        let summary = matrix.summary();
        assert_eq!(summary.rows, 3);
        assert_eq!(summary.finite_values, 2);
        assert_eq!(summary.nonfinite_values, 1);
        assert_eq!(summary.missing_cells, 1);
    }

    #[test]
    fn detects_duplicate_matrix_cells() {
        let rows = vec![
            LongRow {
                date_ns: 10,
                instrument: "a".to_owned(),
                value: 1.0,
            },
            LongRow {
                date_ns: 10,
                instrument: "a".to_owned(),
                value: 2.0,
            },
        ];
        assert!(MatrixFrame::from_rows(rows, Path::new("test.parquet")).is_err());
    }
}
