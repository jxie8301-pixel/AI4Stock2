from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_required_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def read_required_json(run_dir: Path, filename: str) -> dict[str, Any]:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    if isinstance(value, dict) and "risk" in value:
        return value["risk"]
    return value


def parse_bucket_ids(raw: str, *, option_name: str = "--middle-buckets") -> set[int]:
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not values:
        raise ValueError(f"{option_name} must contain at least one bucket id.")
    return values


def bucket_shape(frame: pd.DataFrame, *, top_bucket: int, middle_buckets: set[int]) -> dict[str, Any]:
    if "bucket" not in frame or "label_mean" not in frame:
        raise ValueError("Bucket report must contain 'bucket' and 'label_mean'.")
    numeric = frame.copy()
    numeric["bucket"] = pd.to_numeric(numeric["bucket"], errors="coerce")
    numeric["label_mean"] = pd.to_numeric(numeric["label_mean"], errors="coerce")
    numeric = numeric.dropna(subset=["bucket", "label_mean"]).sort_values("bucket")
    numeric["bucket"] = numeric["bucket"].astype(int)
    if top_bucket not in set(numeric["bucket"]):
        raise ValueError(f"Top bucket {top_bucket} not found in bucket report.")

    bottom_bucket = int(numeric["bucket"].max())
    by_bucket = numeric.set_index("bucket")
    top_label = float(by_bucket.at[top_bucket, "label_mean"])
    bottom_label = float(by_bucket.at[bottom_bucket, "label_mean"])
    middle = numeric[numeric["bucket"].isin(middle_buckets)]
    middle_mean = float(middle["label_mean"].mean()) if not middle.empty else float("nan")
    middle_best = float(middle["label_mean"].max()) if not middle.empty else float("nan")
    best_row = numeric.loc[numeric["label_mean"].idxmax()]
    worst_row = numeric.loc[numeric["label_mean"].idxmin()]
    ranked = numeric.sort_values("label_mean", ascending=False).reset_index(drop=True)
    top_rank = int(ranked.index[ranked["bucket"] == top_bucket][0]) + 1
    return {
        "top_bucket": top_bucket,
        "bottom_bucket": bottom_bucket,
        "top_label_mean": top_label,
        "bottom_label_mean": bottom_label,
        "top_minus_bottom": top_label - bottom_label,
        "middle_label_mean": middle_mean,
        "middle_best_label_mean": middle_best,
        "top_minus_middle_mean": top_label - middle_mean,
        "top_minus_middle_best": top_label - middle_best,
        "best_bucket": int(best_row["bucket"]),
        "best_bucket_label_mean": float(best_row["label_mean"]),
        "worst_bucket": int(worst_row["bucket"]),
        "worst_bucket_label_mean": float(worst_row["label_mean"]),
        "top_bucket_label_rank": top_rank,
        "bucket_label_spearman": float(numeric["bucket"].corr(numeric["label_mean"], method="spearman")),
    }
