from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("mae", "rmse", "nrmse", "pearson", "r2")


def point_metrics(actual: np.ndarray, predicted: np.ndarray, scale: float) -> dict[str, float]:
  actual = np.asarray(actual, dtype=np.float64)
  predicted = np.asarray(predicted, dtype=np.float64)
  error = predicted - actual
  mae = float(np.mean(np.abs(error)))
  rmse = float(np.sqrt(np.mean(error**2)))
  nrmse = float(rmse / scale)
  if np.std(actual) > 0 and np.std(predicted) > 0:
    pearson = float(np.corrcoef(actual, predicted)[0, 1])
  else:
    pearson = np.nan
  denominator = float(np.sum((actual - np.mean(actual)) ** 2))
  r2 = float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else np.nan
  return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "pearson": pearson, "r2": r2}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    path.write_text("", encoding="utf-8")
    return
  fields: list[str] = []
  for row in rows:
    for key in row:
      if key not in fields:
        fields.append(key)
  with path.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def aggregate_by_shot(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
  for row in window_rows:
    grouped[(str(row["model"]), str(row["shot"]))].append(row)
  output = []
  for (model, shot), rows in sorted(grouped.items()):
    summary: dict[str, Any] = {"model": model, "shot": shot, "windows": len(rows)}
    for metric in METRICS:
      values = np.asarray([row[metric] for row in rows], dtype=np.float64)
      summary[metric] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
    output.append(summary)
  return output


def aggregate_predictions_by_shot(
  shots: np.ndarray,
  scales: np.ndarray,
  actual: np.ndarray,
  predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
  """Compute shot metrics after concatenating all horizons for that shot.

  Pearson correlation and R² are unstable on an individual 20-point window when
  the truth is nearly flat. Concatenating all evaluated horizons within a shot
  gives those shape metrics a meaningful variance while preserving paired model
  comparisons.
  """
  output: list[dict[str, Any]] = []
  shot_strings = np.asarray(shots, dtype=str)
  for model, model_predictions in predictions.items():
    for shot in sorted(set(shot_strings), key=int):
      indices = np.flatnonzero(shot_strings == shot)
      scale = float(np.median(scales[indices]))
      metrics = point_metrics(
        actual[indices].reshape(-1),
        model_predictions[indices].reshape(-1),
        scale,
      )
      output.append({"model": model, "shot": shot, "windows": int(indices.size), **metrics})
  return output


def aggregate_summary(by_shot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for row in by_shot_rows:
    grouped[str(row["model"])].append(row)
  output = []
  for model, rows in sorted(grouped.items()):
    summary: dict[str, Any] = {
      "model": model,
      "shots": len(rows),
      "windows": sum(int(row["windows"]) for row in rows),
    }
    for metric in METRICS:
      values = np.asarray([row[metric] for row in rows], dtype=np.float64)
      summary[metric] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
    output.append(summary)

  indexed = {row["model"]: row for row in output}
  baseline = indexed.get("target_only")
  if baseline:
    for row in output:
      if row["model"] == "target_only":
        continue
      for metric in ("mae", "rmse", "nrmse"):
        base_value = float(baseline[metric])
        row[f"{metric}_improvement_pct_vs_target_only"] = (
          100.0 * (base_value - float(row[metric])) / base_value if base_value else np.nan
        )
  return output
