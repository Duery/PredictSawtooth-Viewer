from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data import PreparedData, Window, iter_chunks
from src.metrics import aggregate_predictions_by_shot, aggregate_summary, point_metrics, write_csv


def _persistence(window: Window) -> np.ndarray:
  return np.full(window.target_future.shape, window.target_context[-1], dtype=np.float32)


def _model_predictions(
  forecaster: Any,
  windows: list[Window],
  horizon: int,
  use_covariates: bool,
  chunk_size: int,
) -> list[np.ndarray]:
  predictions: list[np.ndarray] = []
  for chunk_index, chunk in enumerate(iter_chunks(windows, chunk_size), start=1):
    contexts = [window.target_context for window in chunk]
    past_only = [window.covariates for window in chunk] if use_covariates else None
    outputs = forecaster.predict_batch(
      contexts=contexts,
      horizon=horizon,
      past_only_covariates=past_only,
      return_quantiles=False,
      use_symmetric_averaging=False,
      make_positive=False,
      use_znorm=False,
    )
    predictions.extend(np.asarray(output.forecast, dtype=np.float32) for output in outputs)
    if chunk_index % 10 == 0:
      print(f"  completed {min(chunk_index * chunk_size, len(windows))}/{len(windows)} windows")
  return predictions


def run_evaluation(
  config: dict[str, Any],
  prepared: PreparedData,
  output_dir: Path,
) -> list[dict[str, Any]]:
  from timesfm3 import TimesFM3Forecaster

  windows = prepared.windows
  if not windows:
    raise RuntimeError("No evaluation windows were prepared.")
  model_config = config["model"]
  horizon = int(config["window"]["horizon_ms"])
  batch_size = int(model_config["batch_size"])

  print(f"Loading {model_config['id']} on {model_config['device']} ...")
  forecaster = TimesFM3Forecaster.from_pretrained(
    str(model_config["id"]),
    device=str(model_config["device"]),
    per_core_batch_size=batch_size,
    cache_dir=str(config["paths"]["model_cache"]),
  )

  all_predictions: dict[str, np.ndarray] = {
    "persistence": np.stack([_persistence(window) for window in windows]),
  }
  print("Running target-only TimesFM forecast ...")
  all_predictions["target_only"] = np.stack(
    _model_predictions(forecaster, windows, horizon, False, batch_size)
  )
  print("Running all-information TimesFM forecast ...")
  all_predictions["all_information"] = np.stack(
    _model_predictions(forecaster, windows, horizon, True, batch_size)
  )

  actual = np.stack([window.target_future for window in windows])
  window_rows: list[dict[str, Any]] = []
  for model_name, predictions in all_predictions.items():
    for index, window in enumerate(windows):
      row: dict[str, Any] = {
        "model": model_name,
        "shot": window.shot,
        "anchor_ms": window.anchor_ms,
      }
      row.update(point_metrics(actual[index], predictions[index], window.nrmse_scale))
      window_rows.append(row)

  shots = np.asarray([window.shot for window in windows])
  scales = np.asarray([window.nrmse_scale for window in windows], dtype=np.float64)
  by_shot = aggregate_predictions_by_shot(shots, scales, actual, all_predictions)
  summary = aggregate_summary(by_shot)
  write_csv(output_dir / "window_metrics.csv", window_rows)
  write_csv(output_dir / "by_shot.csv", by_shot)
  write_csv(output_dir / "summary.csv", summary)
  np.savez_compressed(
    output_dir / "predictions.npz",
    actual=actual,
    **all_predictions,
    shot=shots,
    anchor_ms=np.asarray([window.anchor_ms for window in windows]),
    nrmse_scale=scales,
  )
  return summary
