from __future__ import annotations

import numpy as np

from src.data import causal_ema, detect_gate_segments, preprocess_power
from src.metrics import aggregate_predictions_by_shot, point_metrics


def test_causal_ema_does_not_look_ahead() -> None:
  original = np.zeros(20)
  changed_future = original.copy()
  changed_future[10:] = 100.0
  first = causal_ema(original, smoothing_ms=4)
  second = causal_ema(changed_future, smoothing_ms=4)
  np.testing.assert_allclose(first[:10], second[:10])


def test_power_negative_values_become_zero() -> None:
  time = np.arange(1000, dtype=float) * 0.1
  values = np.full(1000, -2.0)
  values[500:800] = 100.0
  cleaned, _, _ = preprocess_power(time, values, duration_ms=100, smoothing_ms=5)
  assert np.nanmin(cleaned) >= 0.0


def test_gate_requires_minimum_duration() -> None:
  power = np.zeros(500)
  power[20:100] = 100.0
  power[200:450] = 100.0
  segments, _ = detect_gate_segments(
    power,
    threshold_ratio=0.1,
    off_ratio=0.5,
    maximum_gap_ms=10,
    minimum_duration_ms=180,
  )
  assert segments == [(200, 450)]


def test_metrics_perfect_prediction() -> None:
  actual = np.linspace(0.0, 1.0, 20)
  metrics = point_metrics(actual, actual, scale=1.0)
  assert metrics["mae"] == 0.0
  assert metrics["rmse"] == 0.0
  assert metrics["nrmse"] == 0.0
  assert np.isclose(metrics["pearson"], 1.0)
  assert np.isclose(metrics["r2"], 1.0)


def test_shot_metrics_concatenate_short_windows() -> None:
  actual = np.asarray([[0.0, 0.0], [1.0, 1.0]])
  predicted = {"model": actual.copy()}
  rows = aggregate_predictions_by_shot(
    shots=np.asarray(["1", "1"]),
    scales=np.asarray([1.0, 1.0]),
    actual=actual,
    predictions=predicted,
  )
  assert len(rows) == 1
  assert np.isclose(rows[0]["pearson"], 1.0)
  assert np.isclose(rows[0]["r2"], 1.0)
