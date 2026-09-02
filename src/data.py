from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Window:
  shot: str
  anchor_ms: int
  target_context: np.ndarray
  target_future: np.ndarray
  covariates: np.ndarray
  nrmse_scale: float


@dataclass(frozen=True)
class PreparedData:
  windows: list[Window]
  quality_rows: list[dict[str, Any]]
  channel_names: list[str]


def _as_1d(value: Any) -> np.ndarray | None:
  if value is None:
    return None
  array = np.asarray(value).squeeze()
  if array.ndim != 1 or array.size == 0:
    return None
  return array.astype(np.float64, copy=False)


def load_mat_series(path: Path, duration_ms: int) -> tuple[np.ndarray, np.ndarray, bool]:
  if not path.is_file():
    raise FileNotFoundError(f"missing file: {path}")
  data = loadmat(path, variable_names=["t", "y"], squeeze_me=True)
  values = _as_1d(data.get("y"))
  if values is None:
    raise ValueError("missing non-empty y")

  time = _as_1d(data.get("t"))
  inferred = time is None or time.size != values.size
  if inferred:
    if values.size > 1 and (values.size - 1) % duration_ms == 0:
      time = np.linspace(0.0, float(duration_ms), values.size, endpoint=True)
    elif values.size % duration_ms == 0:
      time = np.linspace(0.0, float(duration_ms), values.size, endpoint=False)
    else:
      time = np.linspace(0.0, float(duration_ms), values.size, endpoint=True)

  assert time is not None
  finite_time = np.isfinite(time)
  time = time[finite_time]
  values = values[finite_time]
  if time.size < 2:
    raise ValueError("too few valid timestamps")

  if np.any(np.diff(time) < 0):
    order = np.argsort(time, kind="stable")
    time = time[order]
    values = values[order]
  return time, values, inferred


def _bin_sorted_series(
  time_ms: np.ndarray,
  values: np.ndarray,
  duration_ms: int,
  reducer: str,
) -> np.ndarray:
  valid = (
    np.isfinite(time_ms)
    & np.isfinite(values)
    & (time_ms >= 0.0)
    & (time_ms < float(duration_ms))
  )
  time_ms = time_ms[valid]
  values = values[valid]
  output = np.full(duration_ms, np.nan, dtype=np.float64)
  if time_ms.size == 0:
    return output

  edges = np.arange(duration_ms + 1, dtype=np.float64)
  left = np.searchsorted(time_ms, edges[:-1], side="left")
  right = np.searchsorted(time_ms, edges[1:], side="left")
  for index, (start, stop) in enumerate(zip(left, right, strict=True)):
    if stop <= start:
      continue
    block = values[start:stop]
    output[index] = np.median(block) if reducer == "median" else np.mean(block)
  return output


def causal_ema(values: np.ndarray, smoothing_ms: int) -> np.ndarray:
  if smoothing_ms <= 1:
    return values.copy()
  alpha = 1.0 - np.exp(-1.0 / float(smoothing_ms))
  output = np.full_like(values, np.nan, dtype=np.float64)
  state: float | None = None
  for index, value in enumerate(values):
    if not np.isfinite(value):
      continue
    state = float(value) if state is None else alpha * float(value) + (1.0 - alpha) * state
    output[index] = state
  return output


def trailing_median(values: np.ndarray, window_ms: int) -> np.ndarray:
  output = np.full_like(values, np.nan, dtype=np.float64)
  for index in range(values.size):
    block = values[max(0, index - window_ms + 1) : index + 1]
    finite = block[np.isfinite(block)]
    if finite.size:
      output[index] = np.median(finite)
  return output


def preprocess_target(
  time_ms: np.ndarray,
  values: np.ndarray,
  duration_ms: int,
  smoothing_ms: int,
  baseline_ms: int,
) -> np.ndarray:
  binned = _bin_sorted_series(time_ms, values, duration_ms, reducer="median")
  baseline_block = binned[:baseline_ms]
  finite = baseline_block[np.isfinite(baseline_block)]
  if finite.size:
    binned = binned - np.median(finite)
  return causal_ema(binned, smoothing_ms)


def preprocess_diagnostic(
  signal: str,
  time_ms: np.ndarray,
  values: np.ndarray,
  duration_ms: int,
) -> np.ndarray:
  reducer = "mean" if signal in {"IP_VV_2M", "WE_2M"} else "median"
  return _bin_sorted_series(time_ms, values, duration_ms, reducer=reducer)


def preprocess_power(
  time_ms: np.ndarray,
  values: np.ndarray,
  duration_ms: int,
  smoothing_ms: int,
) -> tuple[np.ndarray, float, float]:
  in_range = (
    np.isfinite(time_ms)
    & np.isfinite(values)
    & (time_ms >= 0.0)
    & (time_ms < float(duration_ms))
  )
  clean = values[in_range].astype(np.float64, copy=True)
  clean_time = time_ms[in_range]
  if clean.size == 0:
    return np.full(duration_ms, np.nan), np.nan, np.nan

  baseline = float(np.median(clean))
  mad = float(np.median(np.abs(clean - baseline)))
  noise = 1.4826 * mad
  clean = clean - baseline
  floor = max(3.0 * noise, np.finfo(np.float64).eps)
  clean[clean <= floor] = 0.0
  clean = np.clip(clean, 0.0, None)
  binned = _bin_sorted_series(clean_time, clean, duration_ms, reducer="mean")
  binned = trailing_median(binned, smoothing_ms)
  return binned, baseline, noise


def detect_gate_segments(
  power: np.ndarray,
  threshold_ratio: float,
  off_ratio: float,
  maximum_gap_ms: int,
  minimum_duration_ms: int,
) -> tuple[list[tuple[int, int]], float]:
  finite = power[np.isfinite(power)]
  positive = finite[finite > 0]
  if positive.size == 0:
    return [], np.nan
  on_threshold = float(threshold_ratio * np.percentile(positive, 95))
  if on_threshold <= 0:
    return [], on_threshold
  off_threshold = off_ratio * on_threshold

  raw: list[tuple[int, int]] = []
  state = False
  start = 0
  for index, value in enumerate(np.nan_to_num(power, nan=0.0)):
    if not state and value >= on_threshold:
      state = True
      start = index
    elif state and value <= off_threshold:
      raw.append((start, index))
      state = False
  if state:
    raw.append((start, power.size))

  merged: list[tuple[int, int]] = []
  for start, end in raw:
    if merged and start - merged[-1][1] <= maximum_gap_ms:
      merged[-1] = (merged[-1][0], end)
    else:
      merged.append((start, end))
  return [(start, end) for start, end in merged if end - start >= minimum_duration_ms], on_threshold


def _signal_path(data_dir: Path, shot: str, signal: str) -> Path:
  subdir = "heating_power" if signal.startswith("Pow_") else "dig_data"
  return data_dir / subdir / f"{shot}_{signal}.mat"


def _target_shots(data_dir: Path, target: str) -> list[str]:
  shots = []
  for path in (data_dir / "dig_data").glob(f"*_{target}.mat"):
    shots.append(path.name.split("_", 1)[0])
  return sorted(set(shots), key=int)


def _finite_slice(values: np.ndarray, start: int, end: int) -> bool:
  return end <= values.size and bool(np.all(np.isfinite(values[start:end])))


def prepare_windows(config: dict[str, Any], max_shots: int | None = None) -> PreparedData:
  data_dir = Path(config["paths"]["data_dir"])
  duration_ms = int(config["data"]["duration_ms"])
  target_name = str(config["data"]["target"])
  diagnostics = list(config["channels"]["diagnostics"])
  heating = list(config["channels"]["heating"])
  channel_names = diagnostics + heating
  gate_name = str(config["nbi_gate"]["signal"])
  quality_rows: list[dict[str, Any]] = []
  windows: list[Window] = []

  shots = _target_shots(data_dir, target_name)
  if max_shots is not None:
    shots = shots[:max_shots]

  for shot in shots:
    quality: dict[str, Any] = {"shot": shot, "status": "excluded", "reason": "", "windows": 0}
    try:
      target_path = _signal_path(data_dir, shot, target_name)
      target_t, target_y, target_inferred = load_mat_series(target_path, duration_ms)
      target = preprocess_target(
        target_t,
        target_y,
        duration_ms,
        int(config["data"]["target_smoothing_ms"]),
        int(config["data"]["target_baseline_ms"]),
      )

      gate_path = _signal_path(data_dir, shot, gate_name)
      gate_t, gate_y, _ = load_mat_series(gate_path, duration_ms)
      gate, gate_baseline, gate_noise = preprocess_power(
        gate_t,
        gate_y,
        duration_ms,
        int(config["nbi_gate"]["smoothing_ms"]),
      )
      segments, gate_threshold = detect_gate_segments(
        gate,
        float(config["nbi_gate"]["threshold_ratio"]),
        float(config["nbi_gate"]["off_ratio"]),
        int(config["nbi_gate"]["maximum_gap_ms"]),
        int(config["nbi_gate"]["minimum_on_duration_ms"]),
      )
      if not segments:
        raise ValueError("no qualifying NBI1-on segment")

      covariate_values: list[np.ndarray] = []
      for signal in diagnostics:
        path = _signal_path(data_dir, shot, signal)
        time, values, _ = load_mat_series(path, duration_ms)
        covariate_values.append(preprocess_diagnostic(signal, time, values, duration_ms))
      for signal in heating:
        if signal == gate_name:
          covariate_values.append(gate)
          continue
        path = _signal_path(data_dir, shot, signal)
        time, values, _ = load_mat_series(path, duration_ms)
        power, _, _ = preprocess_power(
          time,
          values,
          duration_ms,
          int(config["nbi_gate"]["smoothing_ms"]),
        )
        covariate_values.append(power)
      covariates = np.stack(covariate_values, axis=0)

      active_target = np.concatenate([target[start:end] for start, end in segments])
      active_target = active_target[np.isfinite(active_target)]
      if active_target.size == 0:
        raise ValueError("target is empty inside NBI1-on segment")
      scale = float(np.percentile(active_target, 95) - np.percentile(active_target, 5))
      if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        scale = float(np.std(active_target))
      if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        scale = 1.0

      context = int(config["window"]["context_ms"])
      horizon = int(config["window"]["horizon_ms"])
      stride = int(config["window"]["stride_ms"])
      shot_windows = 0
      for segment_start, segment_end in segments:
        first_anchor = segment_start + context
        last_anchor = segment_end - horizon
        for anchor in range(first_anchor, last_anchor + 1, stride):
          context_start = anchor - context
          future_end = anchor + horizon
          if not _finite_slice(target, context_start, future_end):
            continue
          if not all(_finite_slice(values, context_start, anchor) for values in covariates):
            continue
          windows.append(
            Window(
              shot=shot,
              anchor_ms=anchor,
              target_context=target[context_start:anchor].astype(np.float32),
              target_future=target[anchor:future_end].astype(np.float32),
              covariates=covariates[:, context_start:anchor].astype(np.float32),
              nrmse_scale=scale,
            )
          )
          shot_windows += 1

      if shot_windows == 0:
        raise ValueError("no complete windows after channel alignment")
      quality.update(
        status="included",
        reason="",
        windows=shot_windows,
        target_time_inferred=target_inferred,
        nbi_segments=len(segments),
        nbi_gate_threshold=gate_threshold,
        nbi_baseline=gate_baseline,
        nbi_noise=gate_noise,
      )
    except (FileNotFoundError, ValueError, OSError) as error:
      quality["reason"] = str(error)
    quality_rows.append(quality)

  return PreparedData(windows=windows, quality_rows=quality_rows, channel_names=channel_names)


def iter_chunks(items: list[Window], size: int) -> Iterable[list[Window]]:
  for start in range(0, len(items), size):
    yield items[start : start + size]
