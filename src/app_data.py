from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data import PreparedData, Window, prepare_windows


@dataclass(frozen=True)
class ShotSeries:
  shot: str
  start_ms: int
  target: np.ndarray
  covariates: np.ndarray
  nrmse_scale: float

  @property
  def end_ms(self) -> int:
    return self.start_ms + self.target.size

  def valid_anchors(self, context_ms: int, horizon_ms: int) -> np.ndarray:
    if self.target.size < context_ms + horizon_ms:
      return np.empty(0, dtype=np.int32)
    target_bad = (~np.isfinite(self.target)).astype(np.int32)
    covariate_bad = (~np.all(np.isfinite(self.covariates), axis=0)).astype(np.int32)
    target_prefix = np.concatenate(([0], np.cumsum(target_bad)))
    covariate_prefix = np.concatenate(([0], np.cumsum(covariate_bad)))
    local_anchors = np.arange(
      context_ms,
      self.target.size - horizon_ms + 1,
      dtype=np.int32,
    )
    target_missing = (
      target_prefix[local_anchors + horizon_ms]
      - target_prefix[local_anchors - context_ms]
    )
    covariate_missing = (
      covariate_prefix[local_anchors] - covariate_prefix[local_anchors - context_ms]
    )
    return local_anchors[(target_missing == 0) & (covariate_missing == 0)] + self.start_ms

  def window(
    self,
    anchor_ms: int,
    channel_names: tuple[str, ...],
    selected_channels: Iterable[str] | None = None,
    context_ms: int = 160,
    horizon_ms: int = 20,
  ) -> Window:
    local_anchor = int(anchor_ms) - self.start_ms
    if selected_channels is None:
      channel_indices = np.arange(len(channel_names), dtype=int)
    else:
      requested = set(selected_channels)
      channel_indices = np.asarray(
        [index for index, name in enumerate(channel_names) if name in requested],
        dtype=int,
      )
    context_start = local_anchor - context_ms
    future_end = local_anchor + horizon_ms
    if context_start < 0 or future_end > self.target.size:
      raise KeyError(f"window is outside reconstructed shot series: {anchor_ms}")
    target_context = self.target[context_start:local_anchor]
    target_future = self.target[local_anchor:future_end]
    covariates = self.covariates[channel_indices, context_start:local_anchor]
    if not (
      np.all(np.isfinite(target_context))
      and np.all(np.isfinite(target_future))
      and np.all(np.isfinite(covariates))
    ):
      raise KeyError(f"window contains missing values: {anchor_ms}")
    return Window(
      shot=self.shot,
      anchor_ms=int(anchor_ms),
      target_context=target_context.astype(np.float32, copy=True),
      target_future=target_future.astype(np.float32, copy=True),
      covariates=covariates.astype(np.float32, copy=True),
      nrmse_scale=self.nrmse_scale,
    )


@dataclass(frozen=True)
class AppDataset:
  shots: np.ndarray
  anchors_ms: np.ndarray
  target_contexts: np.ndarray
  target_futures: np.ndarray
  covariates: np.ndarray
  nrmse_scales: np.ndarray
  channel_names: tuple[str, ...]

  def __post_init__(self) -> None:
    count = self.shots.size
    if not all(
      array.shape[0] == count
      for array in (
        self.anchors_ms,
        self.target_contexts,
        self.target_futures,
        self.covariates,
        self.nrmse_scales,
      )
    ):
      raise ValueError("app cache arrays do not have the same window count")
    if self.covariates.ndim != 3 or self.covariates.shape[1] != len(self.channel_names):
      raise ValueError("app cache channel metadata does not match covariates")

  @property
  def shot_choices(self) -> list[str]:
    return sorted({str(shot) for shot in self.shots}, key=int)

  def anchor_choices(self, shot: str) -> list[int]:
    mask = self.shots == str(shot)
    return sorted(int(value) for value in np.unique(self.anchors_ms[mask]))

  def index_for(self, shot: str, anchor_ms: int) -> int:
    matches = np.flatnonzero(
      (self.shots == str(shot)) & (self.anchors_ms == int(anchor_ms))
    )
    if matches.size != 1:
      raise KeyError(f"window not found: shot={shot}, anchor_ms={anchor_ms}")
    return int(matches[0])

  def window_at(self, index: int, selected_channels: Iterable[str] | None = None) -> Window:
    if selected_channels is None:
      channel_indices = np.arange(len(self.channel_names), dtype=int)
    else:
      requested = set(selected_channels)
      channel_indices = np.asarray(
        [index for index, name in enumerate(self.channel_names) if name in requested],
        dtype=int,
      )
    return Window(
      shot=str(self.shots[index]),
      anchor_ms=int(self.anchors_ms[index]),
      target_context=self.target_contexts[index],
      target_future=self.target_futures[index],
      covariates=self.covariates[index, channel_indices],
      nrmse_scale=float(self.nrmse_scales[index]),
    )

  def build_shot_series(self, shot: str) -> ShotSeries:
    window_indices = np.flatnonzero(self.shots == str(shot))
    if window_indices.size == 0:
      raise KeyError(f"shot not found: {shot}")
    context_ms = self.target_contexts.shape[1]
    horizon_ms = self.target_futures.shape[1]
    anchors = self.anchors_ms[window_indices]
    start_ms = int(np.min(anchors - context_ms))
    end_ms = int(np.max(anchors + horizon_ms))
    length = end_ms - start_ms

    target_sum = np.zeros(length, dtype=np.float64)
    target_count = np.zeros(length, dtype=np.int32)
    covariate_sum = np.zeros((len(self.channel_names), length), dtype=np.float64)
    covariate_count = np.zeros((len(self.channel_names), length), dtype=np.int32)

    for index in window_indices:
      anchor = int(self.anchors_ms[index])
      context_start = anchor - context_ms - start_ms
      context_end = anchor - start_ms
      future_end = context_end + horizon_ms
      target_piece = np.concatenate(
        (self.target_contexts[index], self.target_futures[index])
      )
      target_slice = slice(context_start, future_end)
      target_finite = np.isfinite(target_piece)
      target_sum[target_slice][target_finite] += target_piece[target_finite]
      target_count[target_slice][target_finite] += 1

      covariate_piece = self.covariates[index]
      covariate_finite = np.isfinite(covariate_piece)
      covariate_sum[:, context_start:context_end][covariate_finite] += (
        covariate_piece[covariate_finite]
      )
      covariate_count[:, context_start:context_end][covariate_finite] += 1

    target = np.full(length, np.nan, dtype=np.float32)
    target_available = target_count > 0
    target[target_available] = (
      target_sum[target_available] / target_count[target_available]
    ).astype(np.float32)
    covariates = np.full((len(self.channel_names), length), np.nan, dtype=np.float32)
    covariate_available = covariate_count > 0
    covariates[covariate_available] = (
      covariate_sum[covariate_available] / covariate_count[covariate_available]
    ).astype(np.float32)
    return ShotSeries(
      shot=str(shot),
      start_ms=start_ms,
      target=target,
      covariates=covariates,
      nrmse_scale=float(np.median(self.nrmse_scales[window_indices])),
    )


def save_app_cache(path: Path, prepared: PreparedData) -> None:
  if not prepared.windows:
    raise ValueError("cannot cache an empty prepared dataset")
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = path.with_suffix(".tmp.npz")
  np.savez_compressed(
    temporary_path,
    shots=np.asarray([window.shot for window in prepared.windows], dtype=str),
    anchors_ms=np.asarray([window.anchor_ms for window in prepared.windows], dtype=np.int32),
    target_contexts=np.stack([window.target_context for window in prepared.windows]),
    target_futures=np.stack([window.target_future for window in prepared.windows]),
    covariates=np.stack([window.covariates for window in prepared.windows]),
    nrmse_scales=np.asarray(
      [window.nrmse_scale for window in prepared.windows], dtype=np.float32
    ),
    channel_names=np.asarray(prepared.channel_names, dtype=str),
  )
  temporary_path.replace(path)


def load_app_cache(path: Path) -> AppDataset:
  with np.load(path, allow_pickle=False) as bundle:
    return AppDataset(
      shots=bundle["shots"].astype(str),
      anchors_ms=bundle["anchors_ms"].astype(np.int32),
      target_contexts=bundle["target_contexts"].astype(np.float32),
      target_futures=bundle["target_futures"].astype(np.float32),
      covariates=bundle["covariates"].astype(np.float32),
      nrmse_scales=bundle["nrmse_scales"].astype(np.float32),
      channel_names=tuple(bundle["channel_names"].astype(str).tolist()),
    )


def cache_matches_config(dataset: AppDataset, config: dict[str, Any]) -> bool:
  expected_channels = tuple(config["channels"]["diagnostics"]) + tuple(
    config["channels"]["heating"]
  )
  return (
    dataset.target_contexts.shape[1] == int(config["window"]["context_ms"])
    and dataset.target_futures.shape[1] == int(config["window"]["horizon_ms"])
    and dataset.channel_names == expected_channels
  )


def ensure_app_cache(
  config: dict[str, Any],
  path: Path,
  rebuild: bool = False,
  max_shots: int | None = None,
) -> AppDataset:
  if path.is_file() and not rebuild:
    dataset = load_app_cache(path)
    if cache_matches_config(dataset, config):
      return dataset

  prepared = prepare_windows(config, max_shots=max_shots)
  save_app_cache(path, prepared)
  return load_app_cache(path)
