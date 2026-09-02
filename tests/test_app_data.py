from __future__ import annotations

import numpy as np

from src.app_data import AppDataset, load_app_cache, save_app_cache
from src.data import PreparedData, Window


def _prepared() -> PreparedData:
  windows = [
    Window(
      shot="1001",
      anchor_ms=200,
      target_context=np.arange(4, dtype=np.float32),
      target_future=np.arange(2, dtype=np.float32),
      covariates=np.arange(8, dtype=np.float32).reshape(2, 4),
      nrmse_scale=2.0,
    ),
    Window(
      shot="1002",
      anchor_ms=300,
      target_context=np.arange(4, dtype=np.float32) + 1,
      target_future=np.arange(2, dtype=np.float32) + 1,
      covariates=np.arange(8, dtype=np.float32).reshape(2, 4) + 1,
      nrmse_scale=3.0,
    ),
  ]
  return PreparedData(windows=windows, quality_rows=[], channel_names=["IP", "SXR"])


def test_app_cache_round_trip(tmp_path) -> None:
  path = tmp_path / "windows.npz"
  save_app_cache(path, _prepared())
  dataset = load_app_cache(path)
  assert isinstance(dataset, AppDataset)
  assert dataset.shot_choices == ["1001", "1002"]
  assert dataset.anchor_choices("1002") == [300]
  assert dataset.index_for("1001", 200) == 0
  assert dataset.channel_names == ("IP", "SXR")


def test_selected_channels_keep_configured_order(tmp_path) -> None:
  path = tmp_path / "windows.npz"
  save_app_cache(path, _prepared())
  dataset = load_app_cache(path)
  window = dataset.window_at(0, ["SXR"])
  assert window.covariates.shape == (1, 4)
  np.testing.assert_allclose(window.covariates[0], np.arange(4, 8, dtype=np.float32))


def test_reconstructed_shot_series_supports_one_ms_selection(tmp_path) -> None:
  windows = [
    Window(
      shot="2001",
      anchor_ms=4,
      target_context=np.arange(4, dtype=np.float32),
      target_future=np.arange(4, 6, dtype=np.float32),
      covariates=np.arange(8, dtype=np.float32).reshape(2, 4),
      nrmse_scale=2.0,
    ),
    Window(
      shot="2001",
      anchor_ms=6,
      target_context=np.arange(2, 6, dtype=np.float32),
      target_future=np.arange(6, 8, dtype=np.float32),
      covariates=np.arange(4, 12, dtype=np.float32).reshape(2, 4),
      nrmse_scale=2.0,
    ),
  ]
  path = tmp_path / "continuous.npz"
  save_app_cache(
    path,
    PreparedData(windows=windows, quality_rows=[], channel_names=["IP", "SXR"]),
  )
  dataset = load_app_cache(path)
  series = dataset.build_shot_series("2001")
  np.testing.assert_array_equal(series.valid_anchors(4, 2), [4, 5, 6])
  selected = series.window(5, dataset.channel_names, ["SXR"], 4, 2)
  np.testing.assert_allclose(selected.target_context, [1, 2, 3, 4])
  np.testing.assert_allclose(selected.target_future, [5, 6])
  assert selected.covariates.shape == (1, 4)
