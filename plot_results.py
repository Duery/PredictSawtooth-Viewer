from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

MPL_CACHE_DIR = Path(__file__).with_name(".matplotlib-cache")
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
  "actual": "#111827",
  "target_only": "#2563eb",
  "all_information": "#dc2626",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Plot a completed PredictSawtooth run.")
  parser.add_argument("run_dir", type=Path)
  return parser.parse_args()


def load_by_shot(path: Path) -> dict[tuple[str, str], dict[str, str]]:
  with path.open(encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
  return {(row["model"], row["shot"]): row for row in rows}


def select_shots(rows: dict[tuple[str, str], dict[str, str]]) -> dict[str, str]:
  shots = sorted({shot for _, shot in rows}, key=int)
  ranked = []
  for shot in shots:
    target_mae = float(rows[("target_only", shot)]["mae"])
    full_mae = float(rows[("all_information", shot)]["mae"])
    improvement = 100.0 * (target_mae - full_mae) / target_mae
    ranked.append((improvement, shot))
  ranked.sort()
  return {
    "degraded": ranked[0][1],
    "typical": ranked[len(ranked) // 2][1],
    "best": ranked[-1][1],
  }


def stitched_series(
  shot: str,
  shots: np.ndarray,
  anchors: np.ndarray,
  actual: np.ndarray,
  target_only: np.ndarray,
  all_information: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  indices = np.flatnonzero(shots == shot)
  buckets: dict[int, list[list[float]]] = {}
  for index in indices:
    for offset in range(actual.shape[1]):
      time_ms = int(anchors[index] + offset)
      values = buckets.setdefault(time_ms, [[], [], []])
      values[0].append(float(actual[index, offset]))
      values[1].append(float(target_only[index, offset]))
      values[2].append(float(all_information[index, offset]))
  times = np.asarray(sorted(buckets), dtype=int)
  truth = np.asarray([np.mean(buckets[t][0]) for t in times])
  target = np.asarray([np.mean(buckets[t][1]) for t in times])
  full = np.asarray([np.mean(buckets[t][2]) for t in times])
  return times, truth, target, full


def plot_shot(
  output_path: Path,
  label: str,
  shot: str,
  improvement: float,
  series: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
  time_ms, actual, target_only, all_information = series
  fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
  ax.plot(time_ms, actual, color=COLORS["actual"], linewidth=1.7, label="Ground truth")
  ax.plot(
    time_ms,
    target_only,
    color=COLORS["target_only"],
    linewidth=1.15,
    alpha=0.85,
    label="TimesFM target only",
  )
  ax.plot(
    time_ms,
    all_information,
    color=COLORS["all_information"],
    linewidth=1.15,
    alpha=0.9,
    label="TimesFM all information",
  )
  ax.set_title(f"Shot {shot} - {label} case - all-information MAE change {improvement:+.1f}%")
  ax.set_xlabel("Time in shot (ms)")
  ax.set_ylabel("Processed SX2_7_07")
  ax.grid(alpha=0.22)
  ax.legend(ncol=3, frameon=False)
  fig.savefig(output_path, dpi=180)
  plt.close(fig)


def plot_window_examples(
  output_path: Path,
  shots: np.ndarray,
  anchors: np.ndarray,
  actual: np.ndarray,
  target_only: np.ndarray,
  all_information: np.ndarray,
) -> None:
  target_mae = np.mean(np.abs(target_only - actual), axis=1)
  full_mae = np.mean(np.abs(all_information - actual), axis=1)
  improvement = 100.0 * (target_mae - full_mae) / np.maximum(target_mae, 1e-12)
  order = np.argsort(improvement)
  positions = [0, len(order) // 3, 2 * len(order) // 3, len(order) - 1]
  labels = ["Most degraded", "Lower-middle", "Upper-middle", "Most improved"]

  fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True)
  horizon = np.arange(actual.shape[1])
  for ax, position, label in zip(axes.flat, positions, labels, strict=True):
    index = order[position]
    ax.plot(horizon, actual[index], color=COLORS["actual"], linewidth=2.0, label="Ground truth")
    ax.plot(
      horizon,
      target_only[index],
      color=COLORS["target_only"],
      linewidth=1.4,
      label="Target only",
    )
    ax.plot(
      horizon,
      all_information[index],
      color=COLORS["all_information"],
      linewidth=1.4,
      label="All information",
    )
    ax.set_title(
      f"{label}: shot {shots[index]}, anchor {anchors[index]} ms\n"
      f"all-information MAE change {improvement[index]:+.1f}%"
    )
    ax.set_ylabel("Processed SX2_7_07")
    ax.grid(alpha=0.22)
  axes[1, 0].set_xlabel("Forecast offset (ms)")
  axes[1, 1].set_xlabel("Forecast offset (ms)")
  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.suptitle("20 ms forecast examples selected by paired MAE change", y=0.99)
  fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=3,
    frameon=False,
  )
  fig.tight_layout(rect=(0, 0, 1, 0.91))
  fig.savefig(output_path, dpi=180)
  plt.close(fig)


def main() -> None:
  args = parse_args()
  run_dir = args.run_dir
  figures_dir = run_dir / "figures"
  figures_dir.mkdir(parents=True, exist_ok=True)
  rows = load_by_shot(run_dir / "by_shot.csv")
  selected = select_shots(rows)

  with np.load(run_dir / "predictions.npz") as bundle:
    actual = bundle["actual"]
    target_only = bundle["target_only"]
    all_information = bundle["all_information"]
    shots = bundle["shot"].astype(str)
    anchors = bundle["anchor_ms"]

  for label, shot in selected.items():
    target_mae = float(rows[("target_only", shot)]["mae"])
    full_mae = float(rows[("all_information", shot)]["mae"])
    improvement = 100.0 * (target_mae - full_mae) / target_mae
    plot_shot(
      figures_dir / f"rolling_{label}_shot_{shot}.png",
      label,
      shot,
      improvement,
      stitched_series(shot, shots, anchors, actual, target_only, all_information),
    )

  plot_window_examples(
    figures_dir / "window_examples.png",
    shots,
    anchors,
    actual,
    target_only,
    all_information,
  )
  for path in sorted(figures_dir.glob("*.png")):
    print(path)


if __name__ == "__main__":
  main()
