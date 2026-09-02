from __future__ import annotations

import argparse
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

from src.data import prepare_windows
from src.forecast import run_evaluation
from src.metrics import write_csv


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Evaluate TimesFM 3.0 on NBI1-on windows.")
  parser.add_argument("--config", default="config.toml")
  parser.add_argument("--prepare-only", action="store_true")
  parser.add_argument("--max-shots", type=int)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config_path = Path(args.config)
  with config_path.open("rb") as handle:
    config = tomllib.load(handle)

  run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_dir = Path(config["paths"]["output_dir"]) / run_id
  output_dir.mkdir(parents=True, exist_ok=False)
  shutil.copy2(config_path, output_dir / "config.toml")

  print("Preparing cleaned 1 kHz windows ...")
  prepared = prepare_windows(config, max_shots=args.max_shots)
  write_csv(output_dir / "data_quality.csv", prepared.quality_rows)
  included = sum(row["status"] == "included" for row in prepared.quality_rows)
  print(
    f"Prepared {len(prepared.windows)} windows from {included} shots; "
    f"covariates={prepared.channel_names}"
  )
  if not prepared.windows:
    raise SystemExit(f"No usable windows. Inspect {output_dir / 'data_quality.csv'}")
  if args.prepare_only:
    print(f"Preparation complete: {output_dir}")
    return

  summary = run_evaluation(config, prepared, output_dir)
  print("\nMacro-average results:")
  for row in summary:
    print(row)
  print(f"\nArtifacts: {output_dir}")


if __name__ == "__main__":
  main()

