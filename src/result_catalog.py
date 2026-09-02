from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


FAMILY_LABELS = {
    "timexer": "TimeXer",
    "wpmixer": "WPMixer",
    "timesfm": "TimesFM 2.5",
}

TIMESFM_LABELS = {
    "E2_zero_shot": "Zero-shot",
    "E4_LoRA": "LoRA",
    "E5_diagnostic_residual": "LoRA + diagnostics",
}


@dataclass(frozen=True)
class ResultOption:
    id: str
    family: str
    family_label: str
    variant: str
    variant_label: str
    history_ms: float
    gap_ms: float
    prediction_ms: float
    evaluation_scope: str
    shots: tuple[str, ...]
    input_channels: tuple[str, ...]
    input_summary: str
    window_level_ready: bool
    result_dir: str
    baseline: str = "Persistence"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shots"] = list(self.shots)
        payload["input_channels"] = list(self.input_channels)
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sort_shots(values: Iterable[str]) -> tuple[str, ...]:
    unique = {str(value).strip() for value in values if str(value).strip()}

    def key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return tuple(sorted(unique, key=key))


def _shots_from_csv(path: Path, model: str | None = None) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    shots: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if model and row.get("model") and row["model"] != model:
                    continue
                value = row.get("shot") or row.get("Shot") or row.get("shot_id")
                if value:
                    shots.append(value)
    except OSError:
        return ()
    return _sort_shots(shots)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _run_variant_label(directory: Path, gap_ms: float, prediction_ms: float) -> str:
    name = directory.name.lower()
    prefix = ""
    if "grouped_causal_time" in str(directory).lower():
        prefix = "Grouped causal · "
    if "loss_current_trend" in name:
        prefix += "Current trend loss · "
    elif "loss_dilate" in name:
        prefix += "DILATE loss · "
    elif "loss_label_soft_dtw_inverse" in name:
        prefix += "Soft-DTW + label loss · "
    else:
        prefix += "Standard · "
    return f"{prefix}gap {gap_ms:g} → {prediction_ms:g} ms"


def _timesfm_inputs(config: dict[str, Any], model: str) -> tuple[tuple[str, ...], str]:
    if model != "E5_diagnostic_residual":
        return ("SX2_7_07",), "仅使用 SX2_7_07 的 160 ms 历史"
    channels = tuple(str(value) for value in config.get("diagnostic_channels", []))
    display = ("SX2_7_07",) + channels
    return display, f"SX2_7_07 历史 + {len(channels)} 路固定诊断残差校正"


def _standard_inputs(config: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    values = _nested(config, "experiment", "input_channels", default=[])
    channels = tuple(str(value) for value in values if not str(value).endswith("_available"))
    availability = sum(1 for value in values if str(value).endswith("_available"))
    summary = f"训练时固定使用 {len(channels)} 路物理信号"
    if availability:
        summary += f" + {availability} 路可用性标记"
    return channels, summary


def _scan_directory(directory: Path, results_root: Path) -> list[ResultOption]:
    if any(part.lower().startswith("smoke") for part in directory.parts):
        return []
    run_config_path = directory / "run_config.json"
    quick_config_path = directory / "config.json"
    config = _read_json(run_config_path if run_config_path.is_file() else quick_config_path)
    if not config or bool(config.get("smoke", False)):
        return []

    data_config = _nested(config, "config", "data", default={})
    history_ms = _as_float(data_config.get("history_ms", config.get("history_ms")), -1)
    if history_ms != 160:
        return []
    gap_ms = _as_float(data_config.get("forecast_gap_ms", config.get("forecast_gap_ms")))
    prediction_ms = _as_float(data_config.get("prediction_ms", config.get("prediction_ms")))
    completed = _read_json(directory / "completed.json")
    evaluation_scope = str(completed.get("evaluation_scope") or "逐炮评价")
    relative = directory.relative_to(results_root).as_posix()
    window_ready = (directory / "predictions.npz").is_file()

    models = _nested(config, "experiment", "models_to_run", default=None)
    if not models:
        models = _nested(config, "config", "models_to_run", default=None)
    if not models:
        models = config.get("models", [])

    output: list[ResultOption] = []
    for model_value in models:
        model = str(model_value)
        lowered = model.lower()
        if lowered in {"persistence", "e0_persistence"}:
            continue
        if model in TIMESFM_LABELS:
            family = "timesfm"
            variant_label = TIMESFM_LABELS[model]
            inputs, input_summary = _timesfm_inputs(config, model)
            shots = _shots_from_csv(directory / "per_shot_metrics.csv", model=model)
        elif lowered in {"timexer", "wpmixer"}:
            family = lowered
            variant_label = _run_variant_label(directory, gap_ms, prediction_ms)
            inputs, input_summary = _standard_inputs(config)
            shots = _shots_from_csv(directory / f"{lowered}_metrics_by_shot.csv")
        else:
            continue
        option_id = f"{family}-{_slug(relative)}-{_slug(model)}"
        output.append(
            ResultOption(
                id=option_id,
                family=family,
                family_label=FAMILY_LABELS[family],
                variant=model,
                variant_label=variant_label,
                history_ms=history_ms,
                gap_ms=gap_ms,
                prediction_ms=prediction_ms,
                evaluation_scope=evaluation_scope,
                shots=shots,
                input_channels=inputs,
                input_summary=input_summary,
                window_level_ready=window_ready,
                result_dir=str(directory),
            )
        )
    return output


def scan_result_catalog(results_root: Path) -> list[ResultOption]:
    """Return completed 160 ms result options without modifying result artifacts."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    options: list[ResultOption] = []
    for marker in sorted(root.rglob("completed.json")):
        options.extend(_scan_directory(marker.parent, root))
    family_rank = {"timexer": 0, "wpmixer": 1, "timesfm": 2}
    timesfm_rank = {"E2_zero_shot": 0, "E4_LoRA": 1, "E5_diagnostic_residual": 2}

    def variant_rank(item: ResultOption) -> int:
        if item.family == "timesfm":
            return timesfm_rank.get(item.variant, 99)
        return 0 if item.variant_label.startswith("Standard") else 1

    return sorted(
        options,
        key=lambda item: (
            family_rank.get(item.family, 99),
            variant_rank(item),
            item.gap_ms,
            item.prediction_ms,
            item.variant_label,
        ),
    )
