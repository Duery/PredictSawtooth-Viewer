from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_RUNNERS = {
    "wpmixer.timexer.v1",
    "wpmixer.legacy.v1",
    "timesfm25.lora.v1",
    "timesfm3.zero_shot.v1",
    "timesfm3.multivariate.v1",
}
PACKAGE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
REQUIRED_ARTIFACTS = ("checkpoint", "run_config", "preprocessing", "completed")
REQUIRED_TEXT_FIELDS = (
    "model_package_id",
    "display_name",
    "family",
    "variant",
    "runner",
    "target_channel",
)
REQUIRED_NUMERIC_FIELDS = (
    "sampling_hz",
    "history_ms",
    "forecast_gap_ms",
    "prediction_ms",
)


@dataclass(frozen=True)
class ModelPackage:
    package_dir: Path
    manifest: dict[str, Any]
    artifact_paths: dict[str, Path]
    errors: tuple[str, ...]
    fingerprint: str = ""

    @property
    def package_id(self) -> str:
        value = self.manifest.get("model_package_id")
        return str(value) if value else self.package_dir.name

    @property
    def ready(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_package_id": self.package_id,
            "display_name": str(
                self.manifest.get("display_name") or self.package_dir.name
            ),
            "family": str(self.manifest.get("family") or "unknown"),
            "variant": str(self.manifest.get("variant") or "unknown"),
            "runner": str(self.manifest.get("runner") or ""),
            "history_ms": self.manifest.get("history_ms"),
            "forecast_gap_ms": self.manifest.get("forecast_gap_ms"),
            "prediction_ms": self.manifest.get("prediction_ms"),
            "sampling_hz": self.manifest.get("sampling_hz"),
            "supported_shots": list(self.manifest.get("supported_shots") or []),
            "ready": self.ready,
            "status": "可实时推理" if self.ready else "；".join(self.errors),
            "fingerprint": self.fingerprint,
        }


def _read_manifest(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, ["缺少 manifest.json"]
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"manifest.json 无法读取：{error}"]
    if not isinstance(payload, dict):
        return {}, ["manifest.json 顶层必须是对象"]
    return payload, []


def _resolve_artifact(package_dir: Path, value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "路径为空"
    relative = Path(value)
    if relative.is_absolute():
        return None, "必须使用模型包内的相对路径"
    resolved = (package_dir / relative).resolve()
    try:
        resolved.relative_to(package_dir.resolve())
    except ValueError:
        return None, "路径越出模型包"
    return resolved, None


def _fingerprint(manifest_path: Path, artifact_paths: Mapping[str, Path]) -> str:
    digest = hashlib.sha256()
    for path in (manifest_path, *(artifact_paths[key] for key in sorted(artifact_paths))):
        if not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_model_package(package_dir: Path) -> ModelPackage:
    package_dir = Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    manifest, errors = _read_manifest(manifest_path)
    artifact_paths: dict[str, Path] = {}
    if not manifest:
        return ModelPackage(package_dir, manifest, artifact_paths, tuple(errors))

    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            f"不支持 schema_version={manifest.get('schema_version')!r}"
        )
    for field in REQUIRED_TEXT_FIELDS:
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            errors.append(f"缺少字段 {field}")
    package_id = str(manifest.get("model_package_id") or "")
    if package_id and not PACKAGE_ID_PATTERN.fullmatch(package_id):
        errors.append("model_package_id 只能包含小写字母、数字、点、下划线和连字符")
    runner = str(manifest.get("runner") or "")
    if runner and runner not in SUPPORTED_RUNNERS:
        errors.append(f"未安装 Runner：{runner}")

    for field in REQUIRED_NUMERIC_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, (int, float)):
            errors.append(f"字段 {field} 必须是数值")
        elif field != "forecast_gap_ms" and float(value) <= 0:
            errors.append(f"字段 {field} 必须大于 0")
        elif field == "forecast_gap_ms" and float(value) < 0:
            errors.append("字段 forecast_gap_ms 不能小于 0")

    input_channels = manifest.get("input_channels")
    physical_channels = manifest.get("physical_input_channels")
    if not isinstance(input_channels, list) or not input_channels:
        errors.append("input_channels 必须是非空列表")
    if not isinstance(physical_channels, list) or not physical_channels:
        errors.append("physical_input_channels 必须是非空列表")
    if isinstance(input_channels, list) and len(input_channels) != len(set(input_channels)):
        errors.append("input_channels 不能包含重复通道")
    if manifest.get("target_channel") not in (physical_channels or []):
        errors.append("target_channel 不在 physical_input_channels 中")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts 必须是对象")
        artifacts = {}
    for name in REQUIRED_ARTIFACTS:
        if name not in artifacts:
            errors.append(f"缺少 artifacts.{name}")
    for name, value in artifacts.items():
        path, path_error = _resolve_artifact(package_dir, value)
        if path_error:
            errors.append(f"artifacts.{name} {path_error}")
            continue
        assert path is not None
        artifact_paths[name] = path
        if not path.is_file():
            errors.append(f"缺少工件 {name}：{path.name}")

    completed_path = artifact_paths.get("completed")
    if completed_path and completed_path.is_file():
        completed, completed_errors = _read_manifest(completed_path)
        if completed_errors or not completed:
            errors.append("完成标记不可读")
        elif not (completed.get("completed_at") or completed.get("completed") is True):
            errors.append("完成标记未声明 completed_at/completed=true")

    fingerprint = _fingerprint(manifest_path, artifact_paths) if not errors else ""
    return ModelPackage(
        package_dir=package_dir,
        manifest=manifest,
        artifact_paths=artifact_paths,
        errors=tuple(errors),
        fingerprint=fingerprint,
    )


def scan_model_registry(root: Path) -> list[ModelPackage]:
    root = Path(root)
    if not root.is_dir():
        return []
    packages = [
        validate_model_package(directory)
        for directory in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if directory.is_dir()
    ]
    duplicate_ids = {
        package.package_id
        for package in packages
        if sum(item.package_id == package.package_id for item in packages) > 1
    }
    if duplicate_ids:
        packages = [
            ModelPackage(
                package.package_dir,
                package.manifest,
                package.artifact_paths,
                package.errors + ("model_package_id 重复",),
                "",
            )
            if package.package_id in duplicate_ids
            else package
            for package in packages
        ]
    return packages


def package_for_result_option(
    option: Mapping[str, Any],
    packages: list[ModelPackage],
    results_root: Path,
) -> ModelPackage | None:
    result_dir = Path(str(option.get("result_dir") or ""))
    try:
        relative_dir = result_dir.resolve().relative_to(Path(results_root).resolve()).as_posix()
    except (OSError, ValueError):
        return None
    for package in packages:
        compatible = package.manifest.get("compatible_result")
        if not isinstance(compatible, dict):
            continue
        if (
            str(compatible.get("relative_dir") or "").replace("\\", "/")
            == relative_dir
            and str(compatible.get("variant") or "")
            == str(option.get("variant") or "")
        ):
            return package
    return None
