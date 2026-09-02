from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.model_registry import (
    package_for_result_option,
    scan_model_registry,
    validate_model_package,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_manifest() -> dict:
    return {
        "schema_version": 1,
        "model_package_id": "timexer_test",
        "display_name": "TimeXer test",
        "family": "timexer",
        "variant": "timexer",
        "runner": "wpmixer.timexer.v1",
        "artifacts": {
            "checkpoint": "model.pt",
            "run_config": "run_config.json",
            "preprocessing": "scaler.json",
            "completed": "completed.json",
        },
        "target_channel": "SX2_7_07",
        "physical_input_channels": ["IP_VV_2M", "SX2_7_07"],
        "availability_channels": [],
        "input_channels": ["IP_VV_2M", "SX2_7_07"],
        "sampling_hz": 10_000,
        "history_ms": 160,
        "forecast_gap_ms": 0,
        "prediction_ms": 10,
        "supported_shots": [1],
        "compatible_result": {
            "relative_dir": "runs/h160",
            "variant": "timexer",
        },
    }


def create_package(root: Path, manifest: dict | None = None) -> Path:
    package = root / "timexer_test"
    package.mkdir(parents=True)
    write_json(package / "manifest.json", manifest or valid_manifest())
    (package / "model.pt").write_bytes(b"weights")
    write_json(package / "run_config.json", {"config": {}})
    write_json(package / "scaler.json", {"channel_names": []})
    write_json(package / "completed.json", {"completed_at": "2026-09-02T00:00:00Z"})
    return package


class ModelRegistryTests(unittest.TestCase):
    def test_valid_package_is_ready_and_fingerprinted(self) -> None:
        with TemporaryDirectory() as directory:
            package_dir = create_package(Path(directory))
            package = validate_model_package(package_dir)
            self.assertTrue(package.ready, package.errors)
            self.assertEqual(package.package_id, "timexer_test")
            self.assertEqual(len(package.fingerprint), 64)

    def test_artifact_path_cannot_escape_package(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = valid_manifest()
            manifest["artifacts"]["checkpoint"] = "../outside.pt"
            package = validate_model_package(create_package(Path(directory), manifest))
            self.assertFalse(package.ready)
            self.assertTrue(any("越出模型包" in error for error in package.errors))

    def test_unsupported_runner_remains_visible_but_unready(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = valid_manifest()
            manifest["runner"] = "new.family.v1"
            packages = scan_model_registry(Path(directory))
            self.assertEqual(packages, [])
            create_package(Path(directory), manifest)
            packages = scan_model_registry(Path(directory))
            self.assertEqual(len(packages), 1)
            self.assertFalse(packages[0].ready)
            self.assertIn("未安装 Runner", packages[0].errors[0])

    def test_package_matches_exact_result_and_variant(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            package = validate_model_package(create_package(registry))
            result = root / "results" / "runs" / "h160"
            result.mkdir(parents=True)
            option = {"result_dir": str(result), "variant": "timexer"}
            self.assertIs(
                package_for_result_option(option, [package], root / "results"),
                package,
            )
            option["variant"] = "wpmixer"
            self.assertIsNone(
                package_for_result_option(option, [package], root / "results")
            )

    def test_optional_artifact_is_validated_and_fingerprinted(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = valid_manifest()
            manifest["artifacts"]["adapter_config"] = "adapter_config.json"
            package_dir = create_package(Path(directory), manifest)
            write_json(package_dir / "adapter_config.json", {"r": 8})
            package = validate_model_package(package_dir)
            self.assertTrue(package.ready, package.errors)
            self.assertIn("adapter_config", package.artifact_paths)


if __name__ == "__main__":
    unittest.main()
