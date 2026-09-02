from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import package_catalog_option
from src.model_registry import ModelPackage
from src.result_catalog import scan_result_catalog


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standard_config(history_ms: float = 160) -> dict:
    return {
        "experiment": {
            "models_to_run": ["timexer", "wpmixer", "persistence"],
            "input_channels": ["SX2_7_07", "IP_VV_2M", "Pow_NBI_available"],
        },
        "config": {
            "data": {
                "history_ms": history_ms,
                "forecast_gap_ms": 5,
                "prediction_ms": 10,
            }
        },
    }


class ResultCatalogTests(unittest.TestCase):
    def test_ready_live_model_package_can_supply_catalog_entry(self) -> None:
        manifest = {
            "model_package_id": "timesfm3_test",
            "family": "timesfm3",
            "variant": "zero_shot",
            "target_channel": "SX2_7_07",
            "physical_input_channels": ["SX2_7_07"],
            "sampling_hz": 10_000,
            "model_sampling_hz": 1_000,
            "history_ms": 160,
            "forecast_gap_ms": 0,
            "prediction_ms": 20,
            "supported_shots": [7, 9],
            "catalog_entry": {
                "family": "timesfm3",
                "family_label": "TimesFM 3.0",
                "variant_label": "Zero-shot",
                "input_summary": "raw target only",
            },
        }
        package = ModelPackage(Path("package"), manifest, {}, ())
        option = package_catalog_option(package)
        self.assertIsNotNone(option)
        self.assertEqual(option["shots"], ["7", "9"])
        self.assertEqual(option["prediction_sampling_hz"], 1_000)
        self.assertTrue(option["inference_ready"])

    def test_scanner_builds_model_options_and_shot_lists(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "comparison" / "active_nbi_history_160ms_gap_005ms_prediction_010ms"
            write_json(run / "run_config.json", standard_config())
            write_json(run / "completed.json", {"completed": True})
            write_metrics(run / "timexer_metrics_by_shot.csv", [{"shot": "20"}, {"shot": "10"}])
            write_metrics(run / "wpmixer_metrics_by_shot.csv", [{"shot": "30"}])

            options = scan_result_catalog(root)

            self.assertEqual([option.family for option in options], ["timexer", "wpmixer"])
            self.assertEqual(options[0].shots, ("10", "20"))
            self.assertEqual(options[1].shots, ("30",))
            self.assertEqual(options[0].input_channels, ("SX2_7_07", "IP_VV_2M"))
            self.assertIn("1 路可用性标记", options[0].input_summary)
            self.assertFalse(options[0].window_level_ready)

    def test_timesfm_variants_are_window_ready_and_keep_fixed_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "timesfm_quick_h160_gap0_pred20_1khz"
            write_json(
                run / "config.json",
                {
                    "history_ms": 160,
                    "forecast_gap_ms": 0,
                    "prediction_ms": 20,
                    "diagnostic_channels": ["SX2_7_02", "IP_VV_2M"],
                    "models": ["E0_persistence", "E2_zero_shot", "E4_LoRA", "E5_diagnostic_residual"],
                },
            )
            write_json(run / "completed.json", {"evaluation_scope": "validation_only"})
            write_metrics(
                run / "per_shot_metrics.csv",
                [
                    {"model": "E2_zero_shot", "shot": "7"},
                    {"model": "E4_LoRA", "shot": "7"},
                    {"model": "E5_diagnostic_residual", "shot": "7"},
                ],
            )
            (run / "predictions.npz").write_bytes(b"placeholder")

            options = scan_result_catalog(root)

            self.assertEqual(len(options), 3)
            self.assertTrue(all(option.family == "timesfm" for option in options))
            self.assertTrue(all(option.window_level_ready for option in options))
            residual = next(option for option in options if option.variant == "E5_diagnostic_residual")
            self.assertEqual(
                residual.input_channels,
                ("SX2_7_07", "SX2_7_02", "IP_VV_2M"),
            )
            self.assertEqual(residual.evaluation_scope, "validation_only")

    def test_scanner_excludes_smoke_incomplete_and_non_160_results(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = root / "experiment" / "smoke-1"
            write_json(smoke / "run_config.json", standard_config())
            write_json(smoke / "completed.json", {"completed": True})

            wrong_history = root / "experiment" / "history_80"
            write_json(wrong_history / "run_config.json", standard_config(history_ms=80))
            write_json(wrong_history / "completed.json", {"completed": True})

            incomplete = root / "experiment" / "history_160_incomplete"
            write_json(incomplete / "run_config.json", standard_config())

            self.assertEqual(scan_result_catalog(root), [])

    def test_frontend_describes_ip_sxr_as_navigation_only(self) -> None:
        html = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("IP + SXR", html)
        self.assertIn("不改变模型输入", html)
        self.assertNotIn("TimesFM 3.0", html)

    def test_frontend_has_persistent_dark_mode_toggle(self) -> None:
        frontend = Path(__file__).parents[1] / "frontend"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        javascript = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="theme-toggle"', html)
        self.assertIn(".dark {", css)
        self.assertIn("predict-sawtooth-theme", html)
        self.assertIn("predict-sawtooth-theme", javascript)

    def test_frontend_exposes_explicit_live_prediction_action(self) -> None:
        frontend = Path(__file__).parents[1] / "frontend"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        javascript = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="run-prediction"', html)
        self.assertIn('id="prediction-panel"', html)
        self.assertIn('fetch("/api/predict"', javascript)
        self.assertIn("model_package_id", javascript)

    def test_result_page_can_switch_shot_and_shows_output_sampling_rate(self) -> None:
        frontend = Path(__file__).parents[1] / "frontend"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        javascript = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="window-shot-select"', html)
        self.assertIn('id="window-output-rate"', html)
        self.assertIn('id="metric-sampling-rate"', html)
        self.assertIn("prediction_sampling_hz", javascript)
        self.assertIn("output_sampling_hz", javascript)
        self.assertIn("TimesFM 3.0", javascript)

    def test_timesfm3_official_source_is_vendored_with_license(self) -> None:
        project = Path(__file__).parents[1]
        self.assertTrue((project / "vendor" / "timesfm3" / "__init__.py").is_file())
        self.assertTrue((project / "vendor" / "TIMESFM_SOURCE_LICENSE.txt").is_file())

    def test_frontend_bundles_inter_and_smiley_sans_webfonts(self) -> None:
        frontend = Path(__file__).parents[1] / "frontend"
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        smiley_font = frontend / "assets" / "SmileySans-Oblique.woff2"
        smiley_license = frontend / "assets" / "SmileySans-LICENSE.txt"
        inter_font = frontend / "assets" / "Inter-Variable.woff2"
        inter_license = frontend / "assets" / "Inter-LICENSE.txt"
        self.assertIn('@font-face', css)
        self.assertIn('font-family: "Inter"', css)
        self.assertIn('Inter-Variable.woff2', css)
        self.assertIn('font-family: "Smiley Sans"', css)
        self.assertIn('SmileySans-Oblique.woff2', css)
        self.assertTrue(inter_font.is_file())
        self.assertGreater(inter_font.stat().st_size, 300_000)
        self.assertTrue(inter_license.is_file())
        self.assertTrue(smiley_font.is_file())
        self.assertGreater(smiley_font.stat().st_size, 1_000_000)
        self.assertTrue(smiley_license.is_file())


if __name__ == "__main__":
    unittest.main()
