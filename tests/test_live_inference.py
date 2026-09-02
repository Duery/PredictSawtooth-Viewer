from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

from src.inference import ModelInferenceService
from src.model_registry import scan_model_registry


PROJECT_DIR = Path(__file__).parents[1]


def _local_path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


config_path = PROJECT_DIR / "config.toml"
if config_path.is_file():
    with config_path.open("rb") as stream:
        local_config = tomllib.load(stream)
else:
    local_config = {}
inference_config = local_config.get("inference", {})
WPMIXER_ROOT = _local_path(inference_config.get("code_root"), PROJECT_DIR)
CACHE_DIR = _local_path(inference_config.get("data_cache"), PROJECT_DIR / "Data" / "cache")
MODEL_DIR = PROJECT_DIR / "model_registry"


@unittest.skipUnless(
    (CACHE_DIR / "manifest.json").is_file()
    and (MODEL_DIR / "timexer_loss_current_trend_h160_g0_p10" / "timexer_best_model.pt").is_file()
    and (MODEL_DIR / "wpmixer_h160_g0_p10" / "wpmixer_best_model.pt").is_file(),
    "真实模型包或 HL-3 缓存不在当前计算机",
)
class LiveInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packages = scan_model_registry(MODEL_DIR)
        cls.service = ModelInferenceService(
            cls.packages,
            cache_dir=CACHE_DIR,
            code_root=WPMIXER_ROOT,
            device="cpu",
        )

    def test_navigation_uses_real_shot_and_valid_active_ranges(self) -> None:
        navigation = self.service.navigation(
            "timexer_loss_current_trend_h160_g0_p10", 8933, max_points=500
        )
        self.assertEqual(navigation["shot"], "8933")
        self.assertEqual(navigation["sampling_hz"], 10_000)
        self.assertEqual(navigation["prediction_sampling_hz"], 10_000)
        self.assertEqual(len(navigation["time_ms"]), 500)
        self.assertEqual(len(navigation["ip"]), 500)
        self.assertEqual(len(navigation["sxr"]), 500)
        self.assertTrue(navigation["valid_history_end_ranges_ms"])

    def test_all_latest_timexer_losses_run_real_windows(self) -> None:
        expected = {
            "timexer_loss_current_trend_h160_g0_p10": (0.42919029, 0.43315924),
            "timexer_loss_dilate_h160_g0_p10": (0.44358017, 0.44691988),
            "timexer_loss_label_soft_dtw_inverse_h160_g0_p10": (0.44119597, 0.43771101),
        }
        for package_id, (first, mean) in expected.items():
            with self.subTest(package_id=package_id):
                navigation = self.service.navigation(package_id, 8933)
                result = self.service.predict(
                    package_id, 8933, navigation["default_history_end_ms"]
                )
                self._assert_result(result, 100, 10_000)
                self.assertAlmostEqual(result["forecast"]["prediction"][0], first, delta=1e-4)
                self.assertAlmostEqual(
                    float(np.mean(result["forecast"]["prediction"])), mean, delta=1e-4
                )

    def test_legacy_wpmixer_runs_real_window(self) -> None:
        package_id = "wpmixer_h160_g0_p10"
        navigation = self.service.navigation(package_id, 8933)
        result = self.service.predict(
            package_id, 8933, navigation["default_history_end_ms"]
        )
        self._assert_result(result, 100, 10_000)
        self.assertAlmostEqual(result["forecast"]["prediction"][0], 0.41568480, delta=1e-4)
        self.assertAlmostEqual(
            float(np.mean(result["forecast"]["prediction"])), 0.41366721, delta=1e-4
        )

    @unittest.skipUnless(
        any(
            (Path.home() / ".cache" / "huggingface" / "hub" /
             "models--google--timesfm-2.5-200m-transformers" / "snapshots").glob("*/model.safetensors")
        ),
        "TimesFM 2.5 基座不在本机 Hugging Face 缓存",
    )
    def test_timesfm_lora_runs_real_downsampled_window(self) -> None:
        package_id = "timesfm25_lora_h160_g0_p20"
        navigation = self.service.navigation(package_id, 8936)
        result = self.service.predict(
            package_id, 8936, navigation["default_history_end_ms"]
        )
        self._assert_result(result, 20, 1_000)
        self.assertAlmostEqual(result["forecast"]["prediction"][0], 0.75821781, delta=1e-4)
        self.assertAlmostEqual(
            float(np.mean(result["forecast"]["prediction"])), 0.78579206, delta=1e-4
        )

    @unittest.skipUnless(
        (MODEL_DIR / "timesfm3_zero_shot_h160_g0_p20" / "base_model" / "model.safetensors").is_file(),
        "TimesFM 3.0 官方权重不在当前模型包",
    )
    def test_timesfm3_zero_shot_uses_timesfm25_data_contract(self) -> None:
        package_id = "timesfm3_zero_shot_h160_g0_p20"
        navigation = self.service.navigation(package_id, 8936)
        self.assertEqual(navigation["sampling_hz"], 10_000)
        self.assertEqual(navigation["prediction_sampling_hz"], 1_000)
        result = self.service.predict(
            package_id, 8936, navigation["default_history_end_ms"]
        )
        self._assert_result(result, 20, 1_000)
        self.assertAlmostEqual(result["forecast"]["prediction"][0], 0.75154257, delta=1e-4)

    @unittest.skipUnless(
        (MODEL_DIR / "timesfm3_multivariate_h160_g0_p20" / "base_model" / "model.safetensors").is_file(),
        "TimesFM 3.0 多变量官方权重不在当前模型包",
    )
    def test_timesfm3_multivariate_forecasts_all_physical_channels(self) -> None:
        package_id = "timesfm3_multivariate_h160_g0_p20"
        navigation = self.service.navigation(package_id, 8936)
        self.assertEqual(navigation["sampling_hz"], 10_000)
        self.assertEqual(navigation["prediction_sampling_hz"], 1_000)
        result = self.service.predict(
            package_id, 8936, navigation["default_history_end_ms"]
        )
        self._assert_result(result, 20, 1_000)
        self.assertAlmostEqual(result["forecast"]["prediction"][0], 0.75417906, delta=1e-4)
        self.assertAlmostEqual(
            float(np.mean(result["forecast"]["prediction"])), 0.79944837, delta=1e-4
        )

    def _assert_result(self, result: dict, prediction_points: int, output_rate: int) -> None:
        self.assertEqual(result["source"], "live_inference")
        self.assertEqual(len(result["history"]["sxr"]), 1600)
        self.assertEqual(len(result["forecast"]["prediction"]), prediction_points)
        self.assertEqual(len(result["forecast"]["ground_truth"]), prediction_points)
        self.assertTrue(np.isfinite(result["forecast"]["prediction"]).all())
        self.assertGreater(result["inference_ms"], 0)
        self.assertNotEqual(
            result["forecast"]["prediction"], result["forecast"]["persistence"]
        )
        self.assertEqual(result["history_range_ms"][1], result["prediction_range_ms"][0])
        self.assertEqual(result["provenance"]["output_sampling_hz"], output_rate)


if __name__ == "__main__":
    unittest.main()
