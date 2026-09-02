from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .model_registry import ModelPackage


class InferenceError(RuntimeError):
    pass


def _finite(values: np.ndarray) -> list[float]:
    array = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    return [float(value) for value in array]


def _point_metrics(actual: np.ndarray, predicted: np.ndarray, scale: float) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else None
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 1e-12 and np.std(predicted) > 1e-12
        else None
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse": float(rmse / scale) if scale > 0 else None,
        "correlation": correlation,
        "r2": r2,
    }


class TimeXerRunner:
    def __init__(self, package: ModelPackage, code_root: Path | None, device: str):
        self.package = package
        self.device_name = device
        if code_root is not None:
            resolved_code_root = Path(code_root).resolve()
            if str(resolved_code_root) not in sys.path:
                sys.path.insert(0, str(resolved_code_root))
        try:
            import torch
            from sawtooth.models.timexer import CrossChannelTimeXer
        except (ImportError, OSError) as error:
            raise InferenceError(f"TimeXer Runner 依赖不可用：{error}") from error

        self.torch = torch
        manifest = package.manifest
        if int(manifest.get("data_processing_version", -1)) != 1:
            raise InferenceError("当前 TimeXer Runner 只支持 data_processing_version=1")
        scaler_payload = json.loads(
            package.artifact_paths["preprocessing"].read_text(encoding="utf-8")
        )
        self.scaler_channels = tuple(str(value) for value in scaler_payload["channel_names"])
        self.center = np.asarray(scaler_payload["center"], dtype=np.float64)
        self.scale = np.asarray(scaler_payload["scale"], dtype=np.float64)
        if self.center.shape != self.scale.shape or self.center.size != len(self.scaler_channels):
            raise InferenceError("scaler.json 维度不一致")
        if np.any(~np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise InferenceError("scaler.json 包含非法 scale")

        input_channels = tuple(str(value) for value in manifest["input_channels"])
        physical_channels = tuple(str(value) for value in manifest["physical_input_channels"])
        availability_channels = tuple(str(value) for value in manifest.get("availability_channels", []))
        if input_channels != physical_channels + availability_channels:
            raise InferenceError("manifest 输入通道顺序与 legacy TimeXer 预处理不一致")
        target_channel = str(manifest["target_channel"])
        parameters = manifest.get("model_parameters") or {}
        sample_rate = int(manifest["sampling_hz"])
        history_points = int(round(float(manifest["history_ms"]) * sample_rate / 1000.0))
        prediction_points = int(round(float(manifest["prediction_ms"]) * sample_rate / 1000.0))
        model = CrossChannelTimeXer(
            history_points=history_points,
            prediction_points=prediction_points,
            input_channels=len(input_channels),
            target_channel_index=input_channels.index(target_channel),
            model_dimension=int(parameters["model_dimension"]),
            attention_heads=int(parameters["attention_heads"]),
            encoder_layers=int(parameters["encoder_layers"]),
            feedforward_dimension=int(parameters["feedforward_dimension"]),
            patch_length=int(parameters["patch_length"]),
            dropout=float(parameters["dropout"]),
            attention_factor=int(parameters["attention_factor"]),
            use_normalization=bool(parameters["use_normalization"]),
        )
        device_value = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_value)
        try:
            state = torch.load(
                package.artifact_paths["checkpoint"],
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
        except Exception as error:
            raise InferenceError(f"checkpoint 与 TimeXer 结构不兼容：{error}") from error
        self.model = model.to(self.device).eval()

    def predict(
        self,
        signals: np.ndarray,
        availability: np.ndarray,
        cache_channels: tuple[str, ...],
        history_start: int,
        history_end: int,
    ) -> tuple[np.ndarray, float, int]:
        manifest = self.package.manifest
        physical_channels = tuple(str(value) for value in manifest["physical_input_channels"])
        physical_indices = tuple(cache_channels.index(name) for name in physical_channels)
        scaler_indices = tuple(self.scaler_channels.index(name) for name in physical_channels)
        raw = np.asarray(signals[history_start:history_end, physical_indices], dtype=np.float32)
        transformed = (raw - self.center[list(scaler_indices)]) / self.scale[list(scaler_indices)]
        np.clip(transformed, -20.0, 20.0, out=transformed)
        x_availability = np.asarray(availability[history_start:history_end], dtype=np.float32)
        x = np.concatenate((transformed, x_availability), axis=1).astype(np.float32, copy=False)
        if x.shape[1] != len(manifest["input_channels"]):
            raise InferenceError("构造后的模型输入通道数不匹配")
        tensor = self.torch.from_numpy(np.ascontiguousarray(x))[None].to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            prediction_scaled = self.model(tensor)[0, :, 0].detach().cpu().numpy()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        target_index = self.scaler_channels.index(str(manifest["target_channel"]))
        prediction = prediction_scaled * self.scale[target_index] + self.center[target_index]
        return (
            np.asarray(prediction, dtype=np.float64),
            elapsed_ms,
            int(manifest["sampling_hz"]),
        )


class WPMixerLegacyRunner:
    """Exact runner for the pre-grouped WPMixer checkpoints saved in 2026-08."""

    def __init__(self, package: ModelPackage, code_root: Path | None, device: str):
        self.package = package
        if code_root is not None:
            resolved_code_root = Path(code_root).resolve()
            if str(resolved_code_root) not in sys.path:
                sys.path.insert(0, str(resolved_code_root))
        try:
            import torch
            from torch import nn
            from sawtooth.models.backbones.wpmixer_core import WPMixerCore
        except (ImportError, OSError) as error:
            raise InferenceError(f"WPMixer Runner 依赖不可用：{error}") from error

        self.torch = torch
        manifest = package.manifest
        if int(manifest.get("data_processing_version", -1)) != 1:
            raise InferenceError("当前 WPMixer Runner 只支持 data_processing_version=1")
        scaler_payload = json.loads(
            package.artifact_paths["preprocessing"].read_text(encoding="utf-8")
        )
        self.scaler_channels = tuple(str(value) for value in scaler_payload["channel_names"])
        self.center = np.asarray(scaler_payload["center"], dtype=np.float64)
        self.scale = np.asarray(scaler_payload["scale"], dtype=np.float64)
        if self.center.shape != self.scale.shape or self.center.size != len(self.scaler_channels):
            raise InferenceError("scaler.json 维度不一致")
        if np.any(~np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise InferenceError("scaler.json 包含非法 scale")

        input_channels = tuple(str(value) for value in manifest["input_channels"])
        physical_channels = tuple(str(value) for value in manifest["physical_input_channels"])
        availability_channels = tuple(str(value) for value in manifest.get("availability_channels", []))
        if input_channels != physical_channels + availability_channels:
            raise InferenceError("manifest 输入通道顺序与 legacy WPMixer 预处理不一致")
        parameters = manifest.get("model_parameters") or {}
        sample_rate = int(manifest["sampling_hz"])
        history_points = int(round(float(manifest["history_ms"]) * sample_rate / 1000.0))
        prediction_points = int(round(float(manifest["prediction_ms"]) * sample_rate / 1000.0))
        device_value = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_value)

        class LegacyWPMixer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.core = WPMixerCore(
                    input_length=history_points,
                    pred_length=prediction_points,
                    wavelet_name=str(parameters["wavelet_name"]),
                    level=int(parameters["level"]),
                    batch_size=1,
                    channel=len(input_channels),
                    d_model=int(parameters["d_model"]),
                    dropout=float(parameters["dropout"]),
                    embedding_dropout=float(parameters["dropout"]),
                    tfactor=int(parameters["temporal_factor"]),
                    dfactor=int(parameters["embedding_factor"]),
                    device=self_device,
                    patch_len=int(parameters["patch_length"]),
                    patch_stride=int(parameters["patch_stride"]),
                    no_decomposition=False,
                    use_amp=False,
                )
                hidden_channels = len(input_channels) * 2
                self.channel_head = nn.Sequential(
                    nn.Linear(len(input_channels), hidden_channels),
                    nn.GELU(),
                    nn.Dropout(float(parameters["dropout"])),
                    nn.Linear(hidden_channels, 1),
                )

            def forward(self, x):
                return self.channel_head(self.core(x))

        self_device = self.device
        model = LegacyWPMixer()
        try:
            state = torch.load(
                package.artifact_paths["checkpoint"],
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
        except Exception as error:
            raise InferenceError(f"checkpoint 与 legacy WPMixer 结构不兼容：{error}") from error
        self.model = model.to(self.device).eval()

    def predict(
        self,
        signals: np.ndarray,
        availability: np.ndarray,
        cache_channels: tuple[str, ...],
        history_start: int,
        history_end: int,
    ) -> tuple[np.ndarray, float, int]:
        manifest = self.package.manifest
        physical_channels = tuple(str(value) for value in manifest["physical_input_channels"])
        physical_indices = tuple(cache_channels.index(name) for name in physical_channels)
        scaler_indices = tuple(self.scaler_channels.index(name) for name in physical_channels)
        raw = np.asarray(signals[history_start:history_end, physical_indices], dtype=np.float32)
        transformed = (raw - self.center[list(scaler_indices)]) / self.scale[list(scaler_indices)]
        np.clip(transformed, -20.0, 20.0, out=transformed)
        x_availability = np.asarray(availability[history_start:history_end], dtype=np.float32)
        x = np.concatenate((transformed, x_availability), axis=1).astype(np.float32, copy=False)
        if x.shape[1] != len(manifest["input_channels"]):
            raise InferenceError("构造后的模型输入通道数不匹配")
        tensor = self.torch.from_numpy(np.ascontiguousarray(x))[None].to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            prediction_scaled = self.model(tensor)[0, :, 0].detach().cpu().numpy()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        target_index = self.scaler_channels.index(str(manifest["target_channel"]))
        prediction = prediction_scaled * self.scale[target_index] + self.center[target_index]
        return np.asarray(prediction, dtype=np.float64), elapsed_ms, int(manifest["sampling_hz"])


class TimesFM25LoRARunner:
    def __init__(self, package: ModelPackage, code_root: Path | None, device: str):
        del code_root
        self.package = package
        try:
            import torch
            from peft import PeftModel
            from scipy.signal import resample_poly
            from transformers import TimesFm2_5ModelForPrediction
        except (ImportError, OSError) as error:
            raise InferenceError(f"TimesFM 2.5 LoRA Runner 依赖不可用：{error}") from error
        self.torch = torch
        self.resample_poly = resample_poly
        manifest = package.manifest
        source_rate = int(manifest["sampling_hz"])
        model_rate = int(manifest["model_sampling_hz"])
        if source_rate % model_rate != 0:
            raise InferenceError("TimesFM 当前只支持整数倍降采样")
        self.downsample = source_rate // model_rate
        self.output_points = int(round(float(manifest["prediction_ms"]) * model_rate / 1000.0))
        device_value = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_value)
        base_model_id = str(manifest.get("base_model_id") or "")
        if not base_model_id:
            raise InferenceError("TimesFM 模型包缺少 base_model_id")
        adapter_config = package.artifact_paths.get("adapter_config")
        if adapter_config is None:
            raise InferenceError("TimesFM 模型包缺少 LoRA adapter_config")
        try:
            base_model = TimesFm2_5ModelForPrediction.from_pretrained(
                base_model_id,
                local_files_only=True,
            )
            model = PeftModel.from_pretrained(
                base_model,
                adapter_config.parent,
                is_trainable=False,
            )
        except Exception as error:
            raise InferenceError(
                f"TimesFM 2.5 基座或 LoRA 无法从本机缓存加载：{error}"
            ) from error
        self.model = model.to(self.device).eval()

    def predict(
        self,
        signals: np.ndarray,
        availability: np.ndarray,
        cache_channels: tuple[str, ...],
        history_start: int,
        history_end: int,
    ) -> tuple[np.ndarray, float, int]:
        del availability
        manifest = self.package.manifest
        target_index = cache_channels.index(str(manifest["target_channel"]))
        raw_history = np.asarray(
            signals[history_start:history_end, target_index], dtype=np.float32
        )
        history = self.resample_poly(
            raw_history,
            up=1,
            down=self.downsample,
            padtype="line",
        ).astype(np.float32, copy=False)
        histories = self.torch.from_numpy(np.ascontiguousarray(history))[None].to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(
                past_values=[row for row in histories],
                forecast_context_len=int(histories.shape[1]),
                return_dict=True,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        point = getattr(outputs, "mean_predictions", None)
        if point is None:
            point = getattr(outputs, "predictions", None)
        if point is None:
            raise InferenceError("TimesFM 输出中没有 mean_predictions/predictions")
        prediction = point[0, : self.output_points].detach().cpu().numpy()
        return (
            np.asarray(prediction, dtype=np.float64),
            elapsed_ms,
            int(manifest["model_sampling_hz"]),
        )


class TimesFM3ZeroShotRunner:
    """Official TimesFM 3.0 zero-shot inference with the TimesFM 2.5 data contract."""

    def __init__(self, package: ModelPackage, code_root: Path | None, device: str):
        del code_root
        self.package = package
        vendor_root = package.package_dir.parents[1] / "vendor"
        if str(vendor_root.resolve()) not in sys.path:
            sys.path.insert(0, str(vendor_root.resolve()))
        try:
            import torch
            from scipy.signal import resample_poly
            from timesfm3 import ModelConfig, TimesFM3Evaluator
        except (ImportError, OSError) as error:
            raise InferenceError(f"TimesFM 3.0 Runner 依赖不可用：{error}") from error

        self.torch = torch
        self.resample_poly = resample_poly
        manifest = package.manifest
        if int(manifest.get("data_processing_version", -1)) != 2:
            raise InferenceError("TimesFM 3.0 必须使用与 TimesFM 2.5 相同的 v2 数据处理")
        source_rate = int(manifest["sampling_hz"])
        model_rate = int(manifest["model_sampling_hz"])
        if source_rate % model_rate != 0:
            raise InferenceError("TimesFM 3.0 当前只支持整数倍降采样")
        self.downsample = source_rate // model_rate
        self.output_points = int(
            round(float(manifest["prediction_ms"]) * model_rate / 1000.0)
        )
        device_value = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_value)
        model_config_path = package.artifact_paths.get("model_config")
        if model_config_path is None:
            raise InferenceError("TimesFM 3.0 模型包缺少 model_config")
        try:
            config = ModelConfig(
                checkpoint_path=str(model_config_path.parent),
                per_core_batch_size=1,
                device=str(self.device),
                local_files_only=True,
            )
            self.forecaster = TimesFM3Evaluator(config)
        except Exception as error:
            raise InferenceError(f"TimesFM 3.0 官方权重无法加载：{error}") from error

    def predict(
        self,
        signals: np.ndarray,
        availability: np.ndarray,
        cache_channels: tuple[str, ...],
        history_start: int,
        history_end: int,
    ) -> tuple[np.ndarray, float, int]:
        del availability
        manifest = self.package.manifest
        target_index = cache_channels.index(str(manifest["target_channel"]))
        raw_history = np.asarray(
            signals[history_start:history_end, target_index], dtype=np.float32
        )
        history = self.resample_poly(
            raw_history,
            up=1,
            down=self.downsample,
            padtype="line",
        ).astype(np.float32, copy=False)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = next(
                iter(
                    self.forecaster.predict_batch(
                        contexts=[history],
                        horizon=self.output_points,
                        return_quantiles=False,
                        use_symmetric_averaging=False,
                        make_positive=False,
                        use_znorm=False,
                    )
                )
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        prediction = np.asarray(output.forecast, dtype=np.float64).reshape(-1)
        if prediction.size != self.output_points:
            raise InferenceError(
                f"TimesFM 3.0 输出长度 {prediction.size}，预期 {self.output_points}"
            )
        return prediction, elapsed_ms, int(manifest["model_sampling_hz"])


class TimesFM3MultivariateRunner(TimesFM3ZeroShotRunner):
    """TimesFM 3.0 native multivariate inference over all physical channels."""

    def predict(
        self,
        signals: np.ndarray,
        availability: np.ndarray,
        cache_channels: tuple[str, ...],
        history_start: int,
        history_end: int,
    ) -> tuple[np.ndarray, float, int]:
        del availability
        manifest = self.package.manifest
        physical_channels = tuple(str(value) for value in manifest["physical_input_channels"])
        if len(physical_channels) < 2:
            raise InferenceError("TimesFM 3.0 多变量模型至少需要两个物理通道")
        try:
            channel_indices = [cache_channels.index(channel) for channel in physical_channels]
            target_row = physical_channels.index(str(manifest["target_channel"]))
        except ValueError as error:
            raise InferenceError(f"TimesFM 3.0 多变量通道配置无效：{error}") from error

        raw_history = np.asarray(
            signals[history_start:history_end, channel_indices], dtype=np.float32
        )
        history = self.resample_poly(
            raw_history,
            up=1,
            down=self.downsample,
            axis=0,
            padtype="line",
        ).astype(np.float32, copy=False)
        expected_history_points = int(
            round(float(manifest["history_ms"]) * int(manifest["model_sampling_hz"]) / 1000.0)
        )
        if history.shape != (expected_history_points, len(physical_channels)):
            raise InferenceError(
                "TimesFM 3.0 多变量历史形状 "
                f"{history.shape}，预期 {(expected_history_points, len(physical_channels))}"
            )

        started = time.perf_counter()
        with self.torch.inference_mode():
            output = next(
                iter(
                    self.forecaster.predict_batch(
                        contexts=[history.T],
                        horizon=self.output_points,
                        return_quantiles=False,
                        use_symmetric_averaging=False,
                        make_positive=False,
                        use_znorm=False,
                    )
                )
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        forecast = np.asarray(output.forecast, dtype=np.float64)
        expected_shape = (len(physical_channels), self.output_points)
        if forecast.shape != expected_shape:
            raise InferenceError(
                f"TimesFM 3.0 多变量输出形状 {forecast.shape}，预期 {expected_shape}"
            )
        return forecast[target_row], elapsed_ms, int(manifest["model_sampling_hz"])


class ModelInferenceService:
    def __init__(
        self,
        packages: list[ModelPackage],
        cache_dir: Path,
        code_root: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.packages = {package.package_id: package for package in packages}
        self.cache_dir = Path(cache_dir)
        self.code_root = Path(code_root) if code_root else None
        self.device = device
        self._runner_cache: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()
        manifest_path = self.cache_dir / "manifest.json"
        try:
            self.cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InferenceError(f"数据缓存不可用：{manifest_path}（{error}）") from error
        self.cache_channels = tuple(self.cache_manifest.get("physical_channels") or [])
        self.cache_shots = {int(value) for value in self.cache_manifest.get("shots") or []}

    def _package(self, package_id: str) -> ModelPackage:
        package = self.packages.get(package_id)
        if package is None:
            raise InferenceError(f"未知模型包：{package_id}")
        if not package.ready:
            raise InferenceError("模型包不可推理：" + "；".join(package.errors))
        return package

    def _load_shot(self, shot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if shot not in self.cache_shots:
            raise InferenceError(f"数据缓存中没有 Shot {shot}")
        shot_dir = self.cache_dir / str(shot)
        try:
            return (
                np.load(shot_dir / "signals.npy", mmap_mode="r"),
                np.load(shot_dir / "heating_available.npy", mmap_mode="r"),
                np.load(shot_dir / "active_mask.npy", mmap_mode="r"),
            )
        except (OSError, ValueError) as error:
            raise InferenceError(f"Shot {shot} 缓存不可读：{error}") from error

    @staticmethod
    def _point_counts(manifest: dict[str, Any]) -> tuple[int, int, int]:
        rate = int(manifest["sampling_hz"])
        return tuple(
            int(round(float(manifest[field]) * rate / 1000.0))
            for field in ("history_ms", "forecast_gap_ms", "prediction_ms")
        )

    def _valid_history_end_ranges(
        self, package: ModelPackage, active_mask: np.ndarray
    ) -> list[tuple[int, int]]:
        history, gap, prediction = self._point_counts(package.manifest)
        size = len(active_mask)
        if not bool(package.manifest.get("active_only", False)):
            return [(history, size - gap - prediction)]
        active = np.asarray(active_mask, dtype=bool)
        padded = np.concatenate(([False], active, [False]))
        transitions = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        ranges: list[tuple[int, int]] = []
        for start, end in zip(starts, ends):
            first = int(start + history)
            last = int(end - gap - prediction)
            if first <= last:
                ranges.append((first, last))
        return ranges

    @staticmethod
    def _index_to_ms(index: int | float, manifest: dict[str, Any]) -> float:
        return float(manifest.get("start_ms", 0.0)) + float(index) * 1000.0 / int(
            manifest["sampling_hz"]
        )

    def navigation(self, package_id: str, shot: int, max_points: int = 1400) -> dict[str, Any]:
        package = self._package(package_id)
        supported = {int(value) for value in package.manifest.get("supported_shots") or []}
        if supported and shot not in supported:
            raise InferenceError(f"模型包未声明支持 Shot {shot}")
        signals, _, active_mask = self._load_shot(shot)
        required_channels = set(package.manifest["physical_input_channels"])
        missing = sorted(required_channels - set(self.cache_channels))
        if missing:
            raise InferenceError(f"数据缓存缺少通道：{', '.join(missing)}")
        ranges = self._valid_history_end_ranges(package, active_mask)
        if not ranges:
            raise InferenceError(f"Shot {shot} 没有满足当前窗口约束的有效 NBI 区间")
        count = min(max_points, signals.shape[0])
        indices = np.unique(np.linspace(0, signals.shape[0] - 1, count, dtype=np.int64))
        ip_index = self.cache_channels.index("IP_VV_2M")
        sxr_index = self.cache_channels.index(str(package.manifest["target_channel"]))
        longest = max(ranges, key=lambda value: value[1] - value[0])
        default_end = int(round((longest[0] + longest[1]) / 2))
        return {
            "model_package_id": package.package_id,
            "shot": str(shot),
            "sampling_hz": int(package.manifest["sampling_hz"]),
            "prediction_sampling_hz": int(
                package.manifest.get(
                    "model_sampling_hz", package.manifest["sampling_hz"]
                )
            ),
            "history_ms": float(package.manifest["history_ms"]),
            "forecast_gap_ms": float(package.manifest["forecast_gap_ms"]),
            "prediction_ms": float(package.manifest["prediction_ms"]),
            "time_ms": _finite(
                float(package.manifest.get("start_ms", 0.0))
                + indices * 1000.0 / int(package.manifest["sampling_hz"])
            ),
            "ip": _finite(signals[indices, ip_index]),
            "sxr": _finite(signals[indices, sxr_index]),
            "active": [bool(value) for value in np.asarray(active_mask[indices], dtype=bool)],
            "valid_history_end_ranges_ms": [
                [self._index_to_ms(start, package.manifest), self._index_to_ms(end, package.manifest)]
                for start, end in ranges
            ],
            "default_history_end_ms": self._index_to_ms(default_end, package.manifest),
        }

    def _runner(self, package: ModelPackage) -> Any:
        key = (package.package_id, package.fingerprint)
        with self._lock:
            runner = self._runner_cache.get(key)
            if runner is None:
                runner_name = str(package.manifest.get("runner") or "")
                runner_types = {
                    "wpmixer.timexer.v1": TimeXerRunner,
                    "wpmixer.legacy.v1": WPMixerLegacyRunner,
                    "timesfm25.lora.v1": TimesFM25LoRARunner,
                    "timesfm3.zero_shot.v1": TimesFM3ZeroShotRunner,
                    "timesfm3.multivariate.v1": TimesFM3MultivariateRunner,
                }
                runner_type = runner_types.get(runner_name)
                if runner_type is None:
                    raise InferenceError(f"未实现 Runner：{runner_name}")
                if runner_name.startswith("timesfm3."):
                    self._runner_cache = {
                        cache_key: value
                        for cache_key, value in self._runner_cache.items()
                        if not str(value.package.manifest.get("runner", "")).startswith(
                            "timesfm3."
                        )
                    }
                else:
                    self._runner_cache = {
                        cache_key: value
                        for cache_key, value in self._runner_cache.items()
                        if cache_key[0] != package.package_id
                    }
                runner = runner_type(package, self.code_root, self.device)
                self._runner_cache[key] = runner
            return runner

    def predict(self, package_id: str, shot: int, history_end_ms: float) -> dict[str, Any]:
        package = self._package(package_id)
        manifest = package.manifest
        supported = {int(value) for value in manifest.get("supported_shots") or []}
        if supported and shot not in supported:
            raise InferenceError(f"模型包未声明支持 Shot {shot}")
        signals, availability, active_mask = self._load_shot(shot)
        rate = int(manifest["sampling_hz"])
        start_ms = float(manifest.get("start_ms", 0.0))
        history_end = int(round((float(history_end_ms) - start_ms) * rate / 1000.0))
        valid_ranges = self._valid_history_end_ranges(package, active_mask)
        if not any(first <= history_end <= last for first, last in valid_ranges):
            raise InferenceError("所选历史终点不在完整有效 NBI 窗口内")
        history_points, gap_points, prediction_points = self._point_counts(manifest)
        history_start = history_end - history_points
        prediction_start = history_end + gap_points
        prediction_end = prediction_start + prediction_points
        runner = self._runner(package)
        prediction, inference_ms, output_rate = runner.predict(
            signals, availability, self.cache_channels, history_start, history_end
        )
        target_index = self.cache_channels.index(str(manifest["target_channel"]))
        ip_index = self.cache_channels.index("IP_VV_2M")
        ground_truth_source = np.asarray(
            signals[prediction_start:prediction_end, target_index], dtype=np.float64
        )
        if output_rate == rate:
            ground_truth = ground_truth_source
        else:
            if rate % output_rate != 0:
                raise InferenceError("Runner 输出采样率与数据采样率不兼容")
            try:
                from scipy.signal import resample_poly
            except (ImportError, OSError) as error:
                raise InferenceError(f"预测真值降采样依赖不可用：{error}") from error
            ground_truth = np.asarray(
                resample_poly(
                    ground_truth_source,
                    up=1,
                    down=rate // output_rate,
                    padtype="line",
                ),
                dtype=np.float64,
            )
        if prediction.shape != ground_truth.shape:
            raise InferenceError(
                f"模型输出长度 {prediction.size} 与真值长度 {ground_truth.size} 不一致"
            )
        persistence = np.full_like(ground_truth, float(signals[history_end - 1, target_index]))
        scaler_payload = json.loads(
            package.artifact_paths["preprocessing"].read_text(encoding="utf-8")
        )
        scaler_channels = tuple(scaler_payload["channel_names"])
        scale = float(scaler_payload["scale"][scaler_channels.index(manifest["target_channel"])])
        history_indices = np.arange(history_start, history_end, dtype=np.int64)
        prediction_time_start_ms = start_ms + prediction_start * 1000.0 / rate
        prediction_time_ms = (
            prediction_time_start_ms
            + np.arange(prediction.size, dtype=np.float64) * 1000.0 / output_rate
        )
        metrics = _point_metrics(ground_truth, prediction, scale)
        metrics["persistence_nrmse"] = _point_metrics(
            ground_truth, persistence, scale
        )["nrmse"]
        return {
            "source": "live_inference",
            "model_package_id": package.package_id,
            "model_display_name": str(manifest["display_name"]),
            "shot": str(shot),
            "device": str(runner.device),
            "history_range_ms": [
                self._index_to_ms(history_start, manifest),
                self._index_to_ms(history_end, manifest),
            ],
            "prediction_range_ms": [
                self._index_to_ms(prediction_start, manifest),
                self._index_to_ms(prediction_end, manifest),
            ],
            "history": {
                "time_ms": _finite(start_ms + history_indices * 1000.0 / rate),
                "sxr": _finite(signals[history_indices, target_index]),
                "ip": _finite(signals[history_indices, ip_index]),
            },
            "forecast": {
                "time_ms": _finite(prediction_time_ms),
                "prediction": _finite(prediction),
                "ground_truth": _finite(ground_truth),
                "persistence": _finite(persistence),
            },
            "metrics": metrics,
            "inference_ms": float(inference_ms),
            "provenance": {
                "runner": str(manifest["runner"]),
                "output_sampling_hz": int(output_rate),
                "package_fingerprint": package.fingerprint,
                "config_fingerprint": (manifest.get("training_provenance") or {}).get(
                    "config_fingerprint"
                ),
                "retrospective": True,
            },
        }
