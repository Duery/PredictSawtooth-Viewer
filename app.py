from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

from src.result_catalog import scan_result_catalog
from src.inference import InferenceError, ModelInferenceService
from src.model_registry import package_for_result_option, scan_model_registry


PROJECT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_DIR / "frontend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HL-3 sawtooth result viewer frontend.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--results-root")
    parser.add_argument("--model-registry")
    parser.add_argument("--data-cache")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        return tomllib.load(stream)


def resolve_results_root(config: dict[str, Any], override: str | None) -> Path:
    value = override or config.get("results", {}).get("root", "")
    if not value:
        return PROJECT_DIR / "outputs"
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def resolve_project_path(value: str | None, default: str) -> Path:
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def package_catalog_option(package) -> dict[str, Any] | None:
    entry = package.manifest.get("catalog_entry")
    if not package.ready or not isinstance(entry, dict):
        return None
    manifest = package.manifest
    required = ("family", "family_label", "variant_label", "input_summary")
    if any(not str(entry.get(field) or "").strip() for field in required):
        return None
    return {
        "id": f"model-package-{package.package_id}",
        "family": str(entry["family"]),
        "family_label": str(entry["family_label"]),
        "variant": str(manifest["variant"]),
        "variant_label": str(entry["variant_label"]),
        "history_ms": float(manifest["history_ms"]),
        "gap_ms": float(manifest["forecast_gap_ms"]),
        "prediction_ms": float(manifest["prediction_ms"]),
        "prediction_sampling_hz": int(
            manifest.get("model_sampling_hz", manifest["sampling_hz"])
        ),
        "evaluation_scope": str(entry.get("evaluation_scope") or "live_inference"),
        "shots": [str(value) for value in manifest.get("supported_shots") or []],
        "input_channels": list(manifest.get("physical_input_channels") or []),
        "input_summary": str(entry["input_summary"]),
        "window_level_ready": True,
        "data_status": "实时推理模型",
        "result_dir": str(package.package_dir),
        "model_package_id": package.package_id,
        "inference_ready": True,
        "inference_status": "可实时推理",
    }


def build_handler(
    catalog_payload: dict[str, Any],
    model_payload: dict[str, Any],
    inference_service: ModelInferenceService | None,
    inference_startup_error: str | None,
):
    class ResultViewerHandler(BaseHTTPRequestHandler):
        server_version = "PredictSawtooth/0.2"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            route = urlparse(self.path).path
            if route == "/api/catalog":
                self._send_json(catalog_payload)
                return
            if route == "/api/models":
                self._send_json(model_payload)
                return
            if route == "/api/navigation":
                if inference_service is None:
                    self._send_error_json(503, inference_startup_error or "推理服务不可用")
                    return
                query = parse_qs(urlparse(self.path).query)
                package_id = str((query.get("model_package_id") or [""])[0])
                shot_text = str((query.get("shot") or [""])[0])
                try:
                    payload = inference_service.navigation(package_id, int(shot_text))
                except (ValueError, InferenceError) as error:
                    self._send_error_json(400, str(error))
                    return
                self._send_json(payload)
                return
            static_routes = {
                "/": FRONTEND_DIR / "index.html",
                "/index.html": FRONTEND_DIR / "index.html",
                "/assets/styles.css": FRONTEND_DIR / "styles.css",
                "/assets/app.js": FRONTEND_DIR / "app.js",
                "/assets/Inter-Variable.woff2": FRONTEND_DIR / "assets" / "Inter-Variable.woff2",
                "/assets/Inter-LICENSE.txt": FRONTEND_DIR / "assets" / "Inter-LICENSE.txt",
                "/assets/SmileySans-Oblique.woff2": FRONTEND_DIR / "assets" / "SmileySans-Oblique.woff2",
                "/assets/SmileySans-LICENSE.txt": FRONTEND_DIR / "assets" / "SmileySans-LICENSE.txt",
            }
            target = static_routes.get(route)
            if target is None or not target.is_file():
                self.send_error(404, "Not found")
                return
            body = target.read_bytes()
            if target.suffix.lower() == ".woff2":
                content_type = "font/woff2"
            else:
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            route = urlparse(self.path).path
            if route != "/api/predict":
                self._send_error_json(404, "Not found")
                return
            if inference_service is None:
                self._send_error_json(503, inference_startup_error or "推理服务不可用")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_error_json(400, "Content-Length 无效")
                return
            if content_length <= 0 or content_length > 65_536:
                self._send_error_json(400, "请求体为空或过大")
                return
            try:
                request = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("JSON 顶层必须是对象")
                payload = inference_service.predict(
                    str(request.get("model_package_id") or ""),
                    int(request.get("shot")),
                    float(request.get("history_end_ms")),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, InferenceError) as error:
                self._send_error_json(400, str(error))
                return
            self._send_json(payload)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[viewer] {self.address_string()} - {format % args}")

    return ResultViewerHandler


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    config = load_config(config_path)
    results_root = resolve_results_root(config, args.results_root)
    inference_config = config.get("inference", {})
    model_registry_root = resolve_project_path(
        args.model_registry or inference_config.get("model_registry"),
        "model_registry",
    )
    data_cache = resolve_project_path(
        args.data_cache or inference_config.get("data_cache"),
        "Data/cache",
    )
    code_root_value = inference_config.get("code_root")
    code_root = resolve_project_path(str(code_root_value), ".") if code_root_value else None
    catalog = scan_result_catalog(results_root)
    packages = scan_model_registry(model_registry_root)
    registered_models_only = bool(
        config.get("selection", {}).get("registered_models_only", False)
    )
    package_payload = [package.to_dict() for package in packages]
    option_payloads: list[dict[str, Any]] = []
    for option in catalog:
        payload = option.to_dict()
        package = package_for_result_option(payload, packages, results_root)
        if registered_models_only and package is None:
            continue
        payload["model_package_id"] = package.package_id if package else None
        payload["prediction_sampling_hz"] = (
            int(
                package.manifest.get(
                    "model_sampling_hz", package.manifest.get("sampling_hz")
                )
            )
            if package
            else None
        )
        payload["inference_ready"] = bool(package and package.ready)
        payload["inference_status"] = (
            "可实时推理"
            if package and package.ready
            else ("；".join(package.errors) if package else "未安装对应模型包")
        )
        option_payloads.append(payload)
    matched_package_ids = {
        str(option.get("model_package_id"))
        for option in option_payloads
        if option.get("model_package_id")
    }
    for package in packages:
        if package.package_id in matched_package_ids:
            continue
        package_option = package_catalog_option(package)
        if package_option is not None:
            option_payloads.append(package_option)
    payload = {
        "results_root": str(results_root),
        "history_ms": 160,
        "options": option_payloads,
    }
    inference_service: ModelInferenceService | None = None
    inference_startup_error: str | None = None
    try:
        inference_service = ModelInferenceService(
            packages=packages,
            cache_dir=data_cache,
            code_root=code_root,
            device=str(inference_config.get("device") or "cpu"),
        )
    except InferenceError as error:
        inference_startup_error = str(error)
    model_payload = {
        "model_registry": str(model_registry_root),
        "data_cache": str(data_cache),
        "packages": package_payload,
        "service_ready": inference_service is not None,
        "service_error": inference_startup_error,
    }
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(payload, model_payload, inference_service, inference_startup_error),
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"Result viewer ready: {url}")
    print(f"Read-only results root: {results_root}")
    print(f"Discovered result options: {len(catalog)} ({len(option_payloads)} visible)")
    print(f"Discovered model packages: {len(packages)}")
    print(
        f"Inference service: {'ready' if inference_service else inference_startup_error}"
    )
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping result viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
