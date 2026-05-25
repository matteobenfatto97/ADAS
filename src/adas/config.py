from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config non trovata: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAS YOLOv8 prototype")
    parser.add_argument("--config", default="configs/default.yaml", help="Percorso YAML di configurazione")
    parser.add_argument("--source", default=None, help="Webcam index o video path. Sovrascrive video.source")
    parser.add_argument("--model", default=None, help="Percorso modello .pt. Sovrascrive model.path")
    parser.add_argument("--no-display", action="store_true", help="Non mostrare la finestra OpenCV")
    parser.add_argument("--save-output", action="store_true", help="Salva il video elaborato")
    parser.add_argument("--output", default=None, help="Percorso output video, es. outputs/demo.mp4")
    parser.add_argument("--no-track", action="store_true", help="Disattiva tracking e usa solo predict()")
    parser.add_argument("--fast", action="store_true", help="Modalità veloce: inferenza più leggera")
    parser.add_argument("--turbo", action="store_true", help="Modalità real-time: YOLO asincrono, lane leggera, HUD senza glow")
    parser.add_argument("--gpu", action="store_true", help="Usa CUDA device 0 + half precision, se disponibile")
    parser.add_argument("--intel-gpu", action="store_true", help="Usa OpenVINO su Intel GPU/iGPU. Richiede modello *_openvino_model")
    parser.add_argument("--intel-cpu", action="store_true", help="Usa OpenVINO su Intel CPU. Richiede modello *_openvino_model")
    parser.add_argument("--no-async", action="store_true", help="Disattiva inferenza YOLO asincrona")
    parser.add_argument("--disable-yolo", action="store_true", help="Disattiva YOLO: utile per testare solo lane/HUD FPS")
    parser.add_argument("--show-filtered", action="store_true", help="Mostra anche oggetti esclusi dai filtri rischio")
    parser.add_argument("--classic-ui", action="store_true", help="Usa la vecchia interfaccia debug minimale")
    parser.add_argument("--hud-debug", action="store_true", help="Mostra una piccola riga debug sopra al nuovo HUD")
    parser.add_argument("--lite-ui", action="store_true", help="HUD performante: meno effetti, più FPS, più adatto a demo live")
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.source is not None:
        source: Any = args.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cfg.setdefault("video", {})["source"] = source
    if args.model is not None:
        cfg.setdefault("model", {})["path"] = args.model
    if args.no_display:
        cfg.setdefault("video", {})["display"] = False
    if args.save_output:
        cfg.setdefault("video", {})["save_output"] = True
    if args.output is not None:
        cfg.setdefault("video", {})["output_path"] = args.output
    if args.no_track:
        cfg.setdefault("model", {})["enable_tracking"] = False
    if args.no_async:
        cfg.setdefault("model", {})["async_inference"] = False
    if args.disable_yolo:
        cfg.setdefault("model", {})["enabled"] = False
    if args.gpu:
        cfg.setdefault("model", {})["device"] = "0"
        cfg.setdefault("model", {})["half"] = True
    if getattr(args, "intel_gpu", False):
        cfg.setdefault("model", {})["device"] = "intel:gpu"
        cfg.setdefault("model", {})["half"] = False
        cfg.setdefault("model", {})["enable_tracking"] = False
        cfg.setdefault("model", {})["async_inference"] = True
    if getattr(args, "intel_cpu", False):
        cfg.setdefault("model", {})["device"] = "intel:cpu"
        cfg.setdefault("model", {})["half"] = False
        cfg.setdefault("model", {})["enable_tracking"] = False
        cfg.setdefault("model", {})["async_inference"] = True
    if args.show_filtered:
        cfg.setdefault("debug", {})["show_filtered"] = True
    if args.classic_ui:
        cfg.setdefault("dashboard", {})["enabled"] = False
    if args.hud_debug:
        cfg.setdefault("dashboard", {})["show_debug_text"] = True
    if getattr(args, "lite_ui", False):
        cfg.setdefault("dashboard", {})["render_mode"] = "performance"
        cfg.setdefault("dashboard", {})["glow"] = False
        cfg.setdefault("dashboard", {})["vignette"] = False
        cfg.setdefault("dashboard", {})["micro_grid"] = False
        cfg.setdefault("dashboard", {})["compact"] = True
        cfg.setdefault("dashboard", {})["render_mode"] = "performance"
    if args.fast:
        cfg.setdefault("model", {})["imgsz"] = min(int(cfg.get("model", {}).get("imgsz", 416)), 416)
        cfg.setdefault("video", {})["inference_width"] = min(int(cfg.get("video", {}).get("inference_width", 736)), 640)
        cfg.setdefault("video", {})["inference_every_n_frames"] = max(int(cfg.get("video", {}).get("inference_every_n_frames", 1)), 3)
        cfg.setdefault("video", {})["lane_every_n_frames"] = max(int(cfg.get("video", {}).get("lane_every_n_frames", 1)), 2)
    if args.turbo:
        cfg.setdefault("model", {})["async_inference"] = True
        cfg.setdefault("model", {})["enable_tracking"] = False
        cfg.setdefault("model", {})["imgsz"] = min(int(cfg.get("model", {}).get("imgsz", 320)), 256)
        cfg.setdefault("model", {})["max_det"] = min(int(cfg.get("model", {}).get("max_det", 12)), 8)
        cfg.setdefault("video", {})["inference_width"] = min(int(cfg.get("video", {}).get("inference_width", 512)), 416)
        cfg.setdefault("video", {})["inference_every_n_frames"] = max(int(cfg.get("video", {}).get("inference_every_n_frames", 6)), 8)
        cfg.setdefault("video", {})["lane_every_n_frames"] = max(int(cfg.get("video", {}).get("lane_every_n_frames", 2)), 2)
        cfg.setdefault("video", {})["target_display_fps"] = int(cfg.get("video", {}).get("target_display_fps", 60))
        cfg.setdefault("dashboard", {})["glow"] = False
        cfg.setdefault("dashboard", {})["vignette"] = False
        cfg.setdefault("dashboard", {})["micro_grid"] = False
        cfg.setdefault("dashboard", {})["compact"] = True
        cfg.setdefault("dashboard", {})["render_mode"] = "performance"
    return cfg
