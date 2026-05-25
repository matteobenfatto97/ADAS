from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO .pt to OpenVINO for Intel Iris Xe / Intel CPU")
    parser.add_argument("--model", default="models/yolov8n.pt", help="Path del modello .pt")
    parser.add_argument("--imgsz", type=int, default=416, help="Dimensione input export")
    parser.add_argument("--half", action="store_true", help="Export FP16. Consigliato per Intel GPU se supportato")
    parser.add_argument("--int8", action="store_true", help="Export INT8. Richiede dataset/calibrazione; non consigliato per il primo test")
    parser.add_argument("--data", default="coco8.yaml", help="Dataset yaml per calibrazione INT8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    from ultralytics import YOLO

    print(f"[EXPORT] Carico modello: {model_path}")
    model = YOLO(str(model_path))

    export_kwargs = dict(format="openvino", imgsz=args.imgsz)
    if args.half:
        export_kwargs["half"] = True
    if args.int8:
        export_kwargs["int8"] = True
        export_kwargs["data"] = args.data

    print(f"[EXPORT] Esporto OpenVINO con parametri: {export_kwargs}")
    out = model.export(**export_kwargs)
    print(f"[EXPORT] Completato: {out}")
    print("[NEXT] Test dispositivi OpenVINO:")
    print("       python scripts/check_accelerators.py")
    print("[NEXT] Avvio ADAS su Intel GPU:")
    print("       python -m adas.main --config configs/realtime_iris_xe.yaml --source data/dashcam1.mp4")


if __name__ == "__main__":
    main()
