from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2


def open_video_source(source, width: Optional[int] = None, height: Optional[int] = None):
    """Apre webcam o file video con messaggi di errore più utili."""
    original_source = source

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, str):
        path = Path(source).expanduser()
        if not path.exists():
            raise RuntimeError(
                "Impossibile aprire sorgente video: "
                f"{original_source}\n"
                "Il file non esiste nel percorso indicato. Esempi validi:\n"
                "  python -m adas.main --source 0 --model models/yolov8n.pt\n"
                "  python -m adas.main --source data/video_strada.mp4 --model models/yolov8n.pt\n"
                "  python -m adas.main --source C:/Users/Matteo/Desktop/video_strada.mp4 --model models/yolov8n.pt"
            )
        source = str(path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire sorgente video: {original_source}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap


def create_writer(path: str, fps: float, frame_size: Tuple[int, int]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, frame_size)
