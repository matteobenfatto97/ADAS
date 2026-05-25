from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from adas.perception.geometry import scale_detections
from adas.perception.object_detector import ObjectDetector
from adas.perception.types import Detection


@dataclass
class AsyncDetectionResult:
    frame_id: int
    detections: List[Detection]
    frame_shape: Tuple[int, int]


class AsyncObjectDetector:
    """Esegue YOLO in un thread separato.

    La UI non deve aspettare YOLO. Il loop video mostra sempre l'ultimo stato disponibile,
    mentre il worker elabora solo il frame piu' recente. Se il PC e' lento, i frame vecchi
    vengono scartati invece di bloccare la riproduzione a 3 FPS.
    """

    def __init__(self, detector: ObjectDetector) -> None:
        self.detector = detector
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: Optional[AsyncDetectionResult] = None
        self._stop = threading.Event()
        self._error_count = 0
        self._last_error_message = ""
        self._cuda_fallback_done = False
        self._thread = threading.Thread(target=self._run, name="ADAS-YOLO-worker", daemon=True)
        self._thread.start()

    def submit(self, frame, *, frame_id: int, scale_x: float, scale_y: float, original_shape: Tuple[int, int]) -> None:
        job = (frame.copy(), frame_id, scale_x, scale_y, original_shape)
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            try:
                self._jobs.get_nowait()  # drop frame vecchio
            except queue.Empty:
                pass
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                pass

    def latest(self) -> Optional[AsyncDetectionResult]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=1.5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                continue
            frame, frame_id, sx, sy, original_shape = job
            try:
                small_detections = self.detector.detect(frame)
                detections = scale_detections(small_detections, sx, sy)
                result = AsyncDetectionResult(frame_id=frame_id, detections=detections, frame_shape=original_shape)
                with self._lock:
                    self._latest = result
            except Exception as exc:
                msg = str(exc)
                self._error_count += 1

                # If CUDA was requested but torch is CPU-only, switch once to CPU and stop spamming.
                if (not self._cuda_fallback_done) and ("Invalid CUDA" in msg or "cuda" in msg.lower()):
                    self._cuda_fallback_done = True
                    try:
                        self.detector.fallback_to_cpu()
                        print("[PERF] YOLO: CUDA non disponibile nel torch installato. Passo a CPU e continuo.")
                    except Exception:
                        pass
                    continue

                # Print first error and then only occasional summaries.
                if msg != self._last_error_message or self._error_count in {1, 10, 50, 100}:
                    print(f"[WARN] YOLO async failed ({self._error_count}): {msg}")
                    self._last_error_message = msg
