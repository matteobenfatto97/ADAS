from __future__ import annotations

import time
from typing import Optional

import cv2


def resolve_torch_device(model_cfg: dict) -> None:
    """Resolve CUDA/CPU before Ultralytics is called.

    If the user passes --gpu while the installed torch build is CPU-only,
    Ultralytics would raise the same CUDA exception at every async frame.
    Here we fail soft once and continue on CPU.
    """
    device = model_cfg.get("device")
    if device is None:
        return
    device_s = str(device).lower().strip()
    if device_s.startswith("intel:"):
        # OpenVINO backend: non passa da torch.cuda. Esempi: intel:gpu, intel:cpu.
        model_cfg["half"] = False
        model_cfg["enable_tracking"] = False
        return
    wants_cuda = device_s in {"0", "cuda", "cuda:0"} or device_s.isdigit()
    if not wants_cuda:
        if device_s == "cpu":
            model_cfg["half"] = False
        return

    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
        torch_version = getattr(torch, "__version__", "unknown")
    except Exception as exc:
        cuda_ok = False
        torch_version = f"unknown ({exc})"

    if not cuda_ok:
        print(
            "[PERF] CUDA richiesta, ma PyTorch non vede GPU CUDA. "
            f"Torch={torch_version}. Fallback automatico su CPU. "
            "Per ora NON usare --gpu in questo ambiente."
        )
        model_cfg["device"] = "cpu"
        model_cfg["half"] = False
    else:
        model_cfg["device"] = "0"
        model_cfg["half"] = bool(model_cfg.get("half", True))


from adas.config import apply_overrides, load_config, parse_args
from adas.control.adas_controller import ADASController
from adas.perception.async_detector import AsyncObjectDetector
from adas.perception.geometry import resize_keep_aspect, scale_detections
from adas.perception.lane_detector import LaneDetector
from adas.perception.object_detector import ObjectDetector
from adas.perception.types import LaneState
from adas.risk.collision_predictor import CollisionPredictor
from adas.utils.video import create_writer, open_video_source
from adas.visualization.overlay import PremiumHUD


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    model_cfg = cfg.get("model", {})
    model_enabled = bool(model_cfg.get("enabled", True))
    if model_enabled:
        resolve_torch_device(model_cfg)
    detector: Optional[ObjectDetector] = None
    async_detector: Optional[AsyncObjectDetector] = None

    if model_enabled:
        detector = ObjectDetector(
            model_path=model_cfg["path"],
            confidence=model_cfg.get("confidence", 0.45),
            iou=model_cfg.get("iou", 0.5),
            imgsz=model_cfg.get("imgsz", 416),
            device=model_cfg.get("device"),
            target_classes=model_cfg.get("target_classes"),
            tracker=model_cfg.get("tracker", "bytetrack.yaml"),
            enable_tracking=model_cfg.get("enable_tracking", True),
            half=model_cfg.get("half", False),
            max_det=model_cfg.get("max_det", 12),
        )
        if bool(model_cfg.get("async_inference", False)):
            async_detector = AsyncObjectDetector(detector)

    lane_detector = LaneDetector(cfg.get("lane", {}))
    predictor = CollisionPredictor(cfg.get("risk", {}))
    controller = ADASController(cfg.get("control", {}))
    hud = PremiumHUD(cfg.get("dashboard", {}))

    video_cfg = cfg.get("video", {})
    cap = open_video_source(video_cfg.get("source", 0), video_cfg.get("width"), video_cfg.get("height"))

    writer = None
    if video_cfg.get("save_output", False):
        fps = cap.get(cv2.CAP_PROP_FPS) or float(video_cfg.get("target_display_fps", 30))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = create_writer(video_cfg.get("output_path", "outputs/adas_output.mp4"), fps, (width, height))

    print("ADAS avviato. Premi Q per uscire.")
    print("Nota: la frenata è simulata. Non collegare questo prototipo a un veicolo reale.")
    if async_detector is not None:
        print("[PERF] YOLO asincrono attivo: la UI non aspetta l'inferenza.")
    if model_enabled and str(model_cfg.get("device", "")).lower().startswith("intel:"):
        print(f"[PERF] OpenVINO attivo su {model_cfg.get('device')}. Tracking disattivato per stabilità.")
    if not model_enabled:
        print("[PERF] YOLO disattivato: test lane/HUD only.")

    frame_idx = 0
    log_every_n_frames = int(video_cfg.get("log_every_n_frames", 30))
    inference_width = int(video_cfg.get("inference_width", 640))
    inference_every_n_frames = max(1, int(video_cfg.get("inference_every_n_frames", 1)))
    lane_every_n_frames = max(1, int(video_cfg.get("lane_every_n_frames", 1)))
    target_display_fps = float(video_cfg.get("target_display_fps", 0) or 0)
    display_width = video_cfg.get("display_width")
    show_filtered = bool(cfg.get("debug", {}).get("show_filtered", False))

    last_detections = []
    last_lane_state = LaneState()
    last_yolo_result_id = -1
    last_time = time.monotonic()
    last_display_time = time.monotonic()
    fps_smooth = 0.0

    try:
        # OpenCV non deve creare troppi thread: su alcuni PC peggiora la latenza.
        try:
            cv2.setNumThreads(int(video_cfg.get("opencv_threads", 1)))
        except Exception:
            pass

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % lane_every_n_frames == 0:
                last_lane_state = lane_detector.detect(frame)
            lane_state = last_lane_state

            if model_enabled and detector is not None and frame_idx % inference_every_n_frames == 0:
                infer_frame, sx, sy = resize_keep_aspect(frame, inference_width)
                if async_detector is not None:
                    async_detector.submit(
                        infer_frame,
                        frame_id=frame_idx,
                        scale_x=sx,
                        scale_y=sy,
                        original_shape=(frame.shape[0], frame.shape[1]),
                    )
                else:
                    detections_small = detector.detect(infer_frame)
                    detections = scale_detections(detections_small, sx, sy)
                    last_detections = predictor.update(
                        detections,
                        lane_state,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )

            if async_detector is not None:
                result = async_detector.latest()
                if result is not None and result.frame_id != last_yolo_result_id:
                    last_yolo_result_id = result.frame_id
                    last_detections = predictor.update(
                        result.detections,
                        lane_state,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )

            detections = last_detections
            decision = controller.decide(detections, lane_state)

            now = time.monotonic()
            dt = max(1e-6, now - last_time)
            instant_fps = 1.0 / dt
            fps_smooth = instant_fps if fps_smooth <= 0 else (fps_smooth * 0.90 + instant_fps * 0.10)
            last_time = now

            output = hud.draw(
                frame,
                detections,
                lane_state,
                decision,
                show_filtered=show_filtered,
                fps=fps_smooth,
                frame_idx=frame_idx,
            )

            if frame_idx % max(log_every_n_frames, 1) == 0:
                if decision.brake:
                    print(f"[BRAKE SIM] {decision.reason}")
                elif decision.warnings:
                    print(f"[WARN] {' | '.join(decision.warnings)}")

            if writer is not None:
                writer.write(output)

            if video_cfg.get("display", True):
                display_frame = output
                if display_width:
                    dw = int(display_width)
                    h, w = display_frame.shape[:2]
                    if w > dw:
                        dh = int(h * (dw / float(w)))
                        display_frame = cv2.resize(display_frame, (dw, dh), interpolation=cv2.INTER_AREA)

                cv2.imshow("ADAS YOLOv8 Prototype", display_frame)

                wait_ms = 1
                if target_display_fps > 0:
                    target_dt = 1.0 / target_display_fps
                    elapsed = time.monotonic() - last_display_time
                    wait_ms = max(1, int(max(0.0, target_dt - elapsed) * 1000))
                    last_display_time = time.monotonic()
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    except KeyboardInterrupt:
        print("\nChiusura richiesta da tastiera. Risorse rilasciate correttamente.")
    finally:
        if async_detector is not None:
            async_detector.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
