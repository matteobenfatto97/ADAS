# Patch 2 — Dashcam realism

Problemi emersi dal test dashcam:

1. Video troppo lento.
2. Cartelli scambiati per persone.
3. Auto nell'altra corsia considerate rischio collisione.
4. Lane departure non rilevato bene.

Correzioni introdotte:

- `inference_width`: YOLO lavora su una copia ridotta del frame.
- `inference_every_n_frames`: puoi eseguire YOLO ogni 2 frame.
- `target_classes` vehicle-only di default.
- `risk_classes` separato: puoi rilevare un oggetto ma non usarlo per frenata.
- `ego_polygon`: corridoio centrale della corsia usato per il rischio collisione.
- `require_lane_for_brake`: se non vedo la corsia, non simulo emergency brake.
- `vehicle_center_ratio`: taratura posizione dashcam.
- `departure_offset_ratio`: sensibilità uscita corsia.
- overlay con stato lane e offset numerico.

Comando consigliato:

```bash
python -m adas.main --config configs/dashcam_fast.yaml --source data/video_strada.mp4 --model models/yolov8n.pt --fast
```

Debug utile:

```bash
python -m adas.main --source data/video_strada.mp4 --model models/yolov8n.pt --show-filtered
```
