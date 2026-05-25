# Patch 7 - Showcase Fix

Questa patch corregge i problemi emersi nel test reale:

- HUD performance mode: meno overlay full-frame, meno glow, meno copie immagine.
- Preset Intel Iris Xe corretti: la Patch 6 usava alcune chiavi lane vecchie, ora sono allineate al LaneDetector polinomiale.
- Collision detection più visibile: aggiunto proximity risk oltre al TTC visuale.
- Matching detection senza track_id: OpenVINO predict non fornisce tracking stabile, quindi ora c'è matching per IoU/centro.
- Lane departure più sensibile: usa un punto più basso vicino al veicolo per rilevare invasione linea.

Comandi consigliati:

```powershell
python scripts/export_openvino.py --model models/yolov8n.pt --half --imgsz 320
python -m adas.main --config configs/live_showcase_iris_xe.yaml --source "data/dashcam1.mp4" --lite-ui
```

Export demo pubblicabile:

```powershell
python -m adas.main --config configs/openvino_export_demo.yaml --source "data/dashcam1.mp4" --save-output --no-display --output outputs/adas_demo_openvino.mp4
```
