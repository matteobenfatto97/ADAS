# Patch 3 - Premium reactive HUD

Questa patch sostituisce la vecchia UI debug con un HUD più moderno e reattivo disegnato direttamente sul frame video con OpenCV.

## Cosa cambia

- Top bar a card traslucide invece del rettangolo nero pieno.
- Stato ADAS colorato:
  - `ACTIVE` verde/cyan;
  - `WARN` ambra;
  - `BRAKE` rosso pulsante.
- Lane corridor più pulito con:
  - corsia completa semitrasparente;
  - ego corridor cyan;
  - linee corsia gialle con glow;
  - frecce/chevron prospettiche;
  - indicatore offset centrale.
- Detection box più pulite con corner-box e chip label.
- Card inferiori per modalità simulata e rischio collisione.
- FPS mostrati nel pannello sistema.

## Comandi

HUD premium:

```powershell
python -m adas.main --config configs/dashcam_fast.yaml --source "data/video_strada.mp4" --model models/yolov8n.pt --fast
```

Vecchia UI debug:

```powershell
python -m adas.main --config configs/dashcam_fast.yaml --source "data/video_strada.mp4" --model models/yolov8n.pt --classic-ui
```

HUD premium con riga debug:

```powershell
python -m adas.main --config configs/dashcam_fast.yaml --source "data/video_strada.mp4" --model models/yolov8n.pt --hud-debug
```

## Parametri utili

Nel file `configs/dashcam_fast.yaml`:

```yaml
dashboard:
  enabled: true
  compact: false
  show_fps: true
  show_debug_text: false
  panel_alpha: 0.62
  road_alpha: 0.22
  ego_alpha: 0.26
  glow: true
  speed_kmh: null
  speed_limit_kmh: null
```

Per più prestazioni:

```yaml
dashboard:
  compact: true
  glow: false
  panel_alpha: 0.55
```

Per mostrare velocità fissa nel mockup:

```yaml
dashboard:
  speed_kmh: 58
  speed_limit_kmh: 60
```

Attenzione: questa velocità è solo grafica. Per leggerla davvero servono OBD-II, GPS, OCR dalla dashcam o metadati del video.
