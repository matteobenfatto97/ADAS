# Patch 4 — Real-time + curved lane detection

Questa patch nasce da quattro problemi reali emersi sul video dashcam:

1. Il video andava a circa 3 FPS.
2. La lane confidence restava sempre bassa/rossa.
3. Il lane departure non veniva segnalato anche invadendo la linea.
4. Le linee erano sempre dritte e non seguivano l'andamento della strada.

## Cosa cambia

### 1. YOLO asincrono

YOLO non blocca più il loop video. Il main thread continua a mostrare il video e il worker YOLO elabora solo il frame più recente.

File nuovo:

```text
src/adas/perception/async_detector.py
```

Preset consigliato:

```powershell
python -m adas.main --config configs/realtime_60.yaml --source "data/video_strada.mp4" --model models/yolov8n.pt --turbo
```

Su CPU, YOLO non calcolerà davvero 60 inferenze al secondo: la UI può però restare fluida perché riusa l'ultima detection disponibile.

### 2. Lane detector curvo

Il vecchio Hough detector generava due rette. Ora il detector usa:

- maschera bianco/giallo + Sobel;
- ROI trapezoidale;
- histogram search;
- sliding windows;
- fit polinomiale `x = ay² + by + c`.

File modificato:

```text
src/adas/perception/lane_detector.py
```

Ora l'HUD può seguire curve e strada reale.

### 3. Lane departure più sensibile

Nuovi parametri:

```yaml
lane:
  departure_offset_ratio: 0.12
  line_invasion_margin_ratio: 0.060
  y_eval_ratio: 0.86
```

`departure_offset_ratio` controlla quanto puoi decentrarti prima dell'avviso.
`line_invasion_margin_ratio` segnala anche quando ti avvicini troppo alla linea, non solo quando il centro veicolo è molto fuori asse.
`y_eval_ratio` indica a che altezza del frame valutare la posizione laterale, evitando il cofano.

### 4. HUD più leggero

Per il preset real-time sono disattivati glow, vignettatura e micro-grid:

```yaml
dashboard:
  glow: false
  vignette: false
  micro_grid: false
```

Puoi riattivarli quando vuoi fare una demo grafica registrata, ma per guida fluida conviene tenerli spenti.

## Preset principali

### Real-time CPU

```powershell
python -m adas.main --config configs/realtime_60.yaml --source "data/video_strada.mp4" --model models/yolov8n.pt --turbo
```

### Solo lane/HUD, per misurare FPS senza YOLO

```powershell
python -m adas.main --config configs/realtime_60.yaml --source "data/video_strada.mp4" --disable-yolo
```

Se questo gira fluido, il collo di bottiglia è YOLO.

### GPU NVIDIA

```powershell
python -m adas.main --config configs/realtime_60.yaml --source "data/video_strada.mp4" --model models/yolov8n.pt --gpu --turbo
```

Richiede PyTorch con CUDA installato correttamente.

## Nota onesta sui 60 FPS

Con `yolov8n.pt` su CPU non è realistico pretendere 60 inferenze YOLO al secondo su video HD. La soluzione corretta è separare:

- FPS di visualizzazione: 60 FPS target;
- FPS di YOLO: aggiornamento ogni N frame, asincrono;
- FPS lane: ogni 1-2 frame, molto più leggero.

Questo è anche il modo corretto di ragionare in un sistema ADAS: non tutti i moduli hanno la stessa frequenza.
