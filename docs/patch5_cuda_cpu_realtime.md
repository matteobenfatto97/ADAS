# Patch 5 - CUDA fallback, CPU real-time preset e export demo

Questa patch risolve il problema visto nel log:

- `torch-...+cpu`
- `torch.cuda.is_available(): False`
- `Invalid CUDA device=0 requested`

Se l'utente lancia `--gpu` ma PyTorch è CPU-only, il sistema ora fa fallback automatico a CPU e non stampa più lo stesso errore a ogni frame.

## Comandi consigliati

### Test CPU live

```powershell
python -m adas.main --config configs/realtime_cpu.yaml --source "data/dashcam1.mp4" --model models/yolov8n.pt
```

### Test lane/HUD, senza YOLO

```powershell
python -m adas.main --config configs/realtime_cpu.yaml --source "data/dashcam1.mp4" --disable-yolo
```

### Video pubblicabile online

```powershell
python -m adas.main --config configs/demo_export.yaml --source "data/dashcam1.mp4" --model models/yolov8n.pt --save-output --no-display --output outputs/adas_demo.mp4
```

L'export può essere più lento del real-time, ma il file finale viene scritto con l'FPS del video originale e quindi risulta fluido in riproduzione.

## Per usare davvero CUDA

Controlla prima:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Se `torch.cuda.is_available()` è `False`, non usare `--gpu`. Serve installare un build PyTorch con CUDA compatibile con la tua versione Python/GPU.
